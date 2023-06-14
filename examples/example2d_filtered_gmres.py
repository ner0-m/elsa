import numpy as np
import pyelsa as elsa
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft


# currently the elsa implementation of gmres can not apply an additional filter during reconstruction
# a custom python implementation is needed to apply GMRES with a filter

# ramp_filter and filter_sinogram implemented like a standard FBP implementation
# see https://ciip.in.tum.de/elsadocs/guides/python_guide/filtered_backprojection.html
def ramp_filter(size):
    n = np.concatenate(
        (
            # increasing range from 1 to size/2, and again down to 1, step size 2
            np.arange(1, size / 2 + 1, 2, dtype=int),
            np.arange(size / 2 - 1, 0, -2, dtype=int),
        )
    )
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # See "Principles of Computerized Tomographic Imaging" by Avinash C. Kak and Malcolm Slaney,
    # Chap 3. Equation 61, for a detailed description of these steps
    return 2 * np.real(fft(f))[:, np.newaxis]


def filter_sinogram(sinogram):
    np_sinogram = np.asarray(sinogram)
    sinogram_shape = np_sinogram.shape[0]

    # Add padding
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * sinogram_shape))))
    pad_width = ((0, projection_size_padded - sinogram_shape), (0, 0))
    padded_sinogram = np.pad(np_sinogram, pad_width, mode="constant", constant_values=0)

    # Ramp filter
    fourier_filter = ramp_filter(projection_size_padded)

    projection = fft(padded_sinogram, axis=0) * fourier_filter
    filtered_sinogram = np.real(ifft(projection, axis=0)[:sinogram_shape, :])

    filtered_sinogram = elsa.DataContainer(filtered_sinogram, sinogram.getDataDescriptor())

    return filtered_sinogram


def apply(A, x, filter=False):
    if not filter:
        if isinstance(A, (list, np.ndarray, np.matrix)):
            return np.dot(A, x)
        else:
            return A.apply(x)
    else:
        return A.apply(filter_sinogram(x))


# specific GMRES functions, which are different for AB- or BA-GMRES
def calc_r0(ABGMRES, A, B, b, x, filter):
    if ABGMRES:
        r0 = np.asarray(b).reshape(-1) - np.asarray(apply(A, x)).reshape(-1)
        r0 = elsa.DataContainer(np.reshape(r0, np.shape(np.asarray(b))), b.getDataDescriptor())
        return r0, np.shape(np.asarray(b))
    else:
        r0 = np.asarray(apply(B, b, filter=filter)).reshape(-1) - np.asarray(
            apply(B, apply(A, x), filter=filter)).reshape(-1)
        r0 = elsa.DataContainer(np.reshape(r0, np.shape(np.asarray(x))), x.getDataDescriptor())
        return r0, np.shape(np.asarray(x))


def calc_q(ABGMRES, A, B, wk, filter):
    if ABGMRES:
        return np.asarray(apply(A, apply(B, wk, filter=filter))).reshape(-1)
    else:
        return np.asarray(apply(B, apply(A, wk), filter=filter)).reshape(-1)


def calc_x(ABGMRES, B, x, wy, filter):
    if ABGMRES:
        return np.asarray(x) + np.asarray(apply(B, wy, filter=filter))
    else:
        return np.asarray(x) + wy


# GMRES Implementation
def GMRES(A, B, b, x0, nmax_iter, epsilon=None, ABGMRES=True, filter=False):
    # --- 1. Instantiate ---

    # r0 = b - Ax
    r0, r0_shape = calc_r0(ABGMRES=ABGMRES, A=A, B=B, b=b, x=x0, filter=filter)

    h = np.zeros((nmax_iter + 1, nmax_iter))
    w = [np.zeros(r0_shape[0] * r0_shape[1])] * nmax_iter
    e = np.zeros(nmax_iter + 1)
    y = [0] * nmax_iter

    e[0] = np.linalg.norm(r0)

    w[0] = np.asarray(r0).reshape(-1) / np.linalg.norm(r0)

    # --- 2. Iterate ---
    for k in range(nmax_iter):

        # q = ABw_k
        q = calc_q(ABGMRES=ABGMRES, A=A, B=B, wk=elsa.DataContainer(np.reshape(w[k], r0_shape), r0.getDataDescriptor()),
                   filter=filter)

        for i in range(k + 1):
            h[i, k] = apply(q.T, w[i])
            q = q - h[i, k] * w[i]

        h[k + 1, k] = np.linalg.norm(q)

        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            w[k + 1] = q / h[k + 1, k]

        # Solving minimization problem using numpy leastsquares
        y = np.linalg.lstsq(h, e, rcond=None)[0]

        # transforming list of vectors to a matrix
        w_copy = np.reshape(np.asarray(w), (nmax_iter, len(w[0]))).T

        # applying estimated guess from our generated krylov subspace to our initial guess x0
        x = calc_x(ABGMRES=ABGMRES, B=B, x=x0,
                   wy=elsa.DataContainer(np.reshape(np.dot(w_copy, y), r0_shape), r0.getDataDescriptor()),
                   filter=filter)

    return x


def example2d_gmres_filter(s: int):
    size = np.array([s] * 2)
    phantom = elsa.phantoms.modifiedSheppLogan(size)

    distance = size[0]
    sino_descriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        np.arange(0, 180, 1), phantom.getDataDescriptor(), distance * 100.0, distance
    )

    # creates the projector
    projector = elsa.JosephsMethod(phantom.getDataDescriptor(), sino_descriptor)
    # and the unmatched backprojector
    backprojector = elsa.adjoint(projector)

    # custom x0 datacontainer needed for a python implementation
    x0 = elsa.DataContainer(np.zeros_like(np.asarray(phantom)), phantom.getDataDescriptor())

    sinogram = projector.apply(phantom)

    # ABGMRES=True -> use ABGMRES | ABGMRES=False -> use BAGMRES
    # filter=True -> apply filter to backprojector
    reconstruction = GMRES(A=projector, B=backprojector, b=sinogram, x0=x0, nmax_iter=2, ABGMRES=False, filter=True)

    plt.imshow(reconstruction)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--size", default=128, type=int, help="size of reconstruction")

    args = parser.parse_args()
    example2d_gmres_filter(args.size)
