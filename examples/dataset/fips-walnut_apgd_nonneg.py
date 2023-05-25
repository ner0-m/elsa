"""This is a reconstruction of the walnut provided by the Finnish Inverse
Problems Society.

The DOI of the data is 10.5281/zenodo.6986012. It can be found at
https://zenodo.org/record/6986012. The run the script you need to download
the zip files and extract them in a common folder.

The walnut data is a well scanned reconstruction with a lot of data. Little
preprocessing is needed, with the expection of some clean up and the
log-likelyhood transform.

In this example, the constrained optimization problem with the least squares
data term with a non-negativity constraint is solved using APGD/FISTA.

With small modifications, the scripts is usable for the seashell and and pine
cone dataset also provided by the Finnish Inverse Problems Society. They
can be found with the DOIs 10.5281/zenodo.6983008 and 10.5281/zenodo.6985407
respectively.
"""

import argparse
from pathlib import Path
import math

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import pyelsa as elsa
import tifffile
import tqdm

# Taken from the txt file provided with the data
sourceToDetector = 553.74
sourceToOrigin = 210.66
originToDetector = sourceToDetector - sourceToOrigin

detectorSpacing = np.asarray([0.050, 0.050])


def rebin2d(arr, binfac):
    """Rebin 2D array arr to shape new_shape by averaging.

    Credits to: https://scipython.com/blog/binning-a-2d-array-in-numpy/
    """

    nx, ny = arr.shape
    bnx, bny = nx // binfac, ny // binfac
    shape = (bnx, arr.shape[0] // bnx, bny, arr.shape[1] // bny)
    return arr.reshape(shape).mean(-1).mean(1)


def preprocess(proj, binning=None, padding=None, cor=0):
    """Preprocessing from transmission to absorption images including some cleaning and correction"""
    # Correct for slight misalignment
    proj = np.roll(proj, cor, axis=1)

    # Take a part of the background and compute an approximation of the
    # initial intensity
    background = proj[:128, :128]
    I0 = np.mean(background)

    # If binning should be performed, do it now
    if binning and binning != 1:
        proj = rebin2d(proj, binning)

    # reduce some noise
    proj[proj > I0] = I0

    # log-transform of projection and rebinning
    proj = -np.log(proj / I0)

    if padding:
        proj = np.pad(proj, padding)

    return proj


def load_dataset(path=Path("."), binning=1, step=1, padding=50):
    # Take all tif files in the path
    files = sorted(path.glob("20201111_walnut_*.tif"))

    projections = []
    # store angles of each projection we load.
    # This avoids computation later based on step
    angles = []

    for file in tqdm.tqdm(files[:-1:step], desc="Loading and preproecessing"):
        raw = tifffile.imread(file).astype("float32")

        # filename are 20201111_walnut_XXXX.tif, so split by '_', then by '.'
        filenum = int(str(file).rsplit("_", maxsplit=1)[-1].rsplit(".", maxsplit=1)[0])
        angle = (filenum - 1) / 2.0
        angles.append(angle)

        projections.append(preprocess(raw, binning=binning, padding=padding))

    # Final step of preprocessing
    projections /= np.max(projections)

    # Change to correct storage order for elsa
    projections = np.transpose(projections, (2, 1, 0))

    return projections, np.array(angles, dtype=np.float32)


def reconstruct(A, b, niters=70):
    """Just a basic reconstruction which looks nice"""
    prox = elsa.ProximalBoxConstraint(0)
    solver = elsa.APGD(A, b, prox)
    return solver.solve(niters)


def define_sino_descriptor(projections, binning, angles):
    num_angles = projections.shape[-1]

    assert len(angles) == num_angles

    sino_descriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        angles,
        volume_descriptor,
        sourceToOrigin,
        originToDetector,
        [0, 0],
        [0, 0, 0],  # Offset of origin
        projections.shape[:-1],
        detectorSpacing * binning,
    )

    return elsa.DataContainer(projections, sino_descriptor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="elsa Reconstruction of Walnut dataset"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to look for tif files",
    )
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=50,
        help="Number of iterations for the reconstruction algorithm",
    )
    parser.add_argument(
        "-b",
        "--binning",
        type=int,
        default=4,
        help="Binning factor, must be factor of two (default 4)",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        default=1,
        help="Take every n-th image only (default 1)",
    )
    parser.add_argument(
        "--padding", type=int, default=0, help="Add additional padding (default 0)"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1,
        help="Change the resolution of the reconstruction (default 1.0)",
    )

    args = parser.parse_args()

    if not elsa.cudaProjectorsEnabled():
        import warnings

        warnings.warn(
            "elsa is build without CUDA support. This reconstruction may take a long time."
        )

    binning = args.binning
    padding = args.padding
    resolution = args.resolution

    # load the dataset
    projections, angles = load_dataset(
        path=args.path, binning=binning, step=args.step, padding=padding
    )

    # Compute magnification
    magnification = sourceToDetector / sourceToOrigin

    # Determine good size and spacing for volume
    vol_spacing = (detectorSpacing[0] * binning) / (magnification * resolution)
    size = int(math.ceil((np.max(projections.shape[:-1]) - padding) * resolution))

    volume_descriptor = elsa.VolumeDescriptor([size] * 3, [vol_spacing] * 3)

    # define the sinogram for elsa
    sinogram = define_sino_descriptor(projections, binning, angles)
    sino_descriptor = sinogram.getDataDescriptor()

    # Set the forward and backward projector
    if elsa.cudaProjectorsEnabled():
        projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor)
    else:
        projector = elsa.JosephsMethod(volume_descriptor, sino_descriptor)

    # Perform reconstruction
    recon = reconstruct(projector, sinogram, args.iters)

    # Do a forward projection of the reconstruction. If this is close to the
    # original projections, the reconstruction should be good as well
    forward = projector.apply(recon)

    # Now just show everything...
    nprecon = np.array(recon)
    npforward = np.array(forward)

    # save reconstruction
    tifffile.imwrite(
        f"recon_fips_walnut_{args.iters}.tif",
        nprecon,
    )

    # And now visualize everything

    def add_colorbar(fig, ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

    fig = plt.figure(constrained_layout=True)

    fig1, fig2 = fig.subfigures(nrows=2, ncols=1)

    axs = fig1.subplots(nrows=1, ncols=2)
    for col, (ax, slice, title) in enumerate(
        zip(
            axs,
            [nprecon[:, size // 2, :], nprecon[:, :, size // 2]],
            ["Axial center slice", "Coronal center slice"],
        )
    ):
        im = ax.imshow(slice, cmap="gray")
        ax.set_title(title)
        # Hide ticks
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        add_colorbar(fig1, ax, im)

    fig2.suptitle(
        "Compare the forward projected reconstruction with the original projections\nFor a good reconstruction, the two should be close"
    )
    axs = fig2.subplots(nrows=1, ncols=2)
    projection = projections.shape[-1] // 2
    for col, (ax, slice, title) in enumerate(
        zip(
            axs,
            [npforward[:, :, projection], projections[:, :, projection]],
            ["Forward projected reconstruction", "Original projection"],
        )
    ):
        im = ax.imshow(slice, cmap="gray")
        ax.set_title(title)
        # Hide ticks
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        add_colorbar(fig2, ax, im)

    plt.show()
