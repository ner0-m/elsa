import numpy as np
import pyelsa as elsa
import matplotlib.pyplot as plt


def reconstruct(A, b, sigma=1, tau=0.000006, l2reg=1, tvreg=1, niters=50):
    grad = elsa.FiniteDifferences(A.getDomainDescriptor())

    # Define Proximal Operators
    proxg1 = elsa.ProximalL2Squared(b)
    proxg2 = elsa.ProximalL1(tvreg)

    # Combine them to one
    proxg = elsa.CombinedProximal(proxg1, proxg2)

    # Setup stacked operator containing A, and Gradient
    K = elsa.BlockLinearOperator(
        [A, grad],
        elsa.BlockLinearOperator.BlockType.ROW,
    )

    # Use box constraint as proximal for f
    proxf = elsa.ProximalBoxConstraint(0)

    admm = elsa.LinearizedADMM(K, proxf, proxg, sigma, tau)
    return admm.solve(niters)


def example2d(s: int, l2reg=1.0, tvreg=1.0, sigma=1.0, tau=1.0, niters=10):
    size = np.array([s] * 2)
    phantom = elsa.phantoms.modifiedSheppLogan(size)

    sinoDescriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        np.arange(0, 360, 1), phantom.getDataDescriptor(), size[0] * 100.0, size[0]
    )

    A = elsa.JosephsMethodCUDA(phantom.getDataDescriptor(), sinoDescriptor)
    b = A.apply(phantom)

    reco = reconstruct(
        A, b, sigma=sigma, tau=tau, l2reg=l2reg, tvreg=tvreg, niters=niters
    )
    plt.imshow(reco, cmap="gray")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reconstruction of a simple Shepp-Logan Phantom using Linearized ADMM"
    )
    parser.add_argument("--size", default=128, type=int, help="size of reconstruction")
    parser.add_argument(
        "--sigma", default=5.0, type=float, help="First hyperparameter for ADMM"
    )
    parser.add_argument(
        "--tau", default=2.8e-5, type=float, help="Second hyperparameter for ADMM"
    )
    parser.add_argument(
        "--l2reg",
        default=1.0,
        type=float,
        help="Regularization Parameter for L2 norm Squared",
    )
    parser.add_argument(
        "--tvreg", default=1, type=float, help="Regularization Parameter for TV"
    )
    parser.add_argument(
        "--iters", default=50, type=int, help="Number of iterations to run"
    )

    args = parser.parse_args()
    example2d(
        args.size,
        l2reg=args.l2reg,
        tvreg=args.tvreg,
        niters=args.iters,
        sigma=args.sigma,
        tau=args.tau,
    )
