import numpy as np
import pyelsa as elsa
import matplotlib.pyplot as plt


def example2d(s: int):
    size = np.array([s] * 2)
    phantom = elsa.phantoms.modifiedSheppLogan(size)

    sinoDescriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        np.arange(0, 360, 1), phantom.getDataDescriptor(), size[0] * 100.0, size[0]
    )

    A = elsa.SiddonsMethodCUDA(phantom.getDataDescriptor(), sinoDescriptor)
    b = A.apply(phantom)

    grad = elsa.FiniteDifferences(phantom.getDataDescriptor())
    proxg = elsa.ProximalL1(10)
    tau = 0.05
    admm = elsa.ADMML2(A, b, grad, proxg, tau)

    niters = 30
    reco = admm.solve(niters)
    plt.imshow(reco)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--size", default=128, type=int, help="size of reconstruction")

    args = parser.parse_args()
    example2d(args.size)
