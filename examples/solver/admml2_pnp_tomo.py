import numpy as np
import pyelsa as elsa
import matplotlib.pyplot as plt


class BM3D:
    def __init__(self, t=1):
        self.factor = t

    def apply(self, x, sigma, out=None):
        import bm3d

        if out is None:
            tmp = bm3d.bm3d(x, self.factor * sigma, profile="np")
            return elsa.DataContainer(tmp, x.getDataDescriptor())

        tmp = bm3d.bm3d(x, self.factor * sigma, profile="np")
        out = elsa.DataContainer(tmp, x.getDataDescriptor())


def example2d(s: int):
    size = np.array([s] * 2)
    phantom = elsa.phantoms.modifiedSheppLogan(size)

    sinoDescriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        np.arange(0, 360, 1), phantom.getDataDescriptor(), size[0] * 100.0, size[0]
    )

    A = elsa.SiddonsMethodCUDA(phantom.getDataDescriptor(), sinoDescriptor)
    b = A.apply(phantom)

    tau = 10
    id = elsa.Identity(A.getDomainDescriptor())
    proxg = BM3D(3)
    admm = elsa.ADMML2(A, b, id, proxg, tau)

    niters = 50
    reco = admm.solve(niters)
    plt.imshow(reco)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--size", default=128, type=int, help="size of reconstruction")

    args = parser.parse_args()
    example2d(args.size)
