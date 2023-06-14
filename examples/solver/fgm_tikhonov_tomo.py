import numpy as np
import pyelsa as elsa
import matplotlib.pyplot as plt


def example2d(s: int, show: bool = True):
    size = np.array([s] * 2)
    phantom = elsa.phantoms.modifiedSheppLogan(size)

    numAngles = 180
    arc = 180
    distance = size[0]
    sinoDescriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        np.arange(0, 180, 1), phantom.getDataDescriptor(), distance * 100.0, distance
    )

    projector = elsa.SiddonsMethod(phantom.getDataDescriptor(), sinoDescriptor)
    sinogram = projector.apply(phantom)

    ls = elsa.LeastSquares(projector, sinogram)
    reg = elsa.L2Squared(phantom.getDataDescriptor())
    problem = ls + 10. * reg

    solver = elsa.FGM(problem)
    reconstruction = solver.solve(20)

    if show:
        plt.imshow(reconstruction)
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--size", default=128, type=int, help="size of reconstruction")
    parser.add_argument("--no-show", action="store_false")

    args = parser.parse_args()
    show = args.no_show
    example2d(args.size, show)
