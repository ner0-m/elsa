import numpy as np
import pyelsa as elsa
import matplotlib.pyplot as plt


def example_leastsquares(s: int):
    size = np.array([s] * 2)
    phantom = elsa.phantoms.modifiedSheppLogan(size)

    distance = size[0]
    sino_descriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        np.arange(0, 180, 1), phantom.getDataDescriptor(), distance * 100.0, distance
    )

    projector = elsa.SiddonsMethod(phantom.getDataDescriptor(), sino_descriptor)
    sinogram = projector.apply(phantom)
    problem = elsa.LeastSquares(projector, sinogram)

    solver = elsa.CGNL(problem)
    reconstruction = solver.solve(20)
    plt.imshow(reconstruction)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--size", default=128, type=int, help="size of reconstruction")

    args = parser.parse_args()
    example_leastsquares(args.size)
