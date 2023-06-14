import numpy as np
import pyelsa as elsa
import matplotlib.pyplot as plt


def example2d_gmres(s: int):
    size = np.array([s] * 2)
    phantom = elsa.phantoms.modifiedSheppLogan(size)

    distance = size[0]
    sinoDescriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        np.arange(0, 180, 1), phantom.getDataDescriptor(), distance * 100.0, distance
    )

    # creates an projector that uses an unmatched backprojector
    projector = elsa.JosephsMethod(phantom.getDataDescriptor(), sinoDescriptor)
    # for a matched projector case JosephsMethod needs to be set to fast=False instead (default is True)
    # projector = elsa.JosephsMethod(phantom.getDataDescriptor(), sinoDescriptor, fast=False)

    sinogram = projector.apply(phantom)

    # default elsa.GMRES solver implementation applies ABGMRES
    # you can explicitly call AB- or BA-GMRES as elsa.ABGMRES(...) or elsa.BAGMRES(...)
    solver = elsa.GMRES(projector, sinogram)

    reconstruction = solver.solve(20)
    plt.imshow(reconstruction)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--size", default=128, type=int, help="size of reconstruction")

    args = parser.parse_args()
    example2d_gmres(args.size)
