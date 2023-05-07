import numpy as np
import pyelsa as elsa
import matplotlib.pyplot as plt


def example2d_phase_contrast(s: int):
    size = np.array([s] * 2)
    phantom = elsa.phantoms.modifiedSheppLogan(size)

    arc = 180
    distance = size[0]
    sinoDescriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        np.arange(0, arc, 1), phantom.getDataDescriptor(), distance * 100.0, distance
    )

    # create projector, alternatively PhaseContrastBSplineVoxelProjector
    projector = elsa.PhaseContrastBlobVoxelProjector(phantom.getDataDescriptor(), sinoDescriptor)
    sinogram = projector.apply(phantom)

    solver = elsa.CGLS(projector, sinogram)
    reconstruction = solver.solve(20)
    plt.imshow(reconstruction)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create 2D Phantom, compute sinogram and reconstruction")
    parser.add_argument("--size", default=128, type=int, help="size of phantom")

    args = parser.parse_args()
    example2d_phase_contrast(args.size)
