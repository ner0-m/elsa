import numpy as np
import pyelsa as elsa


def example2d(s: int):
    size = np.array([s] * 2)
    phantom = elsa.phantoms.modifiedSheppLogan(size)

    numAngles = 180
    arc = 180
    distance = size[0]
    sinoDescriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, distance * 100.0, distance
    )

    projector = elsa.SiddonsMethod(phantom.getDataDescriptor(), sinoDescriptor)
    sinogram = projector.apply(phantom)

    problem = elsa.WLSProblem(projector, sinogram)
    solver = elsa.CG(problem)
    reconstruction = solver.solve(20)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--size", default=128, type=int, help="size of reconstruction")

    args = parser.parse_args()
    example2d(args.size)
