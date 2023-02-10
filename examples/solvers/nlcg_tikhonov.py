import numpy as np
import pyelsa as elsa
import matplotlib.pyplot as plt


def example2d(s: int):
    size = np.array([s] * 2)
    phantom = elsa.phantoms.modifiedSheppLogan(size)

    distance = size[0]
    sino_descriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        np.arange(0, 180, 1), phantom.getDataDescriptor(), distance * 100.0, distance
    )

    projector = elsa.SiddonsMethod(phantom.getDataDescriptor(), sino_descriptor)
    sinogram = projector.apply(phantom)
    wsl_problem = elsa.WLSProblem(projector, sinogram)
    problem = elsa.TikhonovProblem(wsl_problem,
                                   elsa.RegularizationTerm(1.0, elsa.L2NormPow2(phantom.getDataDescriptor())))
    solver = elsa.NLCG(problem)
    reconstruction = solver.solve(20)
    plt.imshow(reconstruction)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--size", default=128, type=int, help="size of reconstruction")

    args = parser.parse_args()
    example2d(args.size)
