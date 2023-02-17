import numpy as np
import pyelsa as elsa
import matplotlib.pyplot as plt


def example(s: int):
    size = np.array([s] * 2)
    phantom = elsa.phantoms.modifiedSheppLogan(size)
    volDesc = phantom.getDataDescriptor()

    distance = size[0]
    sinoDesc = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        np.arange(0, 180, 1), phantom.getDataDescriptor(
        ), distance * 100.0, distance
    )

    projector = elsa.JosephsMethodCUDA(phantom.getDataDescriptor(), sinoDesc)
    sinogram = projector.apply(phantom)

    proxf = elsa.ProximalIdentity()

    prox1 = elsa.ProximalL2Squared(sinogram)
    prox2 = elsa.ProximalL1()
    proxg = elsa.CombinedProximal(prox1, prox2)

    id = elsa.Identity(volDesc)
    grad = elsa.FiniteDifferences(volDesc)

    blockDesc = elsa.RandomBlocksDescriptor(
        [sinoDesc, grad.getRangeDescriptor()])

    K = elsa.BlockLinearOperator(
        volDesc, blockDesc, [projector, grad], elsa.BlockLinearOperator.ROW
    )

    solver = elsa.LinearizedADMM(K, proxf, proxg, 0.0001, 0.00001)
    reco = solver.solve(150)
    plt.imshow(reco)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--size", default=128, type=int,
                        help="size of reconstruction")

    args = parser.parse_args()
    example(args.size)
