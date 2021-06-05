from dataclasses import dataclass, field
from typing import List

import time

import numpy as np
import matplotlib.pyplot as plt

import pyelsa as elsa


class IndexTracker:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def on_scroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


@dataclass
class SolverTest:
    solver_class: elsa.Solver
    solver_name: str
    uses_subsets: bool = False
    n_subsets: int = 5
    needs_lasso: bool = False
    sampling_strategy: str = 'ROUND_ROBIN'  # or 'ROTATIONAL_CLUSTERING'
    extra_args: dict = field(default_factory=dict)


def mse(optimized: np.ndarray, original: np.ndarray) -> float:
    size = original.size
    diff = (original - optimized) ** 2
    return np.sum(diff) / size


def instantiate_solvers(solvers: List[SolverTest], do_3d=False):
    # generate the phantom
    size = np.array([128, 128]) if not do_3d else np.array([32, 32, 32])
    phantom = elsa.PhantomGenerator.createModifiedSheppLogan(size)
    volume_descriptor = phantom.getDataDescriptor()

    num_poses = 180

    if do_3d:
        # generate spherical trajectory
        n_circles = 5
        sino_descriptor = elsa.SphereTrajectoryGenerator.createTrajectory(
            num_poses, phantom.getDataDescriptor(), n_circles, elsa.SourceToCenterOfRotation(size[0] * 100),
            elsa.CenterOfRotationToDetector(size[0]))
    else:
        # generate circular trajectory
        arc = 360
        sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
            num_poses, phantom.getDataDescriptor(), arc, elsa.SourceToCenterOfRotation(size[0] * 100),
            elsa.CenterOfRotationToDetector(size[0]))

    # setup operator for 2d X-ray transform
    projector = elsa.SiddonsMethod(volume_descriptor, sino_descriptor)

    # simulate the sinogram
    sinogram = projector.apply(phantom)

    # setup reconstruction problem
    problem = elsa.WLSProblem(projector, sinogram)

    instantiated_solvers = []
    for solver in solvers:
        if solver.uses_subsets:
            strategy = elsa.SubsetSamplerPlanarDetectorDescriptorfloatSamplingStrategy.ROTATIONAL_CLUSTERING if solver.sampling_strategy == 'ROTATIONAL_CLUSTERING' else elsa.SubsetSamplerPlanarDetectorDescriptorfloatSamplingStrategy.ROUND_ROBIN

            sampler = elsa.SubsetSamplerPlanarDetectorDescriptorfloat(
                volume_descriptor, sino_descriptor, solver.n_subsets, strategy)
            subset_problem = elsa.WLSSubsetProblem(
                sampler.getProjectorSiddonsMethod(), sampler.getPartitionedData(sinogram),
                sampler.getSubsetProjectorsSiddonsMethod())
            s = solver.solver_class(
                subset_problem,
                **solver.extra_args)
        elif solver.needs_lasso:
            reg_func = elsa.L1Norm(volume_descriptor)
            reg_term = elsa.RegularizationTerm(0.000001, reg_func)

            lasso_prob = elsa.LASSOProblem(problem, reg_term)
            s = solver.solver_class(lasso_prob, **solver.extra_args)
        else:
            s = solver.solver_class(problem, **solver.extra_args)
        instantiated_solvers.append((s, solver.solver_name))

    return instantiated_solvers, phantom


def compare_solvers(solvers: List[SolverTest], do_3d=False, max_iterations=50, save_as=None, show_plot=True):
    instantiated_solvers, phantom = instantiate_solvers(solvers, do_3d)

    # solve the reconstruction problem
    distances = [[] for _ in solvers]
    optimal_phantom = np.array(phantom)

    n_iterations = list(filter(lambda x: x <= max_iterations, [1, 2, 3, 4, 5, 10, 15, 20, 50, 100, 150, 200]))
    for i in n_iterations:
        for j, solver in enumerate(instantiated_solvers):
            print(f'Solving with {solver[1]} for {i} iterations')
            reconstruction = np.array(solver[0].solve(i))
            dist = mse(reconstruction, optimal_phantom)
            distances[j].append(dist)

    print(f'Done with optimizing starting to plot now')

    fig, ax = plt.subplots()
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('MSE')
    ax.set_title('Mean Square Error of optimizers over number of iterations')
    for dist, solver in zip(distances, instantiated_solvers):
        ax.plot(n_iterations, dist, label=solver[1])
    ax.legend()

    if save_as:
        plt.savefig(save_as)

    if show_plot:
        plt.show()


def time_solvers(solvers: List[SolverTest], do_3d=False, max_iterations=50, save_as=None, show_plot=True):
    instantiated_solvers, phantom = instantiate_solvers(solvers, do_3d)

    # solve the reconstruction problem
    distances = [[] for _ in solvers]
    times = [[] for _ in solvers]
    optimal_phantom = np.array(phantom)

    n_iterations = np.arange(1, max_iterations)
    for i in n_iterations:
        for j, solver in enumerate(instantiated_solvers):
            print(f'Solving with {solver[1]} for {i} iterations')
            start = time.time()
            reconstruction = np.array(solver[0].solve(i))
            duration = time.time() - start
            times[j].append(duration)
            dist = mse(reconstruction, optimal_phantom)
            distances[j].append(dist)

    print(f'Done with optimizing starting to plot now')

    fig, ax = plt.subplots()
    ax.set_xlabel('execution time [s]')
    ax.set_ylabel('MSE')
    ax.set_title('Mean Square Error of optimizers over execution time')
    for dist, times, solver in zip(distances, times, instantiated_solvers):
        ax.plot(times, dist, label=solver[1])
    ax.legend()

    if save_as:
        plt.savefig(save_as)

    if show_plot:
        plt.show()


def compare_sqs():
    solvers = [
        SolverTest(elsa.SQS, 'SQS with 3 subsets', uses_subsets=True, n_subsets=3),
        SolverTest(elsa.SQS, 'SQS with 5 subsets', uses_subsets=True, n_subsets=5),
        SolverTest(elsa.SQS, 'SQS with 7 subsets', uses_subsets=True, n_subsets=7),
        SolverTest(elsa.SQS, 'SQS with 8 subsets', uses_subsets=True, n_subsets=8),
        SolverTest(elsa.CG, 'CG')
    ]

    compare_solvers(solvers, do_3d=True, max_iterations=10)
    time_solvers(solvers, do_3d=True, max_iterations=10)


def compare_sqs_strategies():
    solvers = [
        SolverTest(elsa.SQS, 'SQS with 3 subsets, round_robin', uses_subsets=True, n_subsets=3),
        SolverTest(elsa.SQS, 'SQS with 5 subsets, round_robin', uses_subsets=True, n_subsets=5),
        SolverTest(elsa.SQS, 'SQS with 7 subsets, round_robin', uses_subsets=True, n_subsets=7),
        SolverTest(elsa.SQS, 'SQS with 3 subsets, rotational clustering', uses_subsets=True, n_subsets=3,
                   sampling_strategy='ROTATIONAL_CLUSTERING'),
        SolverTest(elsa.SQS, 'SQS with 5 subsets, rotational clustering', uses_subsets=True, n_subsets=5,
                   sampling_strategy='ROTATIONAL_CLUSTERING'),
        SolverTest(elsa.SQS, 'SQS with 7 subsets, rotational clustering', uses_subsets=True, n_subsets=7,
                   sampling_strategy='ROTATIONAL_CLUSTERING'),
    ]

    compare_solvers(solvers, do_3d=True, max_iterations=10)
    time_solvers(solvers, do_3d=True, max_iterations=10)


def main():
    solvers = [
        SolverTest(elsa.GradientDescent, 'Gradient Descent'),  # with 1 / lipschitz as step size
        SolverTest(elsa.ISTA, 'ISTA', needs_lasso=True),
        SolverTest(elsa.FISTA, 'FISTA', needs_lasso=True),
        SolverTest(elsa.FGM, 'FGM'),
        SolverTest(elsa.OGM, 'OGM'),
        SolverTest(elsa.SQS, 'SQS'),
        SolverTest(elsa.SQS, 'SQS with 5 ordered subsets', uses_subsets=True),
        SolverTest(elsa.CG, 'CG')
    ]

    compare_solvers(solvers, max_iterations=50)
    time_solvers(solvers, max_iterations=50)


if __name__ == '__main__':
    # compare_sqs()
    compare_sqs_strategies()
    # main()
