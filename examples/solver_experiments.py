import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib
import numpy as np
import pyelsa as elsa
import time


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


def instantiate_solvers(solvers: List[SolverTest], problem_size: int, do_3d=False):
    # generate the phantom
    size = np.array([problem_size, problem_size]) if not do_3d else np.array([problem_size, problem_size, problem_size])
    phantom = elsa.phantoms.modifiedSheppLogan(size)
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
            num_poses, phantom.getDataDescriptor(), arc, size[0] * 100, size[0])

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


def compare_solvers(solvers: List[SolverTest], show_plot: bool, max_iterations: int, problem_size: int, save_as=None,
                    do_3d=False, ):
    instantiated_solvers, phantom = instantiate_solvers(solvers, problem_size=problem_size, do_3d=do_3d)

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

    import matplotlib.pyplot as plt  # local imports so that we can switch to headless mode before importing
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


def time_solvers(solvers: List[SolverTest], max_iterations: int, problem_size: int, show_plot: bool, do_3d=False,
                 save_as=None):
    instantiated_solvers, phantom = instantiate_solvers(solvers, do_3d=do_3d, problem_size=problem_size)

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

    import matplotlib.pyplot as plt  # local imports so that we can switch to headless mode before importing
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


def compare_sqs_strategies(max_iterations: int, show_plots: bool, problem_size: int, plots_dir: Optional[Path] = None):
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
        SolverTest(elsa.CG, 'CG')
    ]

    compare_solvers(solvers, do_3d=True, max_iterations=max_iterations, show_plot=show_plots, problem_size=problem_size,
                    save_as=plots_dir / 'sqs_comparison_convergence.png' if plots_dir else None)
    time_solvers(solvers, do_3d=True, max_iterations=max_iterations, show_plot=show_plots, problem_size=problem_size,
                 save_as=plots_dir / 'sqs_comparison_time.png' if plots_dir else None)


def evaluate_solvers(max_iterations: int, show_plots: bool, problem_size: int, plots_dir: Optional[Path] = None):
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

    compare_solvers(solvers, max_iterations=max_iterations, show_plot=show_plots, problem_size=problem_size,
                    save_as=plots_dir / 'solvers_comparison_convergence.png' if plots_dir else None)
    time_solvers(solvers, max_iterations=max_iterations, show_plot=show_plots, problem_size=problem_size,
                 save_as=plots_dir / 'solvers_comparison_time.png' if plots_dir else None)


def dir_path(path) -> Path:
    if Path(path).is_dir():
        return Path(path)
    raise argparse.ArgumentTypeError(f'directory {path} is not a valid path')


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', help='Run matplotlib in headless mode')
    parser.add_argument('--max-iterations', type=int, default=50,
                        help='Maximum number of iterations for solver experiments')
    parser.add_argument('--plots-dir', type=dir_path, required=False, help='Directory to store the plots in')
    parser.add_argument('--problem-size', type=int, default=128, help='Size of the problems to test')

    return parser


def main(max_iterations: int, headless_mode: bool, plots_dir: Optional[Path], problem_size: int):
    if headless_mode:
        matplotlib.use('Agg')
    compare_sqs_strategies(max_iterations=max_iterations, show_plots=not headless_mode, plots_dir=plots_dir,
                           problem_size=problem_size)
    evaluate_solvers(max_iterations=max_iterations, show_plots=not headless_mode, plots_dir=plots_dir,
                     problem_size=problem_size)


if __name__ == '__main__':
    args = create_parser().parse_args()
    main(args.max_iterations, args.headless, args.plots_dir, args.problem_size)
