"""This is a reconstruction of the open-access chickenbone dataset provided by
the Finnish Inverse Problems Society.

The DOI of the data is 10.5281/zenodo.6986012. It can be found at
https://zenodo.org/record/6986012. The run the script you need to download
the zip files and extract them in a common folder.

A description taken from the FIPS website
(https://www.fips.fi/dataset.php#chickenbone) describing the dataset:
> This is an open-access dataset of a series of 3D cone-beam computed
> tomography scans (CBCT) of a chicken bone imaged at 4 different dose levels.
> The dose was changed by varying the X-ray tube current and detector exposure
> time appropriately.

In this example, the (order subset) maximum likelihood expectation maximization
algorithm is used as a reconstruction algorithm. As such, the negative
log-transform is only performed on the final visualization, the maximization
is performed on the raw transmission data.
"""

import argparse
from pathlib import Path
import math

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import pyelsa as elsa
import tifffile
import tqdm


# Taken from the provided txt file describing the geometry
sourceToDetector = 553.74
sourceToOrigin = 210.66
originToDetector = sourceToDetector - sourceToOrigin

detectorSpacing = np.asarray([0.050, 0.050])


def rebin2d(arr, binfac):
    """Rebin 2D array arr to shape new_shape by averaging.

    Credits to: https://scipython.com/blog/binning-a-2d-array-in-numpy/
    """

    nx, ny = arr.shape
    bnx, bny = nx // binfac, ny // binfac
    shape = (bnx, arr.shape[0] // bnx, bny, arr.shape[1] // bny)
    return arr.reshape(shape).mean(-1).mean(1)


def preprocess(proj, binning=None, padding=None, cor=0):
    """Preprocessing from transmission to absorption images including some cleaning and correction"""
    # Correct for slight misalignment
    proj = np.roll(proj, cor, axis=1)

    # Take a part of the background and compute an approximation of the
    # initial intensity
    background = proj[:128, :128]
    I0 = np.mean(background)

    # If binning should be performed, do it now
    if binning and binning != 1:
        proj = rebin2d(proj, binning)

    if padding:
        proj = np.pad(proj, padding, constant_values=I0)

    # reduce some noise
    proj[proj > I0] = I0

    return proj


def load_dataset(path=Path("."), binning=1, step=1, padding=50):
    # Take all tif files in the path
    files = sorted(path.glob("20220811_chicken_bone_dose_*.tif"))

    projections = []
    angles = []
    for file in tqdm.tqdm(files[:-1:step], desc="Loading and preproecessing"):
        raw = tifffile.imread(file).astype("float32")
        angles.append((int(str(file).split("_")[-1].split(".")[0]) - 1))
        projections.append(preprocess(raw, binning=binning, padding=padding))

    # Final step of preprocessing
    projections /= np.max(projections)

    # Change to correct storage order for elsa
    projections = np.transpose(projections, (2, 1, 0))

    return projections, np.array(angles, dtype=np.float32)


def reconstruct(A, b, niters=10):
    # Dispatch to the correct solver based on the arguments. In our example, the
    # last case should be impossible, but just for good measure :-)
    if isinstance(A, list) and len(A) > 1:
        solver = elsa.OSMLEM(A, b, 1e-8)
        return solver.solve(niters)

    if isinstance(A, list) and len(A) == 1:
        solver = elsa.MLEM(A[0], b[0], 1e-8)
        return solver.solve(niters)

    solver = elsa.MLEM(A, b, 1e-8)
    return solver.solve(niters)


def define_sino_descriptor(projections, volume_descriptor, binning, angles, nsubs=1):
    num_angles = projections.shape[-1]

    assert len(angles) == num_angles

    # split angles and projections into `nsubs` mostly equal parts
    angle_subs = np.array_split(angles, nsubs)
    proj_subs = np.array_split(projections, nsubs, axis=-1)

    # Create `nsubs` descriptors for each subset of angles + projections
    sino_descs = [
        elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
            subset,
            volume_descriptor,
            sourceToOrigin,
            originToDetector,
            [0, 0],
            [0, 0, 0],  # Offset of origin
            projections.shape[:-1],
            detectorSpacing * binning,
        )
        for subset in angle_subs
    ]

    # Now return nsubs sinograms as a list
    return [
        elsa.DataContainer(proj, desc) for (proj, desc) in zip(proj_subs, sino_descs)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="elsa Reconstruction of the open-access chickenbone dataset using (OS)MLEM"
    )
    parser.add_argument("path", type=Path, help="Path to look for tif files")
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=5,
        help="Number of iterations for the reconstruction algorithm",
    )
    parser.add_argument(
        "-b",
        "--binning",
        type=int,
        default=2,
        help="Binning factor of preprocessing (needs to be a power of two)",
    )
    parser.add_argument(
        "-s", "--step", type=int, default=1, help="Take every n-th image only"
    )
    parser.add_argument("--padding", type=int, default=100,
                        help="Additional padding")
    parser.add_argument("--nsubs", type=int, default=1,
                        help="Number of Subsets")
    parser.add_argument(
        "--resolution",
        type=float,
        default=1,
        help="Change the resolution of the reconstruction (default 1.0)",
    )

    args = parser.parse_args()

    if not elsa.cudaProjectorsEnabled():
        import warnings

        warnings.warn(
            "elsa is build without CUDA support. This reconstruction may take a long time."
        )

    binning = args.binning
    padding = args.padding
    resolution = args.resolution

    # load the dataset
    projections, angles = load_dataset(
        path=args.path,
        binning=binning,
        step=args.step,
        padding=padding,
    )

    # compute some I0 to log transform the final visualization, as no flat-field
    # images are provided
    I0 = np.mean(projections[:128, :128, :])

    # Compute magnification
    magnification = sourceToDetector / sourceToOrigin

    # Determine good size and spacing for volume
    vol_spacing = (detectorSpacing[0] * binning) / (magnification * resolution)
    size = int(
        math.ceil((np.max(projections.shape[:-1]) - padding) * resolution))

    volume_descriptor = elsa.VolumeDescriptor([size] * 3, [vol_spacing] * 3)

    # define the sinogram for elsa
    sinogram = define_sino_descriptor(
        projections, volume_descriptor, binning, angles, nsubs=args.nsubs
    )

    # Perform reconstruction
    recon = reconstruct(
        # create an operator for each subset
        [
            elsa.JosephsMethodCUDA(volume_descriptor, sino.getDataDescriptor())
            for sino in sinogram
        ],
        sinogram,
        niters=args.iters,
    )

    # Now just show everything...
    npreco = -np.log(np.array(recon / I0))

    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    fig1, fig2 = fig.subfigures(nrows=2, ncols=1)

    def add_colorbar(fig, ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

    ncols = 4

    axs = fig1.subplots(nrows=1, ncols=ncols)
    slices = np.arange(size // 3, 2 * size // 3, size // 3 / ncols, dtype=int)
    for ax, i in zip(axs, slices):
        im = ax.imshow(npreco[:, :, i], cmap="gray")
        # Hide ticks
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(f"Slice {i}")
        add_colorbar(fig, ax, im)

    axs = fig2.subplots(nrows=1, ncols=ncols)
    for ax, i in zip(axs, slices):
        im = ax.imshow(npreco[:, i, :], cmap="gray")
        # Hide ticks
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(f"Slice {i}")
        add_colorbar(fig, ax, im)
    plt.show()
