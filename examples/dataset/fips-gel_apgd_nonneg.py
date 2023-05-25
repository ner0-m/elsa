"""This is a reconstruction of the gel phantom provided by the Finnish Inverse
Problems Society (FIPS).

The DOI of the data is 10.5281/zenodo.7876521. It can be found at
https://zenodo.org/record/7876521. To run the script, download
the zip files and extract them in a common folder. Then pass the path as
an argument to the script

A description taken from the FIPS website (https://www.fips.fi/dataset.php#gel)
describing the dataset:
> This is an open-access dataset of tomographic X-ray data of a dynamic
> agarose-gel phantom perfused with a liquid contrast agent. The dataset
> consists of 17 consecutive X-ray measurements of a 2D slice of the gel
> phantom. This data is provided in two resolutions and stored in a special
> CT-data structure containing the metadata of the measurement setup.

In this example, the constrained optimization problem with the least squares
data term with a non-negativity constraint is solved using APGD/FISTA.
"""
from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import mat73
import numpy as np
import pyelsa as elsa


def define_sino_descriptor(projections, params, resolution=1.0, binning=2):
    angles = params["angles"]
    sourceToDetector = params["distanceSourceDetector"]
    sourceToOrigin = float(params["distanceSourceOrigin"])
    originToDetector = float(sourceToDetector - sourceToOrigin)

    detectorSize = projections.shape[:-1]
    detectorSpacing = [params["effectivePixelSize"] * binning]

    # Compute magnification
    magnification = sourceToDetector / sourceToOrigin

    # Determine good size and spacing for volume
    vol_spacing = detectorSpacing[0] / (magnification * resolution)
    size = int(math.ceil(np.max(projections.shape[:-1]) * resolution))

    volume_descriptor = elsa.VolumeDescriptor([size] * 2, [vol_spacing] * 2)

    sino_descriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        angles,
        volume_descriptor,
        sourceToOrigin,
        originToDetector,
        [0],
        [0, 0],  # Offset of origin
        detectorSize,
        detectorSpacing,
    )

    projections = np.flip(projections, axis=1)
    projections = np.roll(projections, -90, axis=1)
    return elsa.DataContainer(projections, sino_descriptor), volume_descriptor


def reconstruct(A, b, niters=70):
    """Just a basic reconstruction which looks nice"""
    prox = elsa.ProximalBoxConstraint(0)
    solver = elsa.APGD(A, b, prox)
    return solver.solve(niters)


def do_all_frames(singram, params, binning, niters, resolution=1):
    recos = []
    for i in range(16):
        sino, vol_desc = define_sino_descriptor(
            sinogram[i], params[i], resolution=resolution, binning=binning
        )
        sino_descriptor = sino.getDataDescriptor()

        # Set the forward and backward projector
        projector = elsa.JosephsMethodCUDA(vol_desc, sino_descriptor)

        reco = reconstruct(projector, sino, niters=niters)
        recos.append(reco)

    fig, axs = plt.subplots(4, 4, figsize=(20, 20))

    for i, ax in enumerate(axs.flatten()):
        ax.imshow(np.array(recos[i]), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Timeframe {i}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruction of the open-access dataset of agarose gel phantoms."
    )

    parser.add_argument("path", type=Path, help="Path to the matlab files")
    parser.add_argument(
        "-i",
        "--iters",
        default=70,
        type=int,
        help="Number of iterations for the reconstruction",
    )
    parser.add_argument(
        "-b",
        "--binning",
        default=2,
        type=int,
        help="Binning of the sinogram (must be 2 or 4)",
    )
    parser.add_argument(
        "-f",
        "--frame",
        default=0,
        help="Select one of the 17 frames (0 - 16), or 'all'",
    )

    parser.add_argument(
        "--resolution",
        default=1.0,
        type=float,
        help="Change the resolution of the reconstruction (default 1.0)",
    )

    args = parser.parse_args()

    if args.binning not in (2, 4):
        raise ValueError(f"Binnng must be 2 or 4 (is {args.binning})")

    try:
        frame = int(args.frame)

        if frame < 0 or frame > 16:
            raise ValueError(
                f"Frame should be an int between 0 and 16 or 'all' (is {frame})"
            )
    except ValueError as err:
        if args.frame != "all":
            raise ValueError(
                f"Frame should be an int between 0 and 16 or 'all' (is {frame})"
            )
        frame = args.frame

    phantom = f"GelPhantomData_b{args.binning}"
    mat = mat73.loadmat(f"{args.path / phantom}.mat")
    params = mat[phantom]["parameters"]
    sinogram = mat[phantom]["sinogram"]
    sinogram = np.transpose(sinogram, (0, 2, 1))

    binning = args.binning
    iters = args.iters
    if frame == "all":
        do_all_frames(sinogram, params, binning, iters, resolution=args.resolution)
    else:
        sino, vol_desc = define_sino_descriptor(
            sinogram[frame], params[frame], binning=binning, resolution=args.resolution
        )
        sino_desc = sino.getDataDescriptor()

        # Set the forward and backward projector
        projector = elsa.JosephsMethodCUDA(vol_desc, sino_desc)

        reco = reconstruct(projector, sino, niters=iters)
        plt.imshow(np.asarray(reco), cmap="gray")
        plt.show()
