"""This is a reconstruction of the data provided by the Helsinki Tomography challenge 2022.

The data can be found at 10.5281/zenodo.7418878 To run this script, you need to
download the data extract it and pass it to the script. See the help message of the
script to see exactly how it can be run.

The data provided by the HTC challenge is already well preprocessed, so there
is little to be done there. The only additional preprocessing is beam hardening
correction. The implementation for that was taken from the submission of the
team of the Technical University of Denmark, Department of Applied Mathematics
and Computer Science. Full credits to them!

In this example TV regularization is used together with ADMM to reconstruct
the full data.
"""

import argparse
from pathlib import Path

import pyelsa as elsa

import numpy as np
import scipy.io as spio

import matplotlib
import matplotlib.pyplot as plt


from htc22_utils import loadmat, apply_BHC, load_htc2022data


def reconstruction(A, b, descriptor):
    """Reconstruction with FBP"""

    ramlak = elsa.makeRamLakf(descriptor)
    fbp = elsa.FBPf(A, ramlak) 
    return fbp.apply(b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(
        description="FBP reconstruction of the Helsinki Tomography challenge data"
    )

    parser.add_argument("dir", type=Path, help="Path to extracted dataset")
    parser.add_argument(
        "file",
        type=Path,
        default="htc2022_ta_full.mat",
        help="Sinogram mat file to reconstruct",
    )
  
    parser.add_argument(
        "--no-show", action="store_false", help="Do not show the final image"
    )
    parser.add_argument(
        "--save-to", type=Path, help="Path to save the reconstruction to as '.tif'"
    )

    args = parser.parse_args()

    dir = Path(args.dir)
    if not dir.is_dir():
        raise RuntimeError(f"Input path '{dir}' is not a directory")

    file = dir / args.file
    if not file.exists():
        raise RuntimeError(f"Input file '{file}' not found")

    if not file.suffix == ".mat":
        raise RuntimeError(
            f"Input file '{file}' has extension '{file.suffix}', but it needs to be '.mat'"
        )

    if "limited" in str(args.file):
        dataset = "CtDataLimited"
    else:
        dataset = "CtDataFull"
    

    b, volume_desc = load_htc2022data(file, dataset_name=dataset)
    b = apply_BHC(b)

    if elsa.cudaProjectorsEnabled():
        A = elsa.JosephsMethodCUDA(volume_desc, b.getDataDescriptor())
    else:
        A = elsa.JosephsMethod(volume_desc, b.getDataDescriptor())

    recon = reconstruction(A, b, volume_desc)

    if args.no_show:
        try:
          
            from mpl_toolkits.axes_grid1 import make_axes_locatable
        except ImportError:
            print(
                "Plots can only be created with matplotlib. Please install it and rerun the script"
            )

        fig, (ax1, ax2) = plt.subplots(1, 2)
        im = ax1.imshow(b, cmap="gray")
        ax1.axis("equal")
        ax1.axis("off")
        ax1.set_title(f"Sinogram of {args.file}")

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax2.imshow(recon, cmap="gray")
        ax2.axis("equal")
        ax2.axis("off")
        ax2.set_title(
            f"FBP reconstruction"
        )

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.tight_layout()
        plt.colorbar(im, cax=cax)
        plt.show()

    if args.save_to:
        try:
            import tifffile
        except ImportError:
            print(
                "Could not load module 'tifffile'. To save the reconstruction please install it and run it again"
            )
        savefile = args.save_to

        if not savefile.suffix == ".tif":
            savefile = savefile.parent / (savefile.name + ".tif")

        print(f"Saving to reconstruction to '{savefile}'")
        tifffile.imwrite(savefile, np.array(recon))
