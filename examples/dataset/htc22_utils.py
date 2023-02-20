import argparse
from pathlib import Path

import pyelsa as elsa

import numpy as np
import scipy.io as spio

def loadmat(filename):
    """
    Use SciPy load to load the matlab file and turn it to a Python dictionary.
    Credit to: https://stackoverflow.com/a/8832212
    """

    def _check_keys(d):
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mat_struct):
                d[strg] = _todict(elem)
            else:
                d[strg] = elem
        return d

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def apply_BHC(data):
    """Apply beam hardening correction

    Credits to the team at the Technical University of Denmark, at the
    Department of Applied Mathematics and Computer Science. This was taken
    from their repository to the contribution at:
    https://github.com/TomographicImaging/CIL-HTC2022-Algo2
    """
    # these coefficients are generated from the full disk data
    coefficients = np.array([0.00130522, 0.9995882, -0.01443113, 0.07282656])
    corrected_data = np.polynomial.polynomial.polyval(np.asarray(data), coefficients)
    return elsa.DataContainer(corrected_data, data.getDataDescriptor())


def load_htc2022data(filename, dataset_name="CtDataFull"):
    """Load matlab file and setup elsa geometry and sinogram"""

    # read in matlab file
    mat = loadmat(filename)
    params = mat[dataset_name]["parameters"]

    # read important parameters
    ds2c = params["distanceSourceOrigin"]
    ds2d = params["distanceSourceDetector"]
    dc2d = ds2d - ds2c

    detpixel_spacing = params["pixelSizePost"]
    num_detpixel = params["numDetectorsPost"]
    angles = params["angles"]

    # Rought approximation of a volume size
    vol_npixels = int(num_detpixel / np.sqrt(2))
    vol_spacing = detpixel_spacing

    # Description of the desired volume
    volume_descriptor = elsa.VolumeDescriptor([vol_npixels] * 2, [vol_spacing] * 2)

    sino_descriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
        angles,
        volume_descriptor,
        ds2c,
        dc2d,
        [0],
        [0, 0],
        [num_detpixel] * 2,
        [detpixel_spacing] * 2,
    )

    # read data
    scan_sinogram = mat[dataset_name]["sinogram"].astype("float32")
    sino_data = scan_sinogram.transpose(1, 0)
    sino_data = np.flip(sino_data, axis=1)

    return (
        elsa.DataContainer(sino_data, sino_descriptor),
        volume_descriptor,
    )
