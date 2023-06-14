import pytest

import numpy as np
import pyelsa as elsa


def changedc(dc):
    """Simple function which manipulates a data container"""
    dc[0] += 2


def test_opaque():
    size = [2, 2]
    desc = elsa.VolumeDescriptor(size)

    array = np.random.random(np.prod(size))
    dc = elsa.DataContainer(array, desc)
    changedc(dc)
    assert dc[0] == pytest.approx(array[0] + 2.0)
