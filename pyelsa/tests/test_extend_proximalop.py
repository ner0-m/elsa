import pyelsa as elsa
import numpy as np


def test_extend_proximal():
    size = [2, 2]
    desc = elsa.VolumeDescriptor(size)

    array = np.random.random(np.prod(size))
    dc = elsa.DataContainer(array, desc)

    i = 0
    j = 0

    class TestProx1:
        def apply(self, x, t, out=None):
            # Just to be sure which will be called
            nonlocal i, j
            if out is None:
                i += 1
                return x * t
            else:
                j += 1
                out.set(x * t)

    prox = TestProx1()
    out = elsa.proxapply(prox, dc, 10)
    np.testing.assert_allclose(array * 10, np.array(out).flatten("F"))
    assert i == 1
    assert j == 0

    out = elsa.DataContainer(desc)
    elsa.proxapplyout(prox, dc, 10, out)

    np.testing.assert_allclose(array * 10, np.array(out).flatten("F"))
    assert i == 1
    assert j == 1
