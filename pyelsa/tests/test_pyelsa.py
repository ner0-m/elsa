import pyelsa as elsa

import numpy as np

import unittest
import random
import math
import os


class PyelsaTest(unittest.TestCase):
    def assert_same_shape(
        self, arr: np.array, dc: elsa.DataContainer, fail_msg: str = None
    ):
        "Assert whether arr and dc have the same shape"

        coeffsPerDim = dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()
        for i in range(len(coeffsPerDim)):
            self.assertEqual(arr.shape[i], coeffsPerDim[i], fail_msg)

    def assert_equal(self, arr: np.array, dc: elsa.DataContainer, fail_msg: str = None):
        "Assert whether arr and dc represent the same array"

        arr_f = arr.flatten(order="F")
        for i in range(len(arr_f)):
            self.assertEqual(arr_f[i], dc[i], fail_msg)

    def test_memory_layout(self):
        "Test memory layout of DataContainer and converted np.array"

        # test C-array -> DataContainer conversion
        arr_c = np.array([[1, 2, 3], [4, 5, 6]], order="C")
        dc = elsa.DataContainer(arr_c)
        self.assert_same_shape(
            arr_c,
            dc,
            "Converting a C-array to DataContainer yields container with different shape",
        )
        self.assert_equal(
            arr_c,
            dc,
            "Converting a C-array to DataContainer produces a different array",
        )

        # test Fortran-array -> DataContainer conversion
        arr_f = np.array([[1, 2, 3], [4, 5, 6]], order="F")
        dc = elsa.DataContainer(arr_f)
        self.assert_same_shape(
            arr_f,
            dc,
            "Converting a Fortran-array to DataContainer yields container with different shape",
        )
        self.assert_equal(
            arr_f,
            dc,
            "Converting a Fortran-array to DataContainer produces a different array",
        )

        # test view of DataContainer as numpy.array
        dc = elsa.DataContainer(elsa.VolumeDescriptor([2, 3]), [1, 4, 2, 5, 3, 6])
        arr_view = np.array(dc, copy=False)
        self.assert_same_shape(
            arr_view,
            dc,
            "The view of a DataContainer as a numpy.array and the container have different shapes",
        )
        self.assert_equal(
            arr_view,
            dc,
            "The view of a DataContainer as a numpy.array uses a different indexing convention",
        )
        arr_view[1, 0] = random.randint(1, 1000)
        self.assertEqual(
            arr_view[1, 0],
            dc[dc.getDataDescriptor().getIndexFromCoordinate([1, 0])],
            "Altering the view of DataContainer as a np.array does not alter the DataContainer",
        )

        # test np.array copy of DataContainer
        arr_cpy = np.array(dc, copy=True)
        self.assert_same_shape(
            arr_cpy,
            dc,
            "The np.array copy of a DataContainer and the container have different shapes",
        )
        self.assert_equal(
            arr_cpy,
            dc,
            "The np.array copy of a DataContainer uses a different indexing convention",
        )

        # test indexing invariant under np.array -> DataContainer -> np.array conversion
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        dc = elsa.DataContainer(arr)
        dc_view = np.array(dc, copy=False)
        self.assertEqual(
            arr.shape,
            dc_view.shape,
            "Indexing not invariant under np.array -> DataContainer -> np.array conversion",
        )
        self.assertTrue(
            (arr == dc_view).all(),
            "Indexing not invariant under np.array -> DataContainer -> np.array conversion",
        )

    def test_datacontainer(self):
        size = 1000
        v = np.random.randn(size).astype(np.float32)
        w = np.random.randn(size).astype(np.float32)

        desc = elsa.VolumeDescriptor([size])

        x = elsa.DataContainer(desc, v)
        y = elsa.DataContainer(desc, w)

        np.testing.assert_allclose(+x, +v, rtol=1e-5)
        np.testing.assert_allclose(-x, -v, rtol=1e-5)

        np.testing.assert_allclose(x + y, v + w, rtol=1e-4)
        np.testing.assert_allclose(x - y, v - w, rtol=1e-4)
        np.testing.assert_allclose(x * y, v * w, rtol=1e-4)
        np.testing.assert_allclose(x / y, v / w, rtol=1e-4)

        np.testing.assert_allclose(x + 5, v + 5, rtol=1e-4)
        np.testing.assert_allclose(x - 5, v - 5, rtol=1e-4)
        np.testing.assert_allclose(x * 5, v * 5, rtol=1e-4)
        np.testing.assert_allclose(x / 5, v / 5, rtol=1e-4)

        np.testing.assert_allclose(5 + y, 5 + w, rtol=1e-4)
        np.testing.assert_allclose(5 - y, 5 - w, rtol=1e-4)
        np.testing.assert_allclose(5 * y, 5 * w, rtol=1e-4)
        np.testing.assert_allclose(5 / y, 5 / w, rtol=1e-4)

        x += y
        v += w
        np.testing.assert_allclose(x, v, rtol=1e-4)
        x -= y
        v -= w
        np.testing.assert_allclose(x, v, rtol=1e-4)
        x *= y
        v *= w
        np.testing.assert_allclose(x, v, rtol=1e-4)
        x /= y
        v /= w
        np.testing.assert_allclose(x, v, rtol=1e-4)

        for i in range(0, size):
            self.assertAlmostEqual(x[i], v[i], 5)
            self.assertAlmostEqual(y[i], w[i], 5)

        x[0] = 12345
        self.assertAlmostEqual(x[0], 12345)

    def _test_reconstruction(
        self,
        size,
        phantom,
        trajectory_generator,
        *args,
        iterCount=20,
        error=0.004,
        **kwargs
    ):
        "Try performing a simple reconstruction"

        numAngles = 50
        arc = 360

        volDesc = phantom.getDataDescriptor()

        # generate circular trajectory
        sinoDesc = trajectory_generator.createTrajectory(
            numAngles, volDesc, arc, size * 100.0, size, *args, **kwargs
        )

        # setup operator for X-ray transform
        projector = elsa.SiddonsMethod(volDesc, sinoDesc)

        # simulate sinogram
        sino = projector.apply(phantom)

        # solve the reconstruction problem
        solver = elsa.CGLS(projector, sino)
        recon = solver.solve(iterCount)

        # compute mse and check that it is within bounds
        mse = (recon - phantom).squaredL2Norm() / size**2
        self.assertLess(
            mse,
            error,
            "Mean squared error of reconstruction too large",
        )

    def _test_reconstruction_2d(self, *args, **kwargs):
        size = 16.0
        # generating a 2d phantom
        phantom = elsa.phantoms.modifiedSheppLogan([size, size])
        self._test_reconstruction(size, phantom, *args, **kwargs)

    def _test_reconstruction_3d(self, *args, **kwargs):
        size = 16.0
        # generating a 3d phantom
        phantom = elsa.phantoms.modifiedSheppLogan([size, size, size])
        self._test_reconstruction(size, phantom, *args, **kwargs)

    def test_reconstruction_2d_curved_more_iterations(self):
        self._test_reconstruction_2d(
            elsa.CurvedCircleTrajectoryGenerator, elsa.Radian(2.0), iterCount=40
        )

    def test_reconstruction_2d_curved_smaller_angle(self):
        self._test_reconstruction_2d(
            elsa.CurvedCircleTrajectoryGenerator, elsa.Radian(0.5)
        )

    def test_reconstruction_2d_planar(self):
        self._test_reconstruction_2d(elsa.CircleTrajectoryGenerator)

    def test_reconstruction_3d_curved_more_iterations(self):
        self._test_reconstruction_3d(
            elsa.CurvedCircleTrajectoryGenerator,
            elsa.Radian(2.0),
            iterCount=40,
            error=0.2,
        )

    def test_reconstruction_3d_curved_smaller_angle(self):
        self._test_reconstruction_3d(
            elsa.CurvedCircleTrajectoryGenerator, elsa.Radian(0.5), error=0.2
        )

    def test_reconstruction_3d_planar(self):
        self._test_reconstruction_3d(elsa.CircleTrajectoryGenerator, error=0.2)

    # def test_logging(self):
    #     "Test logging module interface"
    #
    #     # the elsa.Logger appears to bypass the Python-side sys.stdout
    #     # so we only test file logging
    #     logfile_name = "test_pyelsa_log.txt"
    #     if os.path.exists(logfile_name):
    #         os.remove(logfile_name)
    #     elsa.Logger.enableFileLogging(logfile_name)
    #
    #     elsa.Logger.setLevel(elsa.LogLevel.OFF)
    #     # this should produce no output when logging is off
    #     self.test_intermodule_object_exchange()
    #     elsa.Logger.flush()
    #     with open(logfile_name) as log_file:
    #         self.assertTrue(len(log_file.readline()) == 0)
    #
    #     elsa.Logger.setLevel(elsa.LogLevel.TRACE)
    #     # this should produce some output when using a low logging level
    #     self.test_intermodule_object_exchange()
    #     elsa.Logger.flush()
    #     with open(logfile_name) as log_file:
    #         self.assertTrue(len(log_file.readline()) > 0)


if __name__ == "main":
    unittest.main()
