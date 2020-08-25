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

    def test_intermodule_object_exchange(self):
        """Test whether instances of a class produced by one module (e.g. pyelsa_functionals)
        are recognised as instances of the same class in another module that is only aware of the
        classes interface (e.g. pyelsa_problems). When that is not the case, dynamic_casts on the 
        C++ side fail (produce a nullptr). This may occur as a result of pybind11 enforcing
        -fvisibility=hidden and libc++ considering objects coming from different modules to be of
        different types.
        """

        # The constructors used to convert between different problem types use
        # dynamic_casts on functionals extensively, so we test whether these conversions
        # work as expected

        # test converting a Tikhonov Problem to a WLSProblem
        size = 2 ** 10
        desc = elsa.VolumeDescriptor([size])
        b = elsa.DataContainer(desc, np.random.randn(size, 1))
        A = elsa.Scaling(desc, 2.0)
        regWeight = 4.0

        # this will throw an exception while performing sanity checks if pyelsa isn't configured properly
        tikhonov = elsa.TikhonovProblem(
            elsa.WLSProblem(A, b),
            elsa.RegularizationTerm(regWeight, elsa.L2NormPow2(desc)),
        )

        wls = elsa.WLSProblem(tikhonov)

        # build up the data term we expect wls to have and compare just to be sure
        blockOp = elsa.BlockLinearOperator(
            [A, elsa.Scaling(desc, math.sqrt(regWeight))],
            elsa.BlockLinearOperatorfBlockType.ROW,
        )
        blockVec = elsa.DataContainer(elsa.RandomBlocksDescriptor([desc, desc]))
        blockVec.getBlock(0).set(b)
        blockVec.getBlock(1).set(0)
        blockWls = elsa.L2NormPow2(elsa.LinearResidual(blockOp, blockVec))

        self.assertEqual(
            wls.getDataTerm(),
            blockWls,
            "TikhonovProblem was not correctly converted to WLSProblem",
        )
        self.assertEqual(
            len(wls.getRegularizationTerms()),
            0,
            "TikhonovProblem was not correctly converted to WLSProblem",
        )

        # test converting a WLSProblem to a QuadricProblem
        quadricProb = elsa.QuadricProblem(wls)

        # build up expected quadric
        quadricOp = elsa.adjoint(blockOp) * blockOp
        quadricVec = blockOp.applyAdjoint(blockVec)
        quadric = elsa.Quadric(quadricOp, quadricVec)

        self.assertEqual(
            quadricProb.getDataTerm(),
            quadric,
            "WLSProblem was not correctly converted to QuadricPforeach",
        )
        self.assertEqual(
            len(quadricProb.getRegularizationTerms()),
            0,
            "WLSProblem was not correctly converted to QuadricProblem",
        )

    def test_reconstruction_2d(self):
        "Try performing a simple reconstruction"

        size = 50
        numAngles = 50
        arc = 360
        noIterations = 20

        # generating a 2d phantom does not work for some reason
        # phantom = elsa.PhantomGenerator.createModifiedSheppLogan([size, size])

        # always generate 3d phantom and get middle slice on Python side
        phantom = elsa.PhantomGenerator.createModifiedSheppLogan([size, size, size])
        phantom = elsa.DataContainer(np.array(phantom, copy=False)[:, :, int(size / 2)])
        volDesc = phantom.getDataDescriptor()

        # generate circular trajectory
        sinoDesc = elsa.CircleTrajectoryGenerator.createTrajectory(
            numAngles, volDesc, arc, size * 100, size
        )

        # setup operator for 2d X-ray transform
        projector = elsa.SiddonsMethod(volDesc, sinoDesc)

        # simulate sinogram
        sino = projector.apply(phantom)

        # setup reconstruction problem
        problem = elsa.WLSProblem(projector, sino)

        # solve the reconstruction problem
        solver = elsa.CG(problem)
        recon = solver.solve(noIterations)

        # compute mse and check that it is within bounds
        mse = (recon - phantom).squaredL2Norm() / size ** 2
        self.assertLess(
            mse, 0.004, "Mean squared error of reconstruction too large",
        )

    def test_logging(self):
        "Test logging module interface"

        # the elsa.Logger appears to bypass the Python-side sys.stdout
        # so we only test file logging
        logfile_name = "test_pyelsa_log.txt"
        if os.path.exists(logfile_name):
            os.remove(logfile_name)
        elsa.Logger.enableFileLogging(logfile_name)

        elsa.Logger.setLevel(elsa.LogLevel.OFF)
        # this should produce no output when logging is off
        self.test_intermodule_object_exchange()
        elsa.Logger.flush()
        with open(logfile_name) as log_file:
            self.assertTrue(len(log_file.readline()) == 0)

        elsa.Logger.setLevel(elsa.LogLevel.TRACE)
        # this should produce some output when using a low logging level
        self.test_intermodule_object_exchange()
        elsa.Logger.flush()
        with open(logfile_name) as log_file:
            self.assertTrue(len(log_file.readline()) > 0)


if __name__ == "main":
    unittest.main()
