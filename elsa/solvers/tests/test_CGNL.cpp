/**
* @file testCGNL.cpp
*
* @brief Tests for the Non-linear conjugate gradient solver
*
* @author Shen Hu - initial code
 */

#include "doctest/doctest.h"

#include "CGNL.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "Identity.h"
#include "WLSProblem.h"
#include "testHelpers.h"

//#include "AXDTStatRecon.h"
//#include "iostream"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

//TEST_CASE_TEMPLATE("CGNL: Solving a simple linear problem", data_t, float, double)
//{
//    srand((unsigned int) 666);
//
//    GIVEN("a linear problem")
//    {
//        IndexVector_t numCoeff(1);
//        numCoeff << 5;
//        VolumeDescriptor dd(numCoeff);
//
//        Vector_t<data_t> axdt_proj_raw(dd.getNumberOfCoefficients());
//        axdt_proj_raw.setConstant(1);
//        DataContainer<data_t> axdt_proj(dd, axdt_proj_raw);
//
//        Scaling<data_t> axdt_op(dd, 2);
//
//        AXDTStatRecon<data_t> func(axdt_proj, axdt_op, AXDTStatRecon<data_t>::Gaussian_log_d);
//
//        index_t numOfDims {1};
//        IndexVector_t dims(numOfDims);
//        dims << 1;
//        auto phDD = std::make_unique<VolumeDescriptor>(dims);
//        std::vector<std::unique_ptr<DataDescriptor>> descs;
//        descs.emplace_back(phDD->clone());
//        descs.emplace_back(dd.clone());
//        auto ansDD = RandomBlocksDescriptor(descs);
//        Vector_t<data_t> ans_raw(ansDD.getNumberOfCoefficients());
//        ans_raw.setConstant(0.5);
//        ans_raw[0] = 0;
//        DataContainer<data_t> ans(ansDD, ans_raw);
//
//        Problem<data_t> prob(func);
//
//        CGNL<data_t> cgnl(prob);
//
//        auto result = cgnl.solve(10);
//
//        for (int i = 0; i < result.getSize(); ++i) {
//            std::cout << result[i] << std::endl;
//        }
//    }
//}

TEST_CASE_TEMPLATE("CGNL: Solving a simple linear problem", data_t, float, double)
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);
    srand((unsigned int) 666);

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 10;
        VolumeDescriptor dd{numCoeff};

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b{dd, bVec};

        Identity<data_t> id{dd};

        WLSProblem<data_t> prob(id, b);

        CGNL<data_t> cgnl(prob);

        auto result = cgnl.solve(5);

        for (int i = 0; i < result.getSize(); ++i) {
            CHECK(checkApproxEq(result[i], bVec[i]));
        }
    }

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 10, 15;
        VolumeDescriptor dd{numCoeff};

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b{dd, bVec};

        Identity<data_t> id{dd};

        WLSProblem<data_t> prob(id, b);

        CGNL<data_t> cgnl(prob);

        auto result = cgnl.solve(5);

        for (int i = 0; i < result.getSize(); ++i) {
            CHECK(checkApproxEq(result[i], bVec[i]));
        }
    }

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 10;
        VolumeDescriptor dd{numCoeff};

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b{dd, bVec};

        Scaling<data_t> scale{dd, 5};

        WLSProblem<data_t> prob(scale, b);

        CGNL<data_t> cgnl(prob);

        auto result = cgnl.solve(5);

        for (int i = 0; i < result.getSize(); ++i) {
            CHECK_EQ(5. * result[i], doctest::Approx(bVec[i]));
        }
    }

    GIVEN("a linear problem")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 10, 15;
        VolumeDescriptor dd{numCoeff};

        Vector_t<data_t> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b{dd, bVec};

        Scaling<data_t> scale{dd, 0.5};

        WLSProblem<data_t> prob(scale, b);

        CGNL<data_t> cgnl(prob);

        auto result = cgnl.solve(5);

        for (int i = 0; i < result.getSize(); ++i) {
            CHECK_EQ(0.5 * result[i], doctest::Approx(bVec[i]));
        }
    }
}

TEST_SUITE_END();
