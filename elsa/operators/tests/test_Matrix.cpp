/**
 * @file test_Scaling.cpp
 *
 * @brief Tests for Scaling operator class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite
 * @author Tobias Lasser - minor extensions
 */

#include "doctest/doctest.h"
#include "Matrix.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

/*TEST_CASE_TEMPLATE("Matrix: Testing construction", data_t, float, double)
{
    GIVEN("a descriptor")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 17;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating an isotropic scaling operator")
        {
            real_t scaleFactor = 3.5f;
            Scaling<data_t> scalingOp(dd, scaleFactor);

            THEN("the descriptors are as expected")
            {
                REQUIRE_EQ(scalingOp.getDomainDescriptor(), dd);
                REQUIRE_EQ(scalingOp.getRangeDescriptor(), dd);
            }

            THEN("the scaling is isotropic and correct")
            {
                REQUIRE_UNARY(scalingOp.isIsotropic());
                REQUIRE_EQ(scalingOp.getScaleFactor(), scaleFactor);
                REQUIRE_THROWS_AS(scalingOp.getScaleFactors(), LogicError);
            }
        }

        WHEN("instantiating an anisotropic scaling operator")
        {
            DataContainer<data_t> dc(dd);
            dc = 3.5f;
            Scaling<data_t> scalingOp(dd, dc);

            THEN("the descriptors  are as expected")
            {
                REQUIRE_EQ(scalingOp.getDomainDescriptor(), dd);
                REQUIRE_EQ(scalingOp.getRangeDescriptor(), dd);
            }

            THEN("the scaling is anisotropic")
            {
                REQUIRE_UNARY_FALSE(scalingOp.isIsotropic());
                REQUIRE_EQ(scalingOp.getScaleFactors(), dc);
                REQUIRE_THROWS_AS(scalingOp.getScaleFactor(), LogicError);
            }
        }

        WHEN("cloning a scaling operator")
        {
            Scaling<data_t> scalingOp(dd, 3.5f);
            auto scalingOpClone = scalingOp.clone();

            THEN("everything matches")
            {
                REQUIRE_NE(scalingOpClone.get(), &scalingOp);
                REQUIRE_EQ(*scalingOpClone, scalingOp);
            }
        }
    }
}
*/

TEST_CASE_TEMPLATE("Matrix: Testing apply to data", data_t, float, double)
{
    GIVEN("some data")
    {
        Eigen::MatrixX<data_t> mat(3, 4);
        mat.setRandom();

        VolumeDescriptor dd({mat.rows(), mat.cols()});
        DataContainer<data_t> matDc(dd, Eigen::Map<Vector_t<data_t>>(mat.data(), mat.size()));
        Matrix<data_t> matOp(matDc);

        WHEN("applying the matrix to a vector")
        {
            Eigen::Vector4<data_t> vec{};
            vec.setRandom();
            DataContainer<data_t> vecDc(VolumeDescriptor({vec.size()}), vec);

            THEN("the result is correct")
            {
                auto result = matOp.apply(vecDc);
                auto expected = mat * vec;
                DataContainer<data_t> expectedDc(VolumeDescriptor({expected.size()}), expected);

                REQUIRE_EQ(result, expectedDc);
            }
        }

        WHEN("applying the adjoint of the matrix to a vector")
        {
            Eigen::Vector3<data_t> vec{};
            vec.setRandom();
            DataContainer<data_t> vecDc(VolumeDescriptor({vec.size()}), vec);

            THEN("the result is correct")
            {
                auto result = matOp.applyAdjoint(vecDc);
                auto expected = mat.transpose() * vec;
                DataContainer<data_t> expectedDc(VolumeDescriptor({expected.size()}), expected);

                REQUIRE_EQ(result, expectedDc);
            }
        }
    }
}
TEST_SUITE_END();
