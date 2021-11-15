/**
 * @file test_DataContainer.cpp
 *
 * @brief Tests for DataContainer class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite to use doctest and BDD
 * @author Tobias Lasser - rewrite and added code coverage
 */

#include "doctest/doctest.h"
#include "DataContainer.h"
#include "IdenticalBlocksDescriptor.h"
#include "testHelpers.h"
#include "VolumeDescriptor.h"

#include <type_traits>

using namespace elsa;
using namespace doctest;

// Provides object to be used with TEMPLATE_PRODUCT_TEST_CASE, necessary because enum cannot be
// passed directly
template <typename T>
struct TestHelperGPU {
    static const DataHandlerType handler_t = DataHandlerType::GPU;
    using data_t = T;
};

// Provides object to be used with TEMPLATE_PRODUCT_TEST_CASE, necessary because enum cannot be
// passed directly
template <typename T>
struct TestHelperCPU {
    static const DataHandlerType handler_t = DataHandlerType::CPU;
    using data_t = T;
};

using CPUTypeTuple =
    std::tuple<TestHelperCPU<float>, TestHelperCPU<double>, TestHelperCPU<complex<float>>,
               TestHelperCPU<complex<double>>, TestHelperCPU<index_t>>;

TYPE_TO_STRING(TestHelperCPU<float>);
TYPE_TO_STRING(TestHelperCPU<double>);
TYPE_TO_STRING(TestHelperCPU<index_t>);
TYPE_TO_STRING(TestHelperCPU<complex<float>>);
TYPE_TO_STRING(TestHelperCPU<complex<double>>);

TYPE_TO_STRING(DataContainer<float>);
TYPE_TO_STRING(DataContainer<double>);
TYPE_TO_STRING(DataContainer<index_t>);
TYPE_TO_STRING(DataContainer<complex<float>>);
TYPE_TO_STRING(DataContainer<complex<double>>);

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

#ifdef ELSA_CUDA_VECTOR
using GPUTypeTuple =
    std::tuple<TestHelperGPU<float>, TestHelperGPU<double>, TestHelperGPU<complex<float>>,
               TestHelperGPU<complex<double>>, TestHelperGPU<index_t>>;

TYPE_TO_STRING(TestHelperGPU<float>);
TYPE_TO_STRING(TestHelperGPU<double>);
TYPE_TO_STRING(TestHelperGPU<index_t>);
TYPE_TO_STRING(TestHelperGPU<complex<float>>);
TYPE_TO_STRING(TestHelperGPU<complex<double>>);
#endif

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE_DEFINE("DataContainer: Testing construction", TestType,
                          datacontainer_construction)
{
    using data_t = typename TestType::data_t;

    INFO("Testing type: " << TypeName_v<const data_t>);

    GIVEN("a DataDescriptor")
    {
        VolumeDescriptor desc({17, 47, 91});

        WHEN("constructing an empty DataContainer")
        {
            DataContainer<data_t> dc(desc, TestType::handler_t);

            THEN("it has the correct DataDescriptor") { REQUIRE_EQ(dc.getDataDescriptor(), desc); }

            THEN("it has a data vector of correct size")
            {
                REQUIRE_EQ(dc.getSize(), desc.getNumberOfCoefficients());
            }
        }

        WHEN("constructing an initialized DataContainer")
        {
            auto data = generateRandomMatrix<data_t>(desc.getNumberOfCoefficients());

            DataContainer<data_t> dc(desc, data, TestType::handler_t);

            THEN("it has the correct DataDescriptor") { REQUIRE_EQ(dc.getDataDescriptor(), desc); }

            THEN("it has correctly initialized data")
            {
                REQUIRE_EQ(dc.getSize(), desc.getNumberOfCoefficients());

                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dc[i], data[i]));
            }
        }
    }

    GIVEN("another DataContainer")
    {
        VolumeDescriptor desc({32, 57});

        DataContainer<data_t> otherDc(desc, TestType::handler_t);

        auto randVec = generateRandomMatrix<data_t>(otherDc.getSize());
        for (index_t i = 0; i < otherDc.getSize(); ++i)
            otherDc[i] = randVec(i);

        WHEN("copy constructing")
        {
            DataContainer dc(otherDc);

            THEN("it copied correctly")
            {
                REQUIRE_EQ(dc.getDataDescriptor(), otherDc.getDataDescriptor());
                REQUIRE_NE(&dc.getDataDescriptor(), &otherDc.getDataDescriptor());

                REQUIRE_EQ(dc, otherDc);
            }
        }

        WHEN("copy assigning")
        {
            DataContainer<data_t> dc(desc, TestType::handler_t);
            dc = otherDc;

            THEN("it copied correctly")
            {
                REQUIRE_EQ(dc.getDataDescriptor(), otherDc.getDataDescriptor());
                REQUIRE_NE(&dc.getDataDescriptor(), &otherDc.getDataDescriptor());

                REQUIRE_EQ(dc, otherDc);
            }
        }

        WHEN("move constructing")
        {
            DataContainer oldOtherDc(otherDc);

            DataContainer dc(std::move(otherDc));

            THEN("it moved correctly")
            {
                REQUIRE_EQ(dc.getDataDescriptor(), oldOtherDc.getDataDescriptor());

                REQUIRE_EQ(dc, oldOtherDc);
            }

            THEN("the moved from object is still valid (but empty)") { otherDc = dc; }
        }

        WHEN("move assigning")
        {
            DataContainer oldOtherDc(otherDc);

            DataContainer<data_t> dc(desc, TestType::handler_t);
            dc = std::move(otherDc);

            THEN("it moved correctly")
            {
                REQUIRE_EQ(dc.getDataDescriptor(), oldOtherDc.getDataDescriptor());

                REQUIRE_EQ(dc, oldOtherDc);
            }

            THEN("the moved from object is still valid (but empty)") { otherDc = dc; }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataContainer: Testing the reduction operations", TestType,
                          datacontainer_reduction)
{
    using data_t = typename TestType::data_t;

    INFO("Testing type: " << TypeName_v<const data_t>);

    GIVEN("a DataContainer")
    {
        VolumeDescriptor desc({11, 73, 45});

        WHEN("putting in some random data")
        {
            auto [dc, randVec] = generateRandomContainer<data_t>(desc, TestType::handler_t);

            THEN("the reductions work a expected")
            {
                REQUIRE_UNARY(checkApproxEq(dc.sum(), randVec.sum()));
                REQUIRE_UNARY(checkApproxEq(
                    dc.l0PseudoNorm(),
                    (randVec.array().cwiseAbs()
                     >= std::numeric_limits<GetFloatingPointType_t<data_t>>::epsilon())
                        .count()));
                REQUIRE_UNARY(checkApproxEq(dc.l1Norm(), randVec.array().abs().sum()));
                REQUIRE_UNARY(checkApproxEq(dc.lInfNorm(), randVec.array().abs().maxCoeff()));
                REQUIRE_UNARY(checkApproxEq(dc.squaredL2Norm(), randVec.squaredNorm()));

                auto [dc2, randVec2] = generateRandomContainer<data_t>(desc, TestType::handler_t);

                REQUIRE_UNARY(checkApproxEq(dc.dot(dc2), randVec.dot(randVec2)));

                if constexpr (isComplex<data_t>) {
                    CHECK_THROWS(dc.minElement());
                    CHECK_THROWS(dc.maxElement());
                } else {
                    REQUIRE_UNARY(checkApproxEq(dc.minElement(), randVec.array().minCoeff()));
                    REQUIRE_UNARY(checkApproxEq(dc.maxElement(), randVec.array().maxCoeff()));
                }
            }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataContainer: Testing element-wise access", TestType,
                          datacontainer_elemwise)
{
    using data_t = typename TestType::data_t;

    INFO("Testing type: " << TypeName_v<const data_t>);

    GIVEN("a DataContainer")
    {
        VolumeDescriptor desc({47, 11});
        DataContainer<data_t> dc(desc, TestType::handler_t);

        WHEN("accessing the elements")
        {
            IndexVector_t coord(2);
            coord << 17, 4;
            index_t index = desc.getIndexFromCoordinate(coord);

            THEN("it works as expected when using indices/coordinates")
            {
                // For integral typeps don't have a floating point value
                if constexpr (std::is_integral_v<data_t>) {
                    dc[index] = data_t(2);
                    REQUIRE_UNARY(checkApproxEq(dc[index], 2));
                    REQUIRE_UNARY(checkApproxEq(dc(coord), 2));
                    REQUIRE_UNARY(checkApproxEq(dc(17, 4), 2));

                    dc(coord) = data_t(3);
                    REQUIRE_UNARY(checkApproxEq(dc[index], 3));
                    REQUIRE_UNARY(checkApproxEq(dc(coord), 3));
                    REQUIRE_UNARY(checkApproxEq(dc(17, 4), 3));
                } else {
                    dc[index] = data_t(2.2f);
                    REQUIRE_UNARY(checkApproxEq(dc[index], 2.2f));
                    REQUIRE_UNARY(checkApproxEq(dc(coord), 2.2f));
                    REQUIRE_UNARY(checkApproxEq(dc(17, 4), 2.2f));

                    dc(coord) = data_t(3.3f);
                    REQUIRE_UNARY(checkApproxEq(dc[index], 3.3f));
                    REQUIRE_UNARY(checkApproxEq(dc(coord), 3.3f));
                    REQUIRE_UNARY(checkApproxEq(dc(17, 4), 3.3f));
                }
            }
        }
    }

    GIVEN("a DataContainer")
    {
        VolumeDescriptor desc({47, 11});

        WHEN("putting in some random data")
        {
            auto [dc, randVec] = generateRandomContainer<data_t>(desc, TestType::handler_t);

            THEN("the element-wise unary operations work as expected")
            {
                DataContainer dcAbs = cwiseAbs(dc);
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dcAbs[i], randVec.array().abs()[i]));

                DataContainer dcSquare = square(dc);
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dcSquare[i], randVec.array().square()[i]));
                DataContainer dcSqrt = sqrt(dcSquare);
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dcSqrt[i], randVec.array().square().sqrt()[i]));

                // do exponent check only for floating point types as for integer will likely
                // lead to overflow due to random init over full value range
                if constexpr (!std::is_integral_v<data_t>) {
                    DataContainer dcExp = exp(dc);
                    for (index_t i = 0; i < dc.getSize(); ++i)
                        REQUIRE_UNARY(checkApproxEq(dcExp[i], randVec.array().exp()[i]));
                }

                DataContainer dcLog = log(dcSquare);
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dcLog[i], randVec.array().square().log()[i]));

                DataContainer dcReal = real(dc);
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dcReal[i], randVec.array().real()[i]));

                DataContainer dcImag = imag(dc);

                if constexpr (isComplex<data_t>) {
                    for (index_t i = 0; i < dc.getSize(); ++i)
                        REQUIRE_UNARY(checkApproxEq(dcImag[i], randVec.array().imag()[i]));
                } else {
                    for (index_t i = 0; i < dc.getSize(); ++i)
                        REQUIRE_UNARY(checkApproxEq(dcImag[i], 0));
                }
            }

            auto scalar = static_cast<data_t>(923.41f);

            THEN("the binary in-place addition of a scalar work as expected")
            {
                dc += scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dc[i], randVec(i) + scalar));
            }

            THEN("the binary in-place subtraction of a scalar work as expected")
            {
                dc -= scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dc[i], randVec(i) - scalar));
            }

            THEN("the binary in-place multiplication with a scalar work as expected")
            {
                dc *= scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dc[i], randVec(i) * scalar));
            }

            THEN("the binary in-place division by a scalar work as expected")
            {
                dc /= scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dc[i], randVec(i) / scalar));
            }

            THEN("the element-wise assignment of a scalar works as expected")
            {
                dc = scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dc[i], scalar));
            }
        }

        WHEN("having two containers with random data")
        {
            auto [dc, randVec] = generateRandomContainer<data_t>(desc, TestType::handler_t);
            auto [dc2, randVec2] = generateRandomContainer<data_t>(desc, TestType::handler_t);

            THEN("the element-wise in-place addition works as expected")
            {
                dc += dc2;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dc[i], randVec(i) + randVec2(i)));
            }

            THEN("the element-wise in-place subtraction works as expected")
            {
                dc -= dc2;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dc[i], randVec(i) - randVec2(i)));
            }

            THEN("the element-wise in-place multiplication works as expected")
            {
                dc *= dc2;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dc[i], randVec(i) * randVec2(i)));
            }

            THEN("the element-wise in-place division works as expected")
            {
                dc /= dc2;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    if (dc2[i] != data_t(0))
                        REQUIRE_UNARY(checkApproxEq(dc[i], randVec(i) / randVec2(i)));
            }
        }

        WHEN("having two containers with real and complex data each")
        {
            auto [dcReals1, realsVec1] = generateRandomContainer<real_t>(desc, TestType::handler_t);
            auto [dcComps1, compsVec1] =
                generateRandomContainer<complex<real_t>>(desc, TestType::handler_t);

            THEN("the element-wise maximum operation works as expected for two real "
                 "DataContainers")
            {
                auto [dcReals2, realsVec2] =
                    generateRandomContainer<real_t>(desc, TestType::handler_t);

                DataContainer dcCWiseMax = cwiseMax(dcReals1, dcReals2);
                for (index_t i = 0; i < dcCWiseMax.getSize(); ++i)
                    REQUIRE_UNARY(
                        checkApproxEq(dcCWiseMax[i], realsVec1.array().max(realsVec2.array())[i]));
            }

            THEN("the element-wise maximum operation works as expected for a real and a complex "
                 "DataContainer")
            {
                auto [dcComps2, compsVec2] =
                    generateRandomContainer<complex<real_t>>(desc, TestType::handler_t);

                DataContainer dcCWiseMax = cwiseMax(dcReals1, dcComps2);
                for (index_t i = 0; i < dcCWiseMax.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dcCWiseMax[i],
                                                realsVec1.array().max(compsVec2.array().abs())[i]));
            }

            THEN("the element-wise maximum operation works as expected for a complex and a real "
                 "DataContainer")
            {
                auto [dcComps2, compsVec2] =
                    generateRandomContainer<complex<real_t>>(desc, TestType::handler_t);

                DataContainer dcCWiseMax = cwiseMax(dcComps2, dcReals1);
                for (index_t i = 0; i < dcCWiseMax.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(dcCWiseMax[i],
                                                compsVec2.array().abs().max(realsVec1.array())[i]));
            }

            THEN("the element-wise maximum operation works as expected for two DataContainers")
            {
                auto [dcComps2, compsVec2] =
                    generateRandomContainer<complex<real_t>>(desc, TestType::handler_t);

                DataContainer dcCWiseMax = cwiseMax(dcComps1, dcComps2);
                for (index_t i = 0; i < dcCWiseMax.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(
                        dcCWiseMax[i], compsVec1.array().abs().max(compsVec2.array().abs())[i]));
            }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE(
    "DataContainer: Testing the arithmetic operations with DataContainers arguments", TestType,
    datacontainer_arithmetic)
{
    using data_t = typename TestType::data_t;

    INFO("Testing type: " << TypeName_v<const data_t>);

    GIVEN("some DataContainers")
    {
        VolumeDescriptor desc({52, 7, 29});

        auto [dc, randVec] = generateRandomContainer<data_t>(desc, TestType::handler_t);
        auto [dc2, randVec2] = generateRandomContainer<data_t>(desc, TestType::handler_t);

        THEN("the binary element-wise operations work as expected")
        {
            DataContainer resultPlus = dc + dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE_UNARY(checkApproxEq(resultPlus[i], dc[i] + dc2[i]));

            DataContainer resultMinus = dc - dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE_UNARY(checkApproxEq(resultMinus[i], dc[i] - dc2[i]));

            DataContainer resultMult = dc * dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE_UNARY(checkApproxEq(resultMult[i], dc[i] * dc2[i]));

            DataContainer resultDiv = dc / dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                if (dc2[i] != data_t(0))
                    REQUIRE_UNARY(checkApproxEq(resultDiv[i], dc[i] / dc2[i]));
        }

        THEN("the operations with a scalar work as expected")
        {
            data_t scalar = static_cast<data_t>(4.92f);

            DataContainer resultScalarPlus = scalar + dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE_UNARY(checkApproxEq(resultScalarPlus[i], scalar + dc[i]));

            DataContainer resultPlusScalar = dc + scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE_UNARY(checkApproxEq(resultPlusScalar[i], dc[i] + scalar));

            DataContainer resultScalarMinus = scalar - dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE_UNARY(checkApproxEq(resultScalarMinus[i], scalar - dc[i]));

            DataContainer resultMinusScalar = dc - scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE_UNARY(checkApproxEq(resultMinusScalar[i], dc[i] - scalar));

            DataContainer resultScalarMult = scalar * dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE_UNARY(checkApproxEq(resultScalarMult[i], scalar * dc[i]));

            DataContainer resultMultScalar = dc * scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE_UNARY(checkApproxEq(resultMultScalar[i], dc[i] * scalar));

            DataContainer resultScalarDiv = scalar / dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                if (dc[i] != data_t(0))
                    REQUIRE_UNARY(checkApproxEq(resultScalarDiv[i], scalar / dc[i]));

            DataContainer resultDivScalar = dc / scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE_UNARY(checkApproxEq(resultDivScalar[i], dc[i] / scalar));
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataContainer: Testing creation of Maps through DataContainer", TestType,
                          datacontainer_maps)
{
    using data_t = typename TestType::data_t;

    INFO("Testing type: " << TypeName_v<const data_t>);

    GIVEN("a non-blocked container")
    {
        VolumeDescriptor desc({52, 7, 29});

        DataContainer<data_t> dc(desc, TestType::handler_t);
        const DataContainer<data_t> constDc(desc, TestType::handler_t);

        WHEN("trying to reference a block")
        {
            THEN("an exception occurs")
            {
                REQUIRE_THROWS(dc.getBlock(0));
                REQUIRE_THROWS(constDc.getBlock(0));
            }
        }

        WHEN("creating a view")
        {
            VolumeDescriptor linearDesc({desc.getNumberOfCoefficients()});
            auto linearDc = dc.viewAs(linearDesc);
            auto linearConstDc = constDc.viewAs(linearDesc);

            THEN("view has the correct descriptor and data")
            {
                REQUIRE_EQ(linearDesc, linearDc.getDataDescriptor());
                REQUIRE_EQ(&linearDc[0], &dc[0]);

                REQUIRE_EQ(linearDesc, linearConstDc.getDataDescriptor());
                REQUIRE_EQ(&linearConstDc[0], &constDc[0]);

                AND_THEN("view is not a shallow copy")
                {
                    const auto dcCopy = dc;
                    const auto constDcCopy = constDc;

                    linearDc[0] = 1;
                    REQUIRE_EQ(&linearDc[0], &dc[0]);
                    REQUIRE_NE(&linearDc[0], &dcCopy[0]);

                    linearConstDc[0] = 1;
                    REQUIRE_EQ(&linearConstDc[0], &constDc[0]);
                    REQUIRE_NE(&linearConstDc[0], &constDcCopy[0]);
                }
            }
        }
    }

    GIVEN("a blocked container")
    {
        VolumeDescriptor desc({52, 29});
        index_t numBlocks = 7;
        IdenticalBlocksDescriptor blockDesc(numBlocks, desc);

        DataContainer<data_t> dc(blockDesc, TestType::handler_t);
        const DataContainer<data_t> constDc(blockDesc, TestType::handler_t);

        WHEN("referencing a block")
        {
            THEN("block has the correct descriptor and data")
            {
                for (index_t i = 0; i < numBlocks; i++) {
                    auto dcBlock = dc.getBlock(i);
                    const auto constDcBlock = constDc.getBlock(i);

                    REQUIRE_EQ(dcBlock.getDataDescriptor(), blockDesc.getDescriptorOfBlock(i));
                    REQUIRE_EQ(&dcBlock[0], &dc[0] + blockDesc.getOffsetOfBlock(i));

                    REQUIRE_EQ(constDcBlock.getDataDescriptor(), blockDesc.getDescriptorOfBlock(i));
                    REQUIRE_EQ(&constDcBlock[0], &constDc[0] + blockDesc.getOffsetOfBlock(i));
                }
            }
        }

        WHEN("creating a view")
        {
            IndexVector_t numCoeff(1);
            numCoeff << blockDesc.getNumberOfCoefficients();
            VolumeDescriptor linearDesc(numCoeff);
            auto linearDc = dc.viewAs(linearDesc);
            auto linearConstDc = constDc.viewAs(linearDesc);

            THEN("view has the correct descriptor and data")
            {
                REQUIRE_EQ(linearDesc, linearDc.getDataDescriptor());
                REQUIRE_EQ(&linearDc[0], &dc[0]);

                REQUIRE_EQ(linearDesc, linearConstDc.getDataDescriptor());
                REQUIRE_EQ(&linearConstDc[0], &constDc[0]);

                AND_THEN("view is not a shallow copy")
                {
                    const auto dcCopy = dc;
                    const auto constDcCopy = constDc;

                    linearDc[0] = 1;
                    REQUIRE_EQ(&linearDc[0], &dc[0]);
                    REQUIRE_NE(&linearDc[0], &dcCopy[0]);

                    linearConstDc[0] = 1;
                    REQUIRE_EQ(&linearConstDc[0], &constDc[0]);
                    REQUIRE_NE(&linearConstDc[0], &constDcCopy[0]);
                }
            }
        }
    }
}

#ifdef ELSA_CUDA_VECTOR
TEST_CASE_TEMPLATE("DataContainer: Testing load data to GPU and vice versa", TestType, float,
                   double, complex<float>, complex<double>, index_t)
{
    GIVEN("A CPU DataContainer with random data")
    {
        VolumeDescriptor desc({52, 7, 29});

        DataContainer<TestType> dcCPU(desc, DataHandlerType::CPU);
        DataContainer<TestType> dcGPU(desc, DataHandlerType::GPU);

        auto randVec = generateRandomMatrix<TestType>(dcCPU.getSize());

        for (index_t i = 0; i < dcCPU.getSize(); ++i) {
            dcCPU[i] = randVec(i);
            dcGPU[i] = randVec(i);
        }

        WHEN("Trying to call loadToCPU on CPU container")
        {
            THEN("Throws") { REQUIRE_THROWS(dcCPU.loadToCPU()); }
        }

        WHEN("Trying to call loadToGPU on GPU container")
        {
            THEN("Throws") { REQUIRE_THROWS(dcGPU.loadToGPU()); }
        }

        WHEN("Loading to GPU from CPU")
        {
            DataContainer dcGPU2 = dcCPU.loadToGPU();

            REQUIRE_EQ(dcGPU2.getDataHandlerType(), DataHandlerType::GPU);

            THEN("all elements have to be the same")
            {
                for (index_t i = 0; i < dcCPU.getSize(); ++i) {
                    REQUIRE_UNARY(checkApproxEq(dcGPU2[i], dcGPU[i]));
                }
            }
        }

        WHEN("Loading to CPU from GPU")
        {
            DataContainer dcCPU2 = dcGPU.loadToCPU();

            REQUIRE_EQ(dcCPU2.getDataHandlerType(), DataHandlerType::CPU);

            THEN("all elements have to be the same")
            {
                for (index_t i = 0; i < dcCPU.getSize(); ++i) {
                    REQUIRE_UNARY(checkApproxEq(dcCPU2[i], dcGPU[i]));
                }
            }
        }

        WHEN("copy-assigning a GPU to a CPU container")
        {
            dcCPU = dcGPU;

            THEN("it should be a GPU container")
            {
                REQUIRE_EQ(dcCPU.getDataHandlerType(), DataHandlerType::GPU);
            }

            AND_THEN("they should be equal")
            {
                REQUIRE_EQ(dcCPU, dcGPU);
                REQUIRE_EQ(dcCPU.getSize(), dcGPU.getSize());

                for (index_t i = 0; i < dcCPU.getSize(); ++i) {
                    REQUIRE_UNARY(checkApproxEq(dcCPU[i], dcGPU[i]));
                }
            }
        }

        WHEN("copy-assigning a CPU to a GPU container")
        {
            dcGPU = dcCPU;

            THEN("it should be a GPU container")
            {
                REQUIRE_EQ(dcGPU.getDataHandlerType(), DataHandlerType::CPU);
            }

            AND_THEN("they should be equal")
            {
                REQUIRE_EQ(dcCPU, dcGPU);
                REQUIRE_EQ(dcCPU.getSize(), dcGPU.getSize());

                for (index_t i = 0; i < dcCPU.getSize(); ++i) {
                    REQUIRE_UNARY(checkApproxEq(dcCPU[i], dcGPU[i]));
                }
            }
        }
    }
}
#endif

TEST_CASE("DataContainer: Testing iterators for DataContainer")
{
    GIVEN("A 1D container")
    {
        constexpr index_t size = 20;
        VolumeDescriptor desc({size});

        DataContainer dc1(desc);
        DataContainer dc2(desc);

        Eigen::VectorXf randVec1 = Eigen::VectorXf::Random(size);
        Eigen::VectorXf randVec2 = Eigen::VectorXf::Random(size);

        for (index_t i = 0; i < size; ++i) {
            dc1[i] = randVec1(i);
            dc2[i] = randVec2(i);
        }

        THEN("We can iterate forward")
        {
            int i = 0;
            for (auto v = dc1.cbegin(); v != dc1.cend(); v++) {
                REQUIRE_UNARY(checkApproxEq(*v, randVec1[i++]));
            }
            REQUIRE_EQ(i, size);
        }

        THEN("We can iterate backward")
        {
            int i = size;
            for (auto v = dc1.crbegin(); v != dc1.crend(); v++) {
                REQUIRE_UNARY(checkApproxEq(*v, randVec1[--i]));
            }
            REQUIRE_EQ(i, 0);
        }

        THEN("We can iterate and mutate")
        {
            int i = 0;
            for (auto& v : dc1) {
                v = v * 2;
                REQUIRE_UNARY(checkApproxEq(v, 2 * randVec1[i++]));
            }
            REQUIRE_EQ(i, size);

            i = 0;
            for (auto v : dc1) {
                REQUIRE_UNARY(checkApproxEq(v, 2 * randVec1[i++]));
            }
            REQUIRE_EQ(i, size);
        }

        THEN("We can use STL algorithms")
        {
            REQUIRE_EQ(*std::min_element(dc1.cbegin(), dc1.cend()), randVec1.minCoeff());
            REQUIRE_EQ(*std::max_element(dc1.cbegin(), dc1.cend()), randVec1.maxCoeff());
        }
    }
    GIVEN("A 2D container")
    {
        constexpr index_t size = 20;
        VolumeDescriptor desc({size, size});

        DataContainer dc1(desc);

        Eigen::VectorXf randVec1 = Eigen::VectorXf::Random(size * size);

        for (index_t i = 0; i < dc1.getSize(); ++i) {
            dc1[i] = randVec1[i];
        }

        THEN("We can iterate forward")
        {
            int i = 0;
            for (auto v : dc1) {
                REQUIRE_UNARY(checkApproxEq(v, randVec1[i++]));
            }
            REQUIRE_EQ(i, size * size);
        }

        THEN("We can iterate backward")
        {
            int i = size * size;
            for (auto v = dc1.crbegin(); v != dc1.crend(); v++) {
                REQUIRE_UNARY(checkApproxEq(*v, randVec1[--i]));
            }
            REQUIRE_EQ(i, 0);
        }

        THEN("We can iterate and mutate")
        {
            int i = 0;
            for (auto& v : dc1) {
                v = v * 2;
                REQUIRE_UNARY(checkApproxEq(v, 2 * randVec1[i++]));
            }
            REQUIRE_EQ(i, size * size);

            i = 0;
            for (auto v : dc1) {
                REQUIRE_UNARY(checkApproxEq(v, 2 * randVec1[i++]));
            }
            REQUIRE_EQ(i, size * size);
        }

        THEN("We can use STL algorithms")
        {
            REQUIRE_EQ(*std::min_element(dc1.cbegin(), dc1.cend()), randVec1.minCoeff());
            REQUIRE_EQ(*std::max_element(dc1.cbegin(), dc1.cend()), randVec1.maxCoeff());
        }
    }
    GIVEN("A 3D container")
    {
        constexpr index_t size = 20;
        VolumeDescriptor desc({size, size, size});

        DataContainer dc1(desc);

        Eigen::VectorXf randVec1 = Eigen::VectorXf::Random(size * size * size);

        for (index_t i = 0; i < dc1.getSize(); ++i) {
            dc1[i] = randVec1[i];
        }

        THEN("We can iterate forward")
        {
            int i = 0;
            for (auto v : dc1) {
                REQUIRE_UNARY(checkApproxEq(v, randVec1[i++]));
            }
            REQUIRE_EQ(i, size * size * size);
        }

        THEN("We can iterate backward")
        {
            int i = size * size * size;
            for (auto v = dc1.crbegin(); v != dc1.crend(); v++) {
                REQUIRE_UNARY(checkApproxEq(*v, randVec1[--i]));
            }
            REQUIRE_EQ(i, 0);
        }

        THEN("We can iterate and mutate")
        {
            int i = 0;
            for (auto& v : dc1) {
                v = v * 2;
                REQUIRE_UNARY(checkApproxEq(v, 2 * randVec1[i++]));
            }
            REQUIRE_EQ(i, size * size * size);

            i = 0;
            for (auto v : dc1) {
                REQUIRE_UNARY(checkApproxEq(v, 2 * randVec1[i++]));
            }
            REQUIRE_EQ(i, size * size * size);
        }

        THEN("We can use STL algorithms")
        {
            REQUIRE_EQ(*std::min_element(dc1.cbegin(), dc1.cend()), randVec1.minCoeff());
            REQUIRE_EQ(*std::max_element(dc1.cbegin(), dc1.cend()), randVec1.maxCoeff());
        }
    }
}

TEST_CASE_TEMPLATE("DataContainer: Concatenate two DataContainers", data_t, float, double,
                   complex<float>, complex<double>)
{
    GIVEN("Two equally sized 1D data containers")
    {
        constexpr index_t size = 20;
        VolumeDescriptor desc({size});

        Vector_t<data_t> randVec1 = Vector_t<data_t>::Random(size);
        Vector_t<data_t> randVec2 = Vector_t<data_t>::Random(size);

        DataContainer dc1(desc, randVec1);
        DataContainer dc2(desc, randVec2);

        auto concated = concatenate(dc1, dc2);
        THEN("The size of the concatenated DataContainer is twice the original one")
        {
            REQUIRE_EQ(concated.getSize(), 2 * size);
        }

        THEN("The values correspond to the original DataContainers")
        {
            for (int i = 0; i < size; ++i) {
                INFO("Error at position: ", i);
                REQUIRE_EQ(concated[i], randVec1[i]);
            }

            for (int i = 0; i < size; ++i) {
                INFO("Error at position: ", i + size);
                REQUIRE_EQ(concated[i + size], randVec2[i]);
            }
        }
    }

    GIVEN("Two differently sized 1D data containers")
    {
        constexpr index_t size1 = 20;
        VolumeDescriptor desc1({size1});

        constexpr index_t size2 = 10;
        VolumeDescriptor desc2({size2});

        Vector_t<data_t> randVec1 = Vector_t<data_t>::Random(size1);
        Vector_t<data_t> randVec2 = Vector_t<data_t>::Random(size2);

        DataContainer dc1(desc1, randVec1);
        DataContainer dc2(desc2, randVec2);

        auto concated = concatenate(dc1, dc2);

        THEN("The size of the concatenated DataContainer is twice the original one")
        {
            REQUIRE_EQ(concated.getSize(), size1 + size2);
        }

        THEN("The values correspond to the original DataContainers")
        {
            for (int i = 0; i < size1; ++i) {
                INFO("Error at position: ", i);
                REQUIRE_EQ(concated[i], randVec1[i]);
            }

            for (int i = 0; i < size2; ++i) {
                INFO("Error at position: ", i + size1);
                REQUIRE_EQ(concated[i + size1], randVec2[i]);
            }
        }
    }

    GIVEN("Two equally sized 2D data containers")
    {
        constexpr index_t size = 20;
        VolumeDescriptor desc({size, size});

        Vector_t<data_t> randVec1 = Vector_t<data_t>::Random(size * size);
        Vector_t<data_t> randVec2 = Vector_t<data_t>::Random(size * size);

        DataContainer dc1(desc, randVec1);
        DataContainer dc2(desc, randVec2);

        auto concated = concatenate(dc1, dc2);
        THEN("The size of the concatenated DataContainer is twice the original one")
        {
            REQUIRE_EQ(concated.getSize(), 2 * (size * size));
        }

        THEN("The values correspond to the original DataContainers")
        {
            for (int i = 0; i < size * size; ++i) {
                INFO("Error at position: ", i);
                REQUIRE_EQ(concated[i], randVec1[i]);
            }

            for (int i = 0; i < size * size; ++i) {
                INFO("Error at position: ", i + size);
                REQUIRE_EQ(concated[i + size * size], randVec2[i]);
            }
        }
    }

    GIVEN("DataContainers of different dimension")
    {
        VolumeDescriptor desc1D({20});
        VolumeDescriptor desc2D({20, 20});

        DataContainer dc1(desc1D);
        DataContainer dc2(desc2D);

        THEN("The concatenation throws") { REQUIRE_THROWS_AS(concatenate(dc1, dc2), LogicError); }
    }
}

TEST_CASE_TEMPLATE("DataContainer: Slice a DataContainer", data_t, float, double, complex<float>,
                   complex<double>)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    GIVEN("A non 3D DataContainer")
    {
        constexpr index_t size = 20;

        const VolumeDescriptor desc({size, size});
        const Vector_t<data_t> randVec = Vector_t<data_t>::Random(size * size);
        const DataContainer<data_t> dc(desc, randVec);

        THEN("Accessing an out of bounds slice throws")
        {
            REQUIRE_THROWS_AS(dc.slice(20), LogicError);
        }

        WHEN("Accessing all the slices")
        {
            for (int i = 0; i < size; ++i) {
                auto slice = dc.slice(i);

                THEN("The the slice is a 2D slice of \"thickness\" 1")
                {
                    REQUIRE_EQ(slice.getDataDescriptor().getNumberOfDimensions(), 2);

                    auto coeffs = slice.getDataDescriptor().getNumberOfCoefficientsPerDimension();
                    auto expectedCoeffs = IndexVector_t(2);
                    expectedCoeffs << size, 1;
                    REQUIRE_EQ(coeffs, expectedCoeffs);
                }

                THEN("All values are the same as of the original DataContainer")
                {
                    // Check that it's read correctly
                    auto vecSlice = randVec.segment(i * size, size);
                    for (int j = 0; j < size; ++j) {
                        REQUIRE_UNARY(checkApproxEq(slice(j, 0), vecSlice[j]));
                    }
                }
            }
        }
    }

    GIVEN("A const 3D DataContainer")
    {
        constexpr index_t size = 20;

        const VolumeDescriptor desc({size, size, size});
        const Vector_t<data_t> randVec = Vector_t<data_t>::Random(size * size * size);
        const DataContainer<data_t> dc(desc, randVec);

        THEN("Accessing an out of bounds slice throws")
        {
            REQUIRE_THROWS_AS(dc.slice(20), LogicError);
        }

        WHEN("Accessing all the slices")
        {
            for (int i = 0; i < size; ++i) {
                auto slice = dc.slice(i);

                THEN("The the slice is a 3D slice of \"thickness\" 1")
                {
                    REQUIRE_EQ(slice.getDataDescriptor().getNumberOfDimensions(), 3);

                    auto coeffs = slice.getDataDescriptor().getNumberOfCoefficientsPerDimension();
                    auto expectedCoeffs = IndexVector_t(3);
                    expectedCoeffs << size, size, 1;
                    REQUIRE_EQ(coeffs, expectedCoeffs);
                }

                THEN("All values are the same as of the original DataContainer")
                {
                    // Check that it's read correctly
                    auto vecSlice = randVec.segment(i * size * size, size * size);
                    for (int j = 0; j < size; ++j) {
                        for (int k = 0; k < size; ++k) {
                            REQUIRE_UNARY(checkApproxEq(slice(k, j, 0), vecSlice[k + j * size]));
                        }
                    }
                }
            }
        }
    }

    GIVEN("A non-const 3D DataContainer")
    {
        constexpr index_t size = 20;

        const VolumeDescriptor desc({size, size, size});
        DataContainer<data_t> dc(desc);
        dc = 0;

        THEN("Accessing an out of bounds slice throws")
        {
            REQUIRE_THROWS_AS(dc.slice(20), LogicError);
        }

        WHEN("Setting the first slice to 1")
        {
            dc.slice(0) = 1;

            THEN("Only the first slice is set to 1")
            {
                for (int j = 0; j < size; ++j) {
                    for (int i = 0; i < size; ++i) {
                        data_t val = dc(i, j, 0);
                        INFO("Expected slice 0 to be ", data_t{1}, " but it's ", val, " (at (", i,
                             ", ", j, ", 0))");
                        REQUIRE_UNARY(checkApproxEq(val, 1));
                    }
                }
            }

            THEN("The other slices are still set to 0")
            {
                for (int k = 1; k < size; ++k) {
                    for (int j = 0; j < size; ++j) {
                        for (int i = 0; i < size; ++i) {
                            data_t val = dc(i, j, k);
                            INFO("Expected all slices but the first to be ", data_t{0},
                                 " but it's ", val, " (at (", i, ", ", j, ", 0))");
                            REQUIRE_UNARY(checkApproxEq(val, 0));
                        }
                    }
                }
            }
        }

        WHEN("Setting the fifth slice to some random data using a 3D DataContainer")
        {
            Vector_t<data_t> randVec = Vector_t<data_t>::Random(size * size * 1);
            const DataContainer slice(VolumeDescriptor({size, size, 1}), randVec);

            dc.slice(5) = slice;
            THEN("The first 4 slices are still zero")
            {
                for (int k = 0; k < 5; ++k) {
                    for (int j = 0; j < size; ++j) {
                        for (int i = 0; i < size; ++i) {
                            data_t val = dc(i, j, k);

                            INFO("Expected all slices but the first to be ", data_t{0},
                                 " but it's ", val, " (at (", i, ", ", j, ", 0))");
                            REQUIRE_UNARY(checkApproxEq(val, 0));
                        }
                    }
                }
            }

            THEN("The fifth slices set correctly")
            {
                for (int j = 0; j < size; ++j) {
                    for (int i = 0; i < size; ++i) {
                        data_t val = dc(i, j, 5);
                        auto expected = randVec[i + j * size];
                        INFO("Expected slice 0 to be ", expected, " but it's ", val, " (at (", i,
                             ", ", j, ", 0))");
                        REQUIRE_UNARY(checkApproxEq(val, expected));
                    }
                }
            }

            THEN("The last 14 slices are still zero")
            {
                // Check last slices
                for (int k = 6; k < size; ++k) {
                    for (int j = 0; j < size; ++j) {
                        for (int i = 0; i < size; ++i) {
                            data_t val = dc(i, j, k);

                            INFO("Expected all slices but the first to be ", data_t{0},
                                 " but it's ", val, " (at (", i, ", ", j, ", 0))");
                            REQUIRE_UNARY(checkApproxEq(val, 0));
                        }
                    }
                }
            }
        }

        WHEN("Setting the first slice to some random data using a 2D DataContainer")
        {
            Vector_t<data_t> randVec = Vector_t<data_t>::Random(size * size);
            const DataContainer slice(VolumeDescriptor({size, size}), randVec);

            dc.slice(0) = slice;
            THEN("The fifth slices set correctly")
            {
                for (int j = 0; j < size; ++j) {
                    for (int i = 0; i < size; ++i) {
                        data_t val = dc(i, j, 0);
                        auto expected = randVec[i + j * size];
                        INFO("Expected slice 0 to be ", expected, " but it's ", val, " (at (", i,
                             ", ", j, ", 0))");
                        REQUIRE_UNARY(checkApproxEq(val, expected));
                    }
                }
            }
            THEN("The other slices are still zero")
            {
                for (int k = 1; k < size; ++k) {
                    for (int j = 0; j < size; ++j) {
                        for (int i = 0; i < size; ++i) {
                            data_t val = dc(i, j, k);
                            INFO("Expected all slices but the first to be ", data_t{0},
                                 " but it's ", val, " (at (", i, ", ", j, ", 0))");
                            REQUIRE_UNARY(checkApproxEq(val, 0));
                        }
                    }
                }
            }
        }
    }

    GIVEN("a 3D DataDescriptor and a 3D random Vector")
    {
        constexpr index_t size = 28;
        constexpr index_t one = 1;

        const VolumeDescriptor desc({size, size, one});
        const Vector_t<data_t> randVec = Vector_t<data_t>::Random(size * size * one);

        WHEN("slicing a non-const DataContainer with the size of the last dimension of 1")
        {
            DataContainer<data_t> dc(desc, randVec);

            DataContainer<data_t> res = dc.slice(0);

            THEN("the DataContainers match") { REQUIRE_EQ(dc, res); }
        }

        WHEN("slicing a const DataContainer with the size of the last dimension of 1")
        {
            const DataContainer<data_t> dc(desc, randVec);

            const DataContainer<data_t> res = dc.slice(0);

            THEN("the DataContainers match") { REQUIRE_EQ(dc, res); }
        }
    }
}

TEST_CASE_TEMPLATE("DataContainer: FFT shift and IFFT shift a DataContainer", data_t, float, double,
                   complex<float>, complex<double>)
{
    GIVEN("a one-element 2D data container")
    {
        DataContainer<data_t> dc(VolumeDescriptor{{1, 1}});
        dc[0] = 8;
        WHEN("running the FFT shift operation to the container")
        {
            DataContainer<data_t> fftShiftedDC = fftShift2D(dc);
            THEN("the data descriptors match")
            {
                REQUIRE_EQ(dc.getDataDescriptor(), fftShiftedDC.getDataDescriptor());
            }
            THEN("the data containers match") { REQUIRE_UNARY(fftShiftedDC == dc); }
        }

        WHEN("running the IFFT shift operation to the container")
        {
            DataContainer<data_t> ifftShiftedDC = ifftShift2D(dc);
            THEN("the data descriptors match")
            {
                REQUIRE_EQ(dc.getDataDescriptor(), ifftShiftedDC.getDataDescriptor());
            }
            THEN("the data containers match") { REQUIRE_UNARY(ifftShiftedDC == dc); }
        }
    }

    GIVEN("a 3x3 2D data container")
    {
        DataContainer<data_t> dc(VolumeDescriptor{{3, 3}});
        dc(0, 0) = 0;
        dc(0, 1) = 1;
        dc(0, 2) = 2;
        dc(1, 0) = 3;
        dc(1, 1) = 4;
        dc(1, 2) = -4;
        dc(2, 0) = -3;
        dc(2, 1) = -2;
        dc(2, 2) = -1;

        DataContainer<data_t> expectedFFTShiftDC(VolumeDescriptor{{3, 3}});
        expectedFFTShiftDC(0, 0) = -1;
        expectedFFTShiftDC(0, 1) = -3;
        expectedFFTShiftDC(0, 2) = -2;
        expectedFFTShiftDC(1, 0) = 2;
        expectedFFTShiftDC(1, 1) = 0;
        expectedFFTShiftDC(1, 2) = 1;
        expectedFFTShiftDC(2, 0) = -4;
        expectedFFTShiftDC(2, 1) = 3;
        expectedFFTShiftDC(2, 2) = 4;

        WHEN("running the FFT shift operation to the container")
        {
            DataContainer<data_t> fftShiftedDC = fftShift2D(dc);
            THEN("the data descriptors match")
            {
                REQUIRE_EQ(fftShiftedDC.getDataDescriptor(),
                           expectedFFTShiftDC.getDataDescriptor());
            }
            THEN("the data containers match") { REQUIRE_UNARY(fftShiftedDC == expectedFFTShiftDC); }
        }

        DataContainer<data_t> expectedIFFTShiftDC(VolumeDescriptor{{3, 3}});
        expectedIFFTShiftDC(0, 0) = 4;
        expectedIFFTShiftDC(0, 1) = -4;
        expectedIFFTShiftDC(0, 2) = 3;
        expectedIFFTShiftDC(1, 0) = -2;
        expectedIFFTShiftDC(1, 1) = -1;
        expectedIFFTShiftDC(1, 2) = -3;
        expectedIFFTShiftDC(2, 0) = 1;
        expectedIFFTShiftDC(2, 1) = 2;
        expectedIFFTShiftDC(2, 2) = 0;

        WHEN("running the IFFT shift operation to the container")
        {
            DataContainer<data_t> ifftShiftedDC = ifftShift2D(dc);
            THEN("the data descriptors match")
            {
                REQUIRE_EQ(ifftShiftedDC.getDataDescriptor(),
                           expectedIFFTShiftDC.getDataDescriptor());
            }
            THEN("the data containers match")
            {
                REQUIRE_UNARY(ifftShiftedDC == expectedIFFTShiftDC);
            }
        }
    }

    GIVEN("a 5x5 2D data container")
    {
        DataContainer<data_t> dc(VolumeDescriptor{{5, 5}});
        dc(0, 0) = 28;
        dc(0, 1) = 1;
        dc(0, 2) = 5;
        dc(0, 3) = -18;
        dc(0, 4) = 8;
        dc(1, 0) = 5;
        dc(1, 1) = 6;
        dc(1, 2) = 50;
        dc(1, 3) = -8;
        dc(1, 4) = 9;
        dc(2, 0) = 8;
        dc(2, 1) = 9;
        dc(2, 2) = 10;
        dc(2, 3) = 11;
        dc(2, 4) = 12;
        dc(3, 0) = -12;
        dc(3, 1) = -41;
        dc(3, 2) = -10;
        dc(3, 3) = -9;
        dc(3, 4) = -8;
        dc(4, 0) = -70;
        dc(4, 1) = -6;
        dc(4, 2) = 22;
        dc(4, 3) = -10;
        dc(4, 4) = -3;

        DataContainer<data_t> expectedFFTShiftDC(VolumeDescriptor{{5, 5}});
        expectedFFTShiftDC(0, 0) = -9;
        expectedFFTShiftDC(0, 1) = -8;
        expectedFFTShiftDC(0, 2) = -12;
        expectedFFTShiftDC(0, 3) = -41;
        expectedFFTShiftDC(0, 4) = -10;
        expectedFFTShiftDC(1, 0) = -10;
        expectedFFTShiftDC(1, 1) = -3;
        expectedFFTShiftDC(1, 2) = -70;
        expectedFFTShiftDC(1, 3) = -6;
        expectedFFTShiftDC(1, 4) = 22;
        expectedFFTShiftDC(2, 0) = -18;
        expectedFFTShiftDC(2, 1) = 8;
        expectedFFTShiftDC(2, 2) = 28;
        expectedFFTShiftDC(2, 3) = 1;
        expectedFFTShiftDC(2, 4) = 5;
        expectedFFTShiftDC(3, 0) = -8;
        expectedFFTShiftDC(3, 1) = 9;
        expectedFFTShiftDC(3, 2) = 5;
        expectedFFTShiftDC(3, 3) = 6;
        expectedFFTShiftDC(3, 4) = 50;
        expectedFFTShiftDC(4, 0) = 11;
        expectedFFTShiftDC(4, 1) = 12;
        expectedFFTShiftDC(4, 2) = 8;
        expectedFFTShiftDC(4, 3) = 9;
        expectedFFTShiftDC(4, 4) = 10;

        WHEN("running the FFT shift operation to the container")
        {
            DataContainer<data_t> fftShiftedDC = fftShift2D(dc);
            THEN("the data descriptors match")
            {
                REQUIRE_EQ(fftShiftedDC.getDataDescriptor(),
                           expectedFFTShiftDC.getDataDescriptor());
            }
            THEN("the data containers match") { REQUIRE_UNARY(fftShiftedDC == expectedFFTShiftDC); }
        }

        DataContainer<data_t> expectedIFFTShiftDC(VolumeDescriptor{{5, 5}});
        expectedIFFTShiftDC(0, 0) = 10;
        expectedIFFTShiftDC(0, 1) = 11;
        expectedIFFTShiftDC(0, 2) = 12;
        expectedIFFTShiftDC(0, 3) = 8;
        expectedIFFTShiftDC(0, 4) = 9;
        expectedIFFTShiftDC(1, 0) = -10;
        expectedIFFTShiftDC(1, 1) = -9;
        expectedIFFTShiftDC(1, 2) = -8;
        expectedIFFTShiftDC(1, 3) = -12;
        expectedIFFTShiftDC(1, 4) = -41;
        expectedIFFTShiftDC(2, 0) = 22;
        expectedIFFTShiftDC(2, 1) = -10;
        expectedIFFTShiftDC(2, 2) = -3;
        expectedIFFTShiftDC(2, 3) = -70;
        expectedIFFTShiftDC(2, 4) = -6;
        expectedIFFTShiftDC(3, 0) = 5;
        expectedIFFTShiftDC(3, 1) = -18;
        expectedIFFTShiftDC(3, 2) = 8;
        expectedIFFTShiftDC(3, 3) = 28;
        expectedIFFTShiftDC(3, 4) = 1;
        expectedIFFTShiftDC(4, 0) = 50;
        expectedIFFTShiftDC(4, 1) = -8;
        expectedIFFTShiftDC(4, 2) = 9;
        expectedIFFTShiftDC(4, 3) = 5;
        expectedIFFTShiftDC(4, 4) = 6;

        WHEN("running the IFFT shift operation to the container")
        {
            DataContainer<data_t> ifftShiftedDC = ifftShift2D(dc);
            THEN("the data descriptors match")
            {
                REQUIRE_EQ(ifftShiftedDC.getDataDescriptor(),
                           expectedIFFTShiftDC.getDataDescriptor());
            }
            THEN("the data containers match")
            {
                REQUIRE_UNARY(ifftShiftedDC == expectedIFFTShiftDC);
            }
        }
    }
}

// "instantiate" the test templates for CPU types
TEST_CASE_TEMPLATE_APPLY(datacontainer_construction, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datacontainer_reduction, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datacontainer_elemwise, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datacontainer_arithmetic, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datacontainer_maps, CPUTypeTuple);

#ifdef ELSA_CUDA_VECTOR
// "instantiate" the test templates for GPU types
TEST_CASE_TEMPLATE_APPLY(datacontainer_construction, GPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datacontainer_reduction, GPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datacontainer_elemwise, GPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datacontainer_arithmetic, GPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datacontainer_maps, GPUTypeTuple);
#endif

TEST_SUITE_END();
