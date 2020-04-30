/**
 * \file test_DataContainer.cpp
 *
 * \brief Tests for DataContainer class
 *
 * \author Matthias Wieczorek - initial code
 * \author David Frank - rewrite to use Catch and BDD
 * \author Tobias Lasser - rewrite and added code coverage
 */

#include <catch2/catch.hpp>
#include "DataContainer.h"
#include "IdenticalBlocksDescriptor.h"
#include "testHelpers.h"
#include "VolumeDescriptor.h"

#include <type_traits>

using namespace elsa;
using namespace Catch::literals; // to enable 0.0_a approximate floats

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

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Constructing DataContainers", "",
                           (TestHelperCPU, TestHelperGPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Constructing DataContainers", "", (TestHelperCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::data_t;

    INFO("Testing type: " << TypeName_v<const data_t>);

    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 17, 47, 91;
        VolumeDescriptor desc(numCoeff);

        WHEN("constructing an empty DataContainer")
        {
            DataContainer<data_t> dc(desc, TestType::handler_t);

            THEN("it has the correct DataDescriptor") { REQUIRE(dc.getDataDescriptor() == desc); }

            THEN("it has a data vector of correct size")
            {
                REQUIRE(dc.getSize() == desc.getNumberOfCoefficients());
            }
        }

        WHEN("constructing an initialized DataContainer")
        {
            auto data = generateRandomMatrix<data_t>(desc.getNumberOfCoefficients());

            DataContainer<data_t> dc(desc, data, TestType::handler_t);

            THEN("it has the correct DataDescriptor") { REQUIRE(dc.getDataDescriptor() == desc); }

            THEN("it has correctly initialized data")
            {
                REQUIRE(dc.getSize() == desc.getNumberOfCoefficients());

                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == data[i]);
            }
        }
    }

    GIVEN("another DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 32, 57;
        VolumeDescriptor desc(numCoeff);

        DataContainer<data_t> otherDc(desc, TestType::handler_t);

        auto randVec = generateRandomMatrix<data_t>(otherDc.getSize());
        for (index_t i = 0; i < otherDc.getSize(); ++i)
            otherDc[i] = randVec(i);

        WHEN("copy constructing")
        {
            DataContainer dc(otherDc);

            THEN("it copied correctly")
            {
                REQUIRE(dc.getDataDescriptor() == otherDc.getDataDescriptor());
                REQUIRE(&dc.getDataDescriptor() != &otherDc.getDataDescriptor());

                REQUIRE(dc == otherDc);
            }
        }

        WHEN("copy assigning")
        {
            DataContainer<data_t> dc(desc, TestType::handler_t);
            dc = otherDc;

            THEN("it copied correctly")
            {
                REQUIRE(dc.getDataDescriptor() == otherDc.getDataDescriptor());
                REQUIRE(&dc.getDataDescriptor() != &otherDc.getDataDescriptor());

                REQUIRE(dc == otherDc);
            }
        }

        WHEN("move constructing")
        {
            DataContainer oldOtherDc(otherDc);

            DataContainer dc(std::move(otherDc));

            THEN("it moved correctly")
            {
                REQUIRE(dc.getDataDescriptor() == oldOtherDc.getDataDescriptor());

                REQUIRE(dc == oldOtherDc);
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
                REQUIRE(dc.getDataDescriptor() == oldOtherDc.getDataDescriptor());

                REQUIRE(dc == oldOtherDc);
            }

            THEN("the moved from object is still valid (but empty)") { otherDc = dc; }
        }
    }
}

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Element-wise access of DataContainers", "",
                           (TestHelperCPU, TestHelperGPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Element-wise access of DataContainers", "", (TestHelperCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::data_t;

    INFO("Testing type: " << TypeName_v<const data_t>);

    GIVEN("a DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        VolumeDescriptor desc(numCoeff);
        DataContainer<data_t> dc(desc, TestType::handler_t);

        WHEN("accessing the elements")
        {
            IndexVector_t coord(2);
            coord << 17, 4;
            index_t index = desc.getIndexFromCoordinate(coord);

            THEN("it works as expected when using indices/coordinates")
            {
                dc[index] = data_t(2.2f);
                REQUIRE(dc[index] == data_t(2.2f));
                REQUIRE(dc(coord) == data_t(2.2f));
                REQUIRE(dc(17, 4) == data_t(2.2f));

                dc(coord) = data_t(3.3f);
                REQUIRE(dc[index] == data_t(3.3f));
                REQUIRE(dc(coord) == data_t(3.3f));
                REQUIRE(dc(17, 4) == data_t(3.3f));
            }
        }
    }
}

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing the reduction operations of DataContainer", "",
                           (TestHelperCPU, TestHelperGPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing the reduction operations of DataContainer", "",
                           (TestHelperCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::data_t;

    INFO("Testing type: " << TypeName_v<const data_t>);

    GIVEN("a DataContainer")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 11, 73, 45;
        VolumeDescriptor desc(numCoeff);

        WHEN("putting in some random data")
        {
            auto [dc, randVec] = generateRandomContainer<data_t>(desc, TestType::handler_t);

            THEN("the reductions work as expected")
            {
                REQUIRE(checkSameNumbers(dc.sum(), randVec.sum()));
                REQUIRE(checkSameNumbers(dc.l1Norm(), randVec.array().abs().sum()));
                REQUIRE(checkSameNumbers(dc.lInfNorm(), randVec.array().abs().maxCoeff()));
                REQUIRE(checkSameNumbers(dc.squaredL2Norm(), randVec.squaredNorm()));

                auto [dc2, randVec2] = generateRandomContainer<data_t>(desc, TestType::handler_t);

                REQUIRE(checkSameNumbers(dc.dot(dc2), randVec.dot(randVec2)));
            }
        }
    }
}

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing the element-wise operations of DataContainer", "",
                           (TestHelperCPU, TestHelperGPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing the element-wise operations of DataContainer", "",
                           (TestHelperCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::data_t;

    INFO("Testing type: " << TypeName_v<const data_t>);

    GIVEN("a DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        VolumeDescriptor desc(numCoeff);

        WHEN("putting in some random data")
        {
            auto [dc, randVec] = generateRandomContainer<data_t>(desc, TestType::handler_t);

            THEN("the element-wise unary operations work as expected")
            {
                DataContainer dcSquare = square(dc);
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(checkSameNumbers(dcSquare[i], randVec.array().square()[i]));
                DataContainer dcSqrt = sqrt(dcSquare);
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(checkSameNumbers(dcSqrt[i], randVec.array().square().sqrt()[i]));

                // do exponent check only for floating point types as for integer will likely lead
                // to overflow due to random init over full value range
                if constexpr (!std::is_integral_v<data_t>) {
                    DataContainer dcExp = exp(dc);
                    for (index_t i = 0; i < dc.getSize(); ++i)
                        REQUIRE(checkSameNumbers(dcExp[i], randVec.array().exp()[i]));
                }

                DataContainer dcLog = log(dcSquare);
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(checkSameNumbers(dcLog[i], randVec.array().square().log()[i]));
            }

            auto scalar = static_cast<data_t>(923.41f);

            THEN("the binary in-place addition of a scalar work as expected")
            {
                dc += scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) + scalar);
            }

            THEN("the binary in-place subtraction of a scalar work as expected")
            {
                dc -= scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) - scalar);
            }

            THEN("the binary in-place multiplication with a scalar work as expected")
            {
                dc *= scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) * scalar);
            }

            THEN("the binary in-place division by a scalar work as expected")
            {
                dc /= scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(checkSameNumbers(dc[i], randVec(i) / scalar));
            }

            THEN("the element-wise assignment of a scalar works as expected")
            {
                dc = scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == scalar);
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
                    REQUIRE(dc[i] == randVec(i) + randVec2(i));
            }

            THEN("the element-wise in-place subtraction works as expected")
            {
                dc -= dc2;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) - randVec2(i));
            }

            THEN("the element-wise in-place multiplication works as expected")
            {
                dc *= dc2;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(checkSameNumbers(dc[i], randVec(i) * randVec2(i)));
            }

            THEN("the element-wise in-place division works as expected")
            {
                dc /= dc2;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    if (dc2[i] != data_t(0))
                        REQUIRE(checkSameNumbers(dc[i], randVec(i) / randVec2(i)));
            }
        }
    }
}

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE(
    "Scenario: Testing the arithmetic operations with DataContainer arguments", "",
    (TestHelperCPU, TestHelperGPU),
    (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE(
    "Scenario: Testing the arithmetic operations with DataContainer arguments", "", (TestHelperCPU),
    (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::data_t;

    INFO("Testing type: " << TypeName_v<const data_t>);

    GIVEN("some DataContainers")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 52, 7, 29;
        VolumeDescriptor desc(numCoeff);

        auto [dc, randVec] = generateRandomContainer<data_t>(desc, TestType::handler_t);
        auto [dc2, randVec2] = generateRandomContainer<data_t>(desc, TestType::handler_t);

        THEN("the binary element-wise operations work as expected")
        {
            DataContainer resultPlus = dc + dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultPlus[i] == dc[i] + dc2[i]);

            DataContainer resultMinus = dc - dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultMinus[i] == dc[i] - dc2[i]);

            DataContainer resultMult = dc * dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(checkSameNumbers(resultMult[i], dc[i] * dc2[i]));

            DataContainer resultDiv = dc / dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                if (dc2[i] != data_t(0))
                    REQUIRE(checkSameNumbers(resultDiv[i], dc[i] / dc2[i]));
        }

        THEN("the operations with a scalar work as expected")
        {
            data_t scalar = static_cast<data_t>(4.92f);

            DataContainer resultScalarPlus = scalar + dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultScalarPlus[i] == scalar + dc[i]);

            DataContainer resultPlusScalar = dc + scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultPlusScalar[i] == dc[i] + scalar);

            DataContainer resultScalarMinus = scalar - dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultScalarMinus[i] == scalar - dc[i]);

            DataContainer resultMinusScalar = dc - scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultMinusScalar[i] == dc[i] - scalar);

            DataContainer resultScalarMult = scalar * dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultScalarMult[i] == scalar * dc[i]);

            DataContainer resultMultScalar = dc * scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultMultScalar[i] == dc[i] * scalar);

            DataContainer resultScalarDiv = scalar / dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                if (dc[i] != data_t(0))
                    REQUIRE(checkSameNumbers(resultScalarDiv[i], scalar / dc[i]));

            DataContainer resultDivScalar = dc / scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(checkSameNumbers(resultDivScalar[i], dc[i] / scalar));
        }
    }
}

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing creation of Maps through DataContainer", "",
                           (TestHelperCPU, TestHelperGPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing creation of Maps through DataContainer", "",
                           (TestHelperCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::data_t;

    INFO("Testing type: " << TypeName_v<const data_t>);

    GIVEN("a non-blocked container")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 52, 7, 29;
        VolumeDescriptor desc(numCoeff);

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
            IndexVector_t numCoeff(1);
            numCoeff << desc.getNumberOfCoefficients();
            VolumeDescriptor linearDesc(numCoeff);
            auto linearDc = dc.viewAs(linearDesc);
            auto linearConstDc = constDc.viewAs(linearDesc);

            THEN("view has the correct descriptor and data")
            {
                REQUIRE(linearDesc == linearDc.getDataDescriptor());
                REQUIRE(&linearDc[0] == &dc[0]);

                REQUIRE(linearDesc == linearConstDc.getDataDescriptor());
                REQUIRE(&linearConstDc[0] == &constDc[0]);

                AND_THEN("view is not a shallow copy")
                {
                    const auto dcCopy = dc;
                    const auto constDcCopy = constDc;

                    linearDc[0] = 1;
                    REQUIRE(&linearDc[0] == &dc[0]);
                    REQUIRE(&linearDc[0] != &dcCopy[0]);

                    linearConstDc[0] = 1;
                    REQUIRE(&linearConstDc[0] == &constDc[0]);
                    REQUIRE(&linearConstDc[0] != &constDcCopy[0]);
                }
            }
        }
    }

    GIVEN("a blocked container")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 52, 29;
        VolumeDescriptor desc(numCoeff);
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

                    REQUIRE(dcBlock.getDataDescriptor() == blockDesc.getDescriptorOfBlock(i));
                    REQUIRE(&dcBlock[0] == &dc[0] + blockDesc.getOffsetOfBlock(i));

                    REQUIRE(constDcBlock.getDataDescriptor() == blockDesc.getDescriptorOfBlock(i));
                    REQUIRE(&constDcBlock[0] == &constDc[0] + blockDesc.getOffsetOfBlock(i));
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
                REQUIRE(linearDesc == linearDc.getDataDescriptor());
                REQUIRE(&linearDc[0] == &dc[0]);

                REQUIRE(linearDesc == linearConstDc.getDataDescriptor());
                REQUIRE(&linearConstDc[0] == &constDc[0]);

                AND_THEN("view is not a shallow copy")
                {
                    const auto dcCopy = dc;
                    const auto constDcCopy = constDc;

                    linearDc[0] = 1;
                    REQUIRE(&linearDc[0] == &dc[0]);
                    REQUIRE(&linearDc[0] != &dcCopy[0]);

                    linearConstDc[0] = 1;
                    REQUIRE(&linearConstDc[0] == &constDc[0]);
                    REQUIRE(&linearConstDc[0] != &constDcCopy[0]);
                }
            }
        }
    }
}

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_TEST_CASE("Scenario: Testing loading data to GPU and vice versa.", "", float, double,
                   std::complex<float>, std::complex<double>, index_t)
{
    GIVEN("A CPU DataContainer with random data")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 52, 7, 29;
        VolumeDescriptor desc(numCoeff);

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

            REQUIRE(dcGPU2.getDataHandlerType() == DataHandlerType::GPU);

            THEN("all elements have to be the same")
            {
                for (index_t i = 0; i < dcCPU.getSize(); ++i) {
                    REQUIRE(dcGPU2[i] == dcGPU[i]);
                }
            }
        }

        WHEN("Loading to CPU from GPU")
        {
            DataContainer dcCPU2 = dcGPU.loadToCPU();

            REQUIRE(dcCPU2.getDataHandlerType() == DataHandlerType::CPU);

            THEN("all elements have to be the same")
            {
                for (index_t i = 0; i < dcCPU.getSize(); ++i) {
                    REQUIRE(dcCPU2[i] == dcGPU[i]);
                }
            }
        }

        WHEN("copy-assigning a GPU to a CPU container")
        {
            dcCPU = dcGPU;

            THEN("it should be a GPU container")
            {
                REQUIRE(dcCPU.getDataHandlerType() == DataHandlerType::GPU);
            }

            AND_THEN("they should be equal") { REQUIRE(dcCPU == dcGPU); }
        }

        WHEN("copy-assigning a CPU to a GPU container")
        {
            dcGPU = dcCPU;

            THEN("it should be a GPU container")
            {
                REQUIRE(dcGPU.getDataHandlerType() == DataHandlerType::CPU);
            }

            AND_THEN("they should be equal") { REQUIRE(dcCPU == dcGPU); }
        }
    }
}
#endif

SCENARIO("Testing iterators for DataContainer")
{
    GIVEN("A 1D container")
    {
        constexpr index_t size = 20;
        IndexVector_t numCoeff(1);
        numCoeff << size;
        VolumeDescriptor desc(numCoeff);

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
                REQUIRE(*v == randVec1[i++]);
            }
            REQUIRE(i == size);
        }

        THEN("We can iterate backward")
        {
            int i = size;
            for (auto v = dc1.crbegin(); v != dc1.crend(); v++) {
                REQUIRE(*v == randVec1[--i]);
            }
            REQUIRE(i == 0);
        }

        THEN("We can iterate and mutate")
        {
            int i = 0;
            for (auto& v : dc1) {
                v = v * 2;
                REQUIRE(v == 2 * randVec1[i++]);
            }
            REQUIRE(i == size);

            i = 0;
            for (auto v : dc1) {
                REQUIRE(v == 2 * randVec1[i++]);
            }
            REQUIRE(i == size);
        }

        THEN("We can use STL algorithms")
        {
            REQUIRE(*std::min_element(dc1.cbegin(), dc1.cend()) == randVec1.minCoeff());
            REQUIRE(*std::max_element(dc1.cbegin(), dc1.cend()) == randVec1.maxCoeff());
        }
    }
    GIVEN("A 2D container")
    {
        constexpr index_t size = 20;
        IndexVector_t numCoeff(2);
        numCoeff << size, size;
        VolumeDescriptor desc(numCoeff);

        DataContainer dc1(desc);

        Eigen::VectorXf randVec1 = Eigen::VectorXf::Random(size * size);

        for (index_t i = 0; i < dc1.getSize(); ++i) {
            dc1[i] = randVec1[i];
        }

        THEN("We can iterate forward")
        {
            int i = 0;
            for (auto v : dc1) {
                REQUIRE(v == randVec1[i++]);
            }
            REQUIRE(i == size * size);
        }

        THEN("We can iterate backward")
        {
            int i = size * size;
            for (auto v = dc1.crbegin(); v != dc1.crend(); v++) {
                REQUIRE(*v == randVec1[--i]);
            }
            REQUIRE(i == 0);
        }

        THEN("We can iterate and mutate")
        {
            int i = 0;
            for (auto& v : dc1) {
                v = v * 2;
                REQUIRE(v == 2 * randVec1[i++]);
            }
            REQUIRE(i == size * size);

            i = 0;
            for (auto v : dc1) {
                REQUIRE(v == 2 * randVec1[i++]);
            }
            REQUIRE(i == size * size);
        }

        THEN("We can use STL algorithms")
        {
            REQUIRE(*std::min_element(dc1.cbegin(), dc1.cend()) == randVec1.minCoeff());
            REQUIRE(*std::max_element(dc1.cbegin(), dc1.cend()) == randVec1.maxCoeff());
        }
    }
    GIVEN("A 3D container")
    {
        constexpr index_t size = 20;
        IndexVector_t numCoeff(3);
        numCoeff << size, size, size;
        VolumeDescriptor desc(numCoeff);

        DataContainer dc1(desc);

        Eigen::VectorXf randVec1 = Eigen::VectorXf::Random(size * size * size);

        for (index_t i = 0; i < dc1.getSize(); ++i) {
            dc1[i] = randVec1[i];
        }

        THEN("We can iterate forward")
        {
            int i = 0;
            for (auto v : dc1) {
                REQUIRE(v == randVec1[i++]);
            }
            REQUIRE(i == size * size * size);
        }

        THEN("We can iterate backward")
        {
            int i = size * size * size;
            for (auto v = dc1.crbegin(); v != dc1.crend(); v++) {
                REQUIRE(*v == randVec1[--i]);
            }
            REQUIRE(i == 0);
        }

        THEN("We can iterate and mutate")
        {
            int i = 0;
            for (auto& v : dc1) {
                v = v * 2;
                REQUIRE(v == 2 * randVec1[i++]);
            }
            REQUIRE(i == size * size * size);

            i = 0;
            for (auto v : dc1) {
                REQUIRE(v == 2 * randVec1[i++]);
            }
            REQUIRE(i == size * size * size);
        }

        THEN("We can use STL algorithms")
        {
            REQUIRE(*std::min_element(dc1.cbegin(), dc1.cend()) == randVec1.minCoeff());
            REQUIRE(*std::max_element(dc1.cbegin(), dc1.cend()) == randVec1.maxCoeff());
        }
    }
}
