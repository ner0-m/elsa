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

using namespace elsa;
using namespace Catch::literals; // to enable 0.0_a approximate floats

SCENARIO("Constructing DataContainers")
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 17, 47, 91;
        DataDescriptor desc(numCoeff);

        WHEN("constructing an empty DataContainer")
        {
            DataContainer dc(desc);

            THEN("it has the correct DataDescriptor") { REQUIRE(dc.getDataDescriptor() == desc); }

            THEN("it has a zero data vector of correct size")
            {
                REQUIRE(dc.getSize() == desc.getNumberOfCoefficients());

                for (index_t i = 0; i < desc.getNumberOfCoefficients(); ++i)
                    REQUIRE(dc[i] == 0.0);
            }
        }

        WHEN("constructing an initialized DataContainer")
        {
            RealVector_t data(desc.getNumberOfCoefficients());
            data.setRandom();

            DataContainer dc(desc, data);

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
        DataDescriptor desc(numCoeff);

        DataContainer otherDc(desc);
        Eigen::VectorXf randVec = Eigen::VectorXf::Random(otherDc.getSize());
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
            DataContainer dc(desc);
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

            DataContainer dc(desc);
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

SCENARIO("Element-wise access of DataContainers")
{
    GIVEN("a DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        DataDescriptor desc(numCoeff);
        DataContainer dc(desc);

        WHEN("accessing the elements")
        {
            IndexVector_t coord(2);
            coord << 17, 4;
            index_t index = desc.getIndexFromCoordinate(coord);

            THEN("it works as expected when using indices/coordinates")
            {
                dc[index] = 2.2f;
                REQUIRE(dc[index] == 2.2_a);
                REQUIRE(dc(coord) == 2.2_a);
                REQUIRE(dc(17, 4) == 2.2_a);

                dc(coord) = 3.3f;
                REQUIRE(dc[index] == 3.3_a);
                REQUIRE(dc(coord) == 3.3_a);
                REQUIRE(dc(17, 4) == 3.3_a);
            }
        }
    }
}

SCENARIO("Testing the reduction operations of DataContainer")
{
    GIVEN("a DataContainer")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 11, 73, 45;
        DataDescriptor desc(numCoeff);
        DataContainer dc(desc);

        WHEN("putting in some random data")
        {
            Eigen::VectorXf randVec = Eigen::VectorXf::Random(dc.getSize());
            for (index_t i = 0; i < dc.getSize(); ++i)
                dc[i] = randVec(i);

            THEN("the reductions work as expected")
            {
                REQUIRE(dc.sum() == Approx(randVec.sum()));
                REQUIRE(dc.l1Norm() == Approx(randVec.array().abs().sum()));
                REQUIRE(dc.lInfNorm() == Approx(randVec.array().abs().maxCoeff()));
                REQUIRE(dc.squaredL2Norm() == Approx(randVec.squaredNorm()));

                Eigen::VectorXf randVec2 = Eigen::VectorXf::Random(dc.getSize());
                DataContainer dc2(desc);
                for (index_t i = 0; i < dc2.getSize(); ++i)
                    dc2[i] = randVec2(i);

                REQUIRE(dc.dot(dc2) == Approx(randVec.dot(randVec2)));
            }
        }
    }
}

SCENARIO("Testing the element-wise operations of DataContainer")
{
    GIVEN("a DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        DataDescriptor desc(numCoeff);
        DataContainer dc(desc);

        WHEN("putting in some random data")
        {
            Eigen::VectorXf randVec = Eigen::VectorXf::Random(dc.getSize());
            for (index_t i = 0; i < dc.getSize(); ++i)
                dc[i] = randVec(i);

            THEN("the element-wise unary operations work as expected")
            {
                DataContainer dcSquare = square(dc);
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dcSquare[i] == Approx(randVec(i) * randVec(i)));

                DataContainer dcSqrt = sqrt(dc);
                for (index_t i = 0; i < dc.getSize(); ++i)
                    if (randVec(i) >= 0)
                        REQUIRE(dcSqrt[i] == Approx(std::sqrt(randVec(i))));

                DataContainer dcExp = exp(dc);
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dcExp[i] == Approx(std::exp(randVec(i))));

                DataContainer dcLog = log(dc);
                for (index_t i = 0; i < dc.getSize(); ++i)
                    if (randVec(i) > 0)
                        REQUIRE(dcLog[i] == Approx(std::log(randVec(i))));
            }

            THEN("the binary in-place addition of a scalar work as expected")
            {
                float scalar = 923.41f;
                dc += scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) + scalar);
            }

            THEN("the binary in-place subtraction of a scalar work as expected")
            {
                float scalar = 74.165f;
                dc -= scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) - scalar);
            }

            THEN("the binary in-place multiplication with a scalar work as expected")
            {
                float scalar = 12.69f;
                dc *= scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) * scalar);
            }

            THEN("the binary in-place division by a scalar work as expected")
            {
                float scalar = 82.61f;
                dc /= scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == randVec(i) / scalar);
            }

            THEN("the element-wise assignment of a scalar works as expected")
            {
                float scalar = 123.45f;
                dc = scalar;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    REQUIRE(dc[i] == scalar);
            }
        }

        WHEN("having two containers with random data")
        {
            Eigen::VectorXf randVec = Eigen::VectorXf::Random(dc.getSize());
            for (index_t i = 0; i < dc.getSize(); ++i)
                dc[i] = randVec(i);

            Eigen::VectorXf randVec2 = Eigen::VectorXf::Random(dc.getSize());
            DataContainer dc2(desc);
            for (index_t i = 0; i < dc2.getSize(); ++i)
                dc2[i] = randVec2[i];

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
                    REQUIRE(dc[i] == randVec(i) * randVec2(i));
            }

            THEN("the element-wise in-place division works as expected")
            {
                dc /= dc2;
                for (index_t i = 0; i < dc.getSize(); ++i)
                    if (dc2[i] != 0)
                        REQUIRE(dc[i] == randVec(i) / randVec2(i));
            }
        }
    }
}

SCENARIO("Testing the arithmetic operations with DataContainer arguments")
{
    GIVEN("some DataContainers")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 52, 7, 29;
        DataDescriptor desc(numCoeff);

        DataContainer dc(desc);
        DataContainer dc2(desc);

        Eigen::VectorXf randVec = Eigen::VectorXf::Random(dc.getSize());
        Eigen::VectorXf randVec2 = Eigen::VectorXf::Random(dc.getSize());

        for (index_t i = 0; i < dc.getSize(); ++i) {
            dc[i] = randVec(i);
            dc2[i] = randVec2(i);
        }

        THEN("the binary element-wise operations work as expected")
        {
            auto resultPlus = dc + dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE((resultPlus.eval())[i] == dc[i] + dc2[i]);

            auto resultMinus = dc - dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE((resultMinus.eval())[i] == dc[i] - dc2[i]);

            auto resultMult = dc * dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultMult.eval()[i] == dc[i] * dc2[i]);

            auto resultDiv = dc / dc2;
            for (index_t i = 0; i < dc.getSize(); ++i)
                if (dc2[i] != 0)
                    REQUIRE(resultDiv.eval()[i] == Approx(dc[i] / dc2[i]));
        }

        THEN("the operations with a scalar work as expected")
        {
            float scalar = 4.92f;

            auto resultScalarPlus = scalar + dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultScalarPlus.eval()[i] == scalar + dc[i]);

            auto resultPlusScalar = dc + scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultPlusScalar.eval()[i] == dc[i] + scalar);

            auto resultScalarMinus = scalar - dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultScalarMinus.eval()[i] == scalar - dc[i]);

            auto resultMinusScalar = dc - scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultMinusScalar.eval()[i] == dc[i] - scalar);

            auto resultScalarMult = scalar * dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultScalarMult.eval()[i] == scalar * dc[i]);

            auto resultMultScalar = dc * scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultMultScalar.eval()[i] == dc[i] * scalar);

            auto resultScalarDiv = scalar / dc;
            for (index_t i = 0; i < dc.getSize(); ++i)
                if (dc[i] != 0)
                    REQUIRE(resultScalarDiv.eval()[i] == scalar / dc[i]);

            auto resultDivScalar = dc / scalar;
            for (index_t i = 0; i < dc.getSize(); ++i)
                REQUIRE(resultDivScalar.eval()[i] == dc[i] / scalar);
        }
    }
}

SCENARIO("Testing creation of Maps through DataContainer")
{
    GIVEN("a non-blocked container")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 52, 7, 29;
        DataDescriptor desc(numCoeff);

        DataContainer dc(desc);
        const DataContainer constDc(desc);

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
            DataDescriptor linearDesc(numCoeff);
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
        DataDescriptor desc(numCoeff);
        index_t numBlocks = 7;
        IdenticalBlocksDescriptor blockDesc(numBlocks, desc);

        DataContainer dc(blockDesc);
        const DataContainer constDc(blockDesc);

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
            DataDescriptor linearDesc(numCoeff);
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

SCENARIO("Testing iterators for DataContainer")
{
    GIVEN("A 1D container")
    {
        constexpr index_t size = 20;
        IndexVector_t numCoeff(1);
        numCoeff << size;
        DataDescriptor desc(numCoeff);

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
        DataDescriptor desc(numCoeff);

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
        DataDescriptor desc(numCoeff);

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
