/**
 * \file test_IdenticalBlocksDescriptor.cpp
 *
 * \brief Tests for IdenticalBlocksDescriptor class
 *
 * \author David Frank - initial version
 * \author Nikola Dinev - various enhancements
 * \author Tobias Lasser - rewrite and added code coverage
 */

#include <catch2/catch.hpp>

#include "IdenticalBlocksDescriptor.h"

using namespace elsa;

SCENARIO("Constructing IdenticalBlocksDescriptors")
{
    GIVEN("a 1D descriptor")
    {
        index_t size = 11;
        DataDescriptor dd(IndexVector_t::Constant(1, size));

        WHEN("creating 0 blocks") { REQUIRE_THROWS(IdenticalBlocksDescriptor(0, dd)); }

        WHEN("creating 5 blocks")
        {
            index_t blocks = 5;
            IdenticalBlocksDescriptor bd(blocks, dd);

            THEN("there are 5 blocks of the correct size")
            {
                REQUIRE(bd.getNumberOfBlocks() == blocks);

                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE(bd.getDescriptorOfBlock(i) == dd);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the new IdenticalBlocksDescriptor has the correct sizes")
            {
                REQUIRE(bd.getNumberOfDimensions() == 2);

                IndexVector_t correctSize(2);
                correctSize << dd.getNumberOfCoefficients(), blocks;
                REQUIRE(bd.getNumberOfCoefficientsPerDimension() == correctSize);
                REQUIRE(bd.getNumberOfCoefficients() == correctSize.prod());

                RealVector_t correctSpacing(2);
                correctSpacing(0) = dd.getSpacingPerDimension()(0);
                correctSpacing(1) = 1.0;
                REQUIRE(bd.getSpacingPerDimension() == correctSpacing);
            }

            THEN("the new IdenticalBlocksDescriptor does index calculations correctly")
            {
                IndexVector_t coordinate(2);
                coordinate << dd.getNumberOfCoefficients() - 1, blocks - 1;
                index_t index = coordinate(0) + coordinate(1) * dd.getNumberOfCoefficients();

                REQUIRE(bd.getIndexFromCoordinate(coordinate) == index);
                REQUIRE(bd.getCoordinateFromIndex(index) == coordinate);
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE(bd.getOffsetOfBlock(i) == i * dd.getNumberOfCoefficients());

                REQUIRE_THROWS(bd.getOffsetOfBlock(blocks));
            }

            THEN("the block descriptor is different from a monolithic descriptor with the same "
                 "dimensions")
            {
                IndexVector_t monoCoeffs(2);
                monoCoeffs << size, blocks;
                DataDescriptor mono(monoCoeffs);
                REQUIRE(mono != bd);
                REQUIRE(bd != mono);
            }
        }
    }

    GIVEN("a 2D descriptor")
    {
        IndexVector_t sizeVector(2);
        sizeVector << 11, 12;
        DataDescriptor dd(sizeVector);

        WHEN("creating 10 blocks")
        {
            index_t blocks = 10;
            IdenticalBlocksDescriptor bd(blocks, dd);

            THEN("there are 10 blocks of the correct size")
            {
                REQUIRE(bd.getNumberOfBlocks() == blocks);

                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE(bd.getDescriptorOfBlock(i) == dd);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the new IdenticalBlocksDescriptor has the correct sizes")
            {
                REQUIRE(bd.getNumberOfDimensions() == 3);

                IndexVector_t correctSize(3);
                correctSize << sizeVector(0), sizeVector(1), blocks;
                REQUIRE(bd.getNumberOfCoefficientsPerDimension() == correctSize);
                REQUIRE(bd.getNumberOfCoefficients() == correctSize.prod());

                RealVector_t correctSpacing(3);
                correctSpacing(0) = dd.getSpacingPerDimension()(0);
                correctSpacing(1) = dd.getSpacingPerDimension()(1);
                correctSpacing(2) = 1.0;
                REQUIRE(bd.getSpacingPerDimension() == correctSpacing);
            }

            THEN("the new IdenticalBlocksDescriptor does index calculations correctly")
            {
                IndexVector_t coordinate(3);
                coordinate << sizeVector(0) - 3, sizeVector(1) - 7, blocks - 2;
                index_t index = coordinate(0) + coordinate(1) * sizeVector(0)
                                + coordinate(2) * sizeVector(0) * sizeVector(1);

                REQUIRE(bd.getIndexFromCoordinate(coordinate) == index);
                REQUIRE(bd.getCoordinateFromIndex(index) == coordinate);
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE(bd.getOffsetOfBlock(i) == i * dd.getNumberOfCoefficients());

                REQUIRE_THROWS(bd.getOffsetOfBlock(blocks));
            }

            THEN("the block descriptor is different from a monolithic descriptor with the same "
                 "dimensions")
            {
                IndexVector_t monoCoeffs(3);
                monoCoeffs << sizeVector, blocks;
                DataDescriptor mono(monoCoeffs);
                REQUIRE(mono != bd);
                REQUIRE(bd != mono);
            }

            THEN("the block descriptor is different from an IdenticalBlocksDescriptor with the "
                 "same dimensions, but with a different descriptor of each block")
            {
                IndexVector_t coeffsBlock = dd.getNumberOfCoefficientsPerDimension();
                DataDescriptor dd2(coeffsBlock.head(coeffsBlock.size() - 1));
                IdenticalBlocksDescriptor dd3(coeffsBlock[coeffsBlock.size() - 1], dd2);
                IdenticalBlocksDescriptor bd2(blocks, dd3);
                REQUIRE(bd2 != bd);
                REQUIRE(bd != bd2);
            }
        }
    }

    GIVEN("a 3D descriptor")
    {
        IndexVector_t sizeVector(3);
        sizeVector << 101, 42, 57;
        DataDescriptor dd(sizeVector);

        WHEN("creating 25 blocks")
        {
            index_t blocks = 25;
            IdenticalBlocksDescriptor bd(blocks, dd);

            THEN("there are 25 blocks of the correct size")
            {
                REQUIRE(bd.getNumberOfBlocks() == blocks);

                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE(bd.getDescriptorOfBlock(i) == dd);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the new IdenticalBlocksDescriptor has the correct sizes")
            {
                REQUIRE(bd.getNumberOfDimensions() == 4);

                IndexVector_t correctSize(4);
                correctSize << sizeVector(0), sizeVector(1), sizeVector(2), blocks;
                REQUIRE(bd.getNumberOfCoefficientsPerDimension() == correctSize);
                REQUIRE(bd.getNumberOfCoefficients() == correctSize.prod());

                RealVector_t correctSpacing(4);
                for (index_t i = 0; i < 3; ++i)
                    correctSpacing(i) = dd.getSpacingPerDimension()(i);
                correctSpacing(3) = 1.0;
                REQUIRE(bd.getSpacingPerDimension() == correctSpacing);
            }

            THEN("the new IdenticalBlocksDescriptor does index calculations correctly")
            {
                IndexVector_t coordinate(4);
                coordinate << sizeVector(0) - 33, sizeVector(1) - 11, sizeVector(2) - 17,
                    blocks - 19;
                index_t index = coordinate(0) + coordinate(1) * sizeVector(0)
                                + coordinate(2) * sizeVector(0) * sizeVector(1)
                                + coordinate(3) * sizeVector(0) * sizeVector(1) * sizeVector(2);

                REQUIRE(bd.getIndexFromCoordinate(coordinate) == index);
                REQUIRE(bd.getCoordinateFromIndex(index) == coordinate);
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE(bd.getOffsetOfBlock(i) == i * dd.getNumberOfCoefficients());

                REQUIRE_THROWS(bd.getOffsetOfBlock(blocks));
            }

            THEN("the block descriptor is different from a monolithic descriptor with the same "
                 "dimensions")
            {
                IndexVector_t monoCoeffs(4);
                monoCoeffs << sizeVector, blocks;
                DataDescriptor mono(monoCoeffs);
                REQUIRE(mono != bd);
                REQUIRE(bd != mono);
            }

            THEN("the block descriptor is different from an IdenticalBlocksDescriptor with the "
                 "same dimensions, but with a different descriptor of each block")
            {
                IndexVector_t coeffsBlock = dd.getNumberOfCoefficientsPerDimension();
                DataDescriptor dd2(coeffsBlock.head(coeffsBlock.size() - 1));
                IdenticalBlocksDescriptor dd3(coeffsBlock[coeffsBlock.size() - 1], dd2);
                IdenticalBlocksDescriptor bd2(blocks, dd3);
                REQUIRE(bd2 != bd);
                REQUIRE(bd != bd2);
            }
        }
    }
}

SCENARIO("Cloning IdenticalBlocksDescriptors")
{
    GIVEN("a 1D IdenticalBlocksDescriptor")
    {
        IndexVector_t sizeVector(1);
        sizeVector << 13;
        DataDescriptor dd(sizeVector);
        index_t blocks = 21;

        WHEN("cloning the descriptor")
        {
            IdenticalBlocksDescriptor bd(blocks, dd);
            auto bdClone = bd.clone();

            THEN("it's a real clone")
            {
                REQUIRE(bdClone.get() != &bd);
                REQUIRE(dynamic_cast<IdenticalBlocksDescriptor*>(bdClone.get()));
                REQUIRE(*bdClone == bd);
            }
        }
    }

    GIVEN("a 2D IdenticalBlocksDescriptor")
    {
        IndexVector_t sizeVector(2);
        sizeVector << 43, 112;
        DataDescriptor dd(sizeVector);
        index_t blocks = 77;

        WHEN("cloning the descriptor")
        {
            IdenticalBlocksDescriptor bd(blocks, dd);
            auto bdClone = bd.clone();

            THEN("it's a real clone")
            {
                REQUIRE(bdClone.get() != &bd);
                REQUIRE(dynamic_cast<IdenticalBlocksDescriptor*>(bdClone.get()));
                REQUIRE(*bdClone == bd);
            }
        }
    }

    GIVEN("a 3D IdenticalBlocksDescriptor")
    {
        IndexVector_t sizeVector(3);
        sizeVector << 47, 11, 53;
        DataDescriptor dd(sizeVector);
        index_t blocks = 13;

        WHEN("cloning the descriptor")
        {
            IdenticalBlocksDescriptor bd(blocks, dd);
            auto bdClone = bd.clone();

            THEN("it's a real clone")
            {
                REQUIRE(bdClone.get() != &bd);
                REQUIRE(dynamic_cast<IdenticalBlocksDescriptor*>(bdClone.get()));
                REQUIRE(*bdClone == bd);
            }
        }
    }
}