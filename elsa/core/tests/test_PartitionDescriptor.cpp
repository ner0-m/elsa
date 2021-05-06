/**
 * @file test_PartitionDescriptor.cpp
 *
 * @brief Tests for PartitionDescriptor class
 *
 * @author Nikola Dinev
 */

#include "doctest/doctest.h"
#include "PartitionDescriptor.h"
#include "VolumeDescriptor.h"
#include <stdexcept>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE("PartitionDescriptor: Testing construction")
{
    GIVEN("a 1D descriptor")
    {
        VolumeDescriptor dd(IndexVector_t::Constant(1, 10));

        WHEN("partitioning it into 1 blocks")
        {
            REQUIRE_THROWS(PartitionDescriptor(dd, 1));
            REQUIRE_THROWS(PartitionDescriptor(dd, IndexVector_t::Constant(1, 10)));
        }

        WHEN("partitioning it into more blocks than the size of the last dimension")
        {
            REQUIRE_THROWS(PartitionDescriptor(dd, 11));
            REQUIRE_THROWS(PartitionDescriptor(dd, IndexVector_t::Ones(11)));
            REQUIRE_THROWS(PartitionDescriptor(dd, IndexVector_t::Zero(11)));
        }

        WHEN("partitioning it into 5 blocks with equal sizes")
        {
            index_t blocks = 5;
            PartitionDescriptor bd(dd, blocks);

            THEN("the partitioned descriptor has the same number of coefficients and spacing per "
                 "dimension as the original")
            {
                REQUIRE_EQ(dd.getNumberOfCoefficientsPerDimension(),
                           bd.getNumberOfCoefficientsPerDimension());
                REQUIRE_EQ(dd.getSpacingPerDimension(), bd.getSpacingPerDimension());
            }

            THEN("there are 5 blocks of the correct size")
            {
                REQUIRE_EQ(bd.getNumberOfBlocks(), blocks);

                VolumeDescriptor bd0(IndexVector_t::Constant(1, 2), dd.getSpacingPerDimension());
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i), bd0);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getOffsetOfBlock(i), i * dd.getNumberOfCoefficients() / blocks);

                REQUIRE_THROWS(bd.getOffsetOfBlock(blocks));
            }

            THEN("original and partitioned descriptor are not equal")
            {
                REQUIRE_NE(bd, dd);
                REQUIRE_NE(dd, bd);
            }
        }

        WHEN("partitioning it into 5 blocks with chosen sizes")
        {
            index_t blocks = 5;
            IndexVector_t split(blocks);
            split << 1, 2, 3, 1, 3;
            PartitionDescriptor bd(dd, split);

            THEN("the partitioned descriptor has the same number of coefficients and spacing per "
                 "dimension as the original")
            {
                REQUIRE_EQ(dd.getNumberOfCoefficientsPerDimension(),
                           bd.getNumberOfCoefficientsPerDimension());
                REQUIRE_EQ(dd.getSpacingPerDimension(), bd.getSpacingPerDimension());
            }

            THEN("there are 5 blocks of the correct size")
            {
                REQUIRE_EQ(bd.getNumberOfBlocks(), blocks);

                for (index_t i = 0; i < blocks; ++i) {
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i).getNumberOfDimensions(), 1);
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i).getNumberOfCoefficients(), split[i]);
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i).getSpacingPerDimension(),
                               bd.getSpacingPerDimension());
                }

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getOffsetOfBlock(i), split.head(i).sum());

                REQUIRE_THROWS(bd.getOffsetOfBlock(blocks));
            }

            THEN("original and partitioned descriptor are not equal")
            {
                REQUIRE_NE(bd, dd);
                REQUIRE_NE(dd, bd);
            }
        }
    }

    GIVEN("a 2D descriptor with blocks")
    {
        IndexVector_t sizeVector(2);
        sizeVector << 11, 102;
        VolumeDescriptor dd(sizeVector);

        IndexVector_t coeffs(2);
        coeffs << 11, 10;
        VolumeDescriptor bd0(coeffs, dd.getSpacingPerDimension());

        coeffs[1] = 11;
        VolumeDescriptor bdn(coeffs, dd.getSpacingPerDimension());

        WHEN("partitioning it into 10 blocks")
        {
            index_t blocks = 10;
            PartitionDescriptor bd(dd, blocks);

            THEN("the partitioned descriptor has the same number of coefficients and spacing per "
                 "dimension as the original")
            {
                REQUIRE_EQ(dd.getNumberOfCoefficientsPerDimension(),
                           bd.getNumberOfCoefficientsPerDimension());
                REQUIRE_EQ(dd.getSpacingPerDimension(), bd.getSpacingPerDimension());
            }

            THEN("there are 10 blocks of the correct size")
            {
                REQUIRE_EQ(bd.getNumberOfBlocks(), blocks);

                for (index_t i = 0; i < blocks - 2; ++i)
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i), bd0);

                for (index_t i = blocks - 2; i < blocks; ++i)
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i), bdn);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the block offsets are correct")
            {
                index_t size0 = bd0.getNumberOfCoefficients();
                index_t sizen = bdn.getNumberOfCoefficients();
                for (index_t i = 0; i < blocks - 2; ++i)
                    REQUIRE_EQ(bd.getOffsetOfBlock(i), i * size0);

                for (index_t i = 0; i < 2; ++i)
                    REQUIRE_EQ(bd.getOffsetOfBlock(blocks - 2 + i),
                               (blocks - 2) * size0 + i * sizen);

                REQUIRE_THROWS(bd.getOffsetOfBlock(blocks));
            }

            THEN("original and partitioned descriptor are not equal")
            {
                REQUIRE_NE(bd, dd);
                REQUIRE_NE(dd, bd);
            }
        }

        WHEN("partitioning it into 10 blocks with chosen sizes")
        {
            index_t blocks = 10;
            IndexVector_t split(blocks);
            split << 1, 2, 3, 4, 5, 6, 7, 8, 9, 57;
            PartitionDescriptor bd(dd, split);

            THEN("the partitioned descriptor has the same number of coefficients and spacing per "
                 "dimension as the original")
            {
                REQUIRE_EQ(dd.getNumberOfCoefficientsPerDimension(),
                           bd.getNumberOfCoefficientsPerDimension());
                REQUIRE_EQ(dd.getSpacingPerDimension(), bd.getSpacingPerDimension());
            }

            THEN("there are 10 blocks of the correct size")
            {
                REQUIRE_EQ(bd.getNumberOfBlocks(), blocks);

                for (index_t i = 0; i < blocks; i++) {
                    auto coeffsPerDim = dd.getNumberOfCoefficientsPerDimension();
                    coeffsPerDim[1] = split[i];

                    REQUIRE_EQ(bd.getDescriptorOfBlock(i).getNumberOfCoefficientsPerDimension(),
                               coeffsPerDim);

                    REQUIRE_EQ(bd.getDescriptorOfBlock(i).getSpacingPerDimension(),
                               bd.getSpacingPerDimension());
                }

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; i++)
                    REQUIRE_EQ(bd.getOffsetOfBlock(i), sizeVector[0] * split.head(i).sum());

                REQUIRE_THROWS(bd.getOffsetOfBlock(blocks));
            }

            THEN("original and partitioned descriptor are not equal")
            {
                REQUIRE_NE(bd, dd);
                REQUIRE_NE(dd, bd);
            }
        }
    }

    GIVEN("a 3D descriptor with blocks")
    {
        IndexVector_t sizeVector(3);
        sizeVector << 101, 42, 750;
        VolumeDescriptor dd(sizeVector);

        sizeVector[2] = 30;
        VolumeDescriptor bd0(sizeVector);
        WHEN("creating 25 blocks")
        {
            index_t blocks = 25;
            PartitionDescriptor bd(dd, blocks);

            THEN("the partitioned descriptor has the same number of coefficients and spacing per "
                 "dimension as the original")
            {
                REQUIRE_EQ(dd.getNumberOfCoefficientsPerDimension(),
                           bd.getNumberOfCoefficientsPerDimension());
                REQUIRE_EQ(dd.getSpacingPerDimension(), bd.getSpacingPerDimension());
            }

            THEN("there are 25 blocks of the correct size")
            {
                REQUIRE_EQ(bd.getNumberOfBlocks(), blocks);

                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i), bd0);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getOffsetOfBlock(i), i * bd0.getNumberOfCoefficients());

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("original and partitioned descriptor are not equal")
            {
                REQUIRE_NE(bd, dd);
                REQUIRE_NE(dd, bd);
            }
        }

        WHEN("creating 25 blocks with chosen sizes")
        {
            index_t blocks = 25;
            IndexVector_t split = IndexVector_t::Constant(blocks, 30);
            split.head(10).array() = 40;
            split.tail(10).array() = 20;
            PartitionDescriptor bd(dd, split);

            THEN("the partitioned descriptor has the same number of coefficients and spacing per "
                 "dimension as the original")
            {
                REQUIRE_EQ(dd.getNumberOfCoefficientsPerDimension(),
                           bd.getNumberOfCoefficientsPerDimension());
                REQUIRE_EQ(dd.getSpacingPerDimension(), bd.getSpacingPerDimension());
            }

            THEN("there are 25 blocks of the correct size")
            {
                REQUIRE_EQ(bd.getNumberOfBlocks(), blocks);

                for (index_t i = 0; i < blocks; ++i) {
                    auto coeffsPerDim = sizeVector;
                    coeffsPerDim[2] = split[i];

                    REQUIRE_EQ(bd.getDescriptorOfBlock(i).getSpacingPerDimension(),
                               dd.getSpacingPerDimension());
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i).getNumberOfCoefficientsPerDimension(),
                               coeffsPerDim);
                }

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getOffsetOfBlock(i),
                               sizeVector.head(2).prod() * split.head(i).sum());

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("original and partitioned descriptor are not equal")
            {
                REQUIRE_NE(bd, dd);
                REQUIRE_NE(dd, bd);
            }
        }
    }
}

TEST_CASE("PartitionDescriptor: Testing clone()")
{
    GIVEN("a 1D PartitionDescriptor")
    {
        IndexVector_t sizeVector(1);
        sizeVector << 3891;
        VolumeDescriptor dd(sizeVector);
        index_t blocks = 21;

        WHEN("cloning the descriptor")
        {
            PartitionDescriptor bd(dd, blocks);
            auto bdClone = bd.clone();

            THEN("it's a real clone")
            {
                REQUIRE_NE(bdClone.get(), &bd);
                REQUIRE_UNARY(dynamic_cast<PartitionDescriptor*>(bdClone.get()));
                REQUIRE_EQ(*bdClone, bd);
            }
        }
    }

    GIVEN("a 2D PartitionDescriptor")
    {
        IndexVector_t sizeVector(2);
        sizeVector << 43, 112;
        VolumeDescriptor dd(sizeVector);
        index_t blocks = 77;

        WHEN("cloning the descriptor")
        {
            PartitionDescriptor bd(dd, blocks);
            auto bdClone = bd.clone();

            THEN("it's a real clone")
            {
                REQUIRE_NE(bdClone.get(), &bd);
                REQUIRE_UNARY(dynamic_cast<PartitionDescriptor*>(bdClone.get()));
                REQUIRE_EQ(*bdClone, bd);
            }
        }
    }

    GIVEN("a 3D PartitionDescriptor")
    {
        IndexVector_t sizeVector(3);
        sizeVector << 47, 11, 53;
        VolumeDescriptor dd(sizeVector);
        index_t blocks = 13;

        WHEN("cloning the descriptor")
        {
            PartitionDescriptor bd(dd, blocks);
            auto bdClone = bd.clone();

            THEN("it's a real clone")
            {
                REQUIRE_NE(bdClone.get(), &bd);
                REQUIRE_UNARY(dynamic_cast<PartitionDescriptor*>(bdClone.get()));
                REQUIRE_EQ(*bdClone, bd);
            }
        }
    }
}

TEST_SUITE_END();
