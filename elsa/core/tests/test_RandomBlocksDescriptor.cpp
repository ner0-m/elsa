/**
 * @file test_RandomBlocksDescriptor.cpp
 *
 * @brief Tests for RandomBlocksDescriptor class
 *
 * @author Nikola Dinev
 */

#include "doctest/doctest.h"
#include "RandomBlocksDescriptor.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

#include <stdexcept>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE("RandomBlocksDescriptors: Testing constructions")
{
    GIVEN("0 an empty descriptor list")
    {
        THEN("construction of a RandomBlocksDescriptor fails")
        {
            REQUIRE_THROWS(RandomBlocksDescriptor(std::vector<std::unique_ptr<DataDescriptor>>(0)));
        }
    }
    GIVEN("five 1D descriptors")
    {
        index_t blocks = 5;
        std::vector<std::unique_ptr<DataDescriptor>> descriptors(0);

        index_t size = 0;
        IndexVector_t offsets(blocks);
        offsets[0] = 0;
        for (index_t i = 0; i < blocks; i++) {
            index_t n = 1 + std::rand() % 100;
            descriptors.emplace_back(
                std::make_unique<VolumeDescriptor>(IndexVector_t::Constant(1, n)));
            size += n;
            if (i != blocks - 1)
                offsets[i + 1] = offsets[i] + n;
        }

        WHEN("creating a RandomBlocksDescriptor containing these descriptors")
        {
            RandomBlocksDescriptor bd(descriptors);

            THEN("there are 5 blocks of the correct size")
            {
                REQUIRE_EQ(bd.getNumberOfBlocks(), blocks);

                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i),
                               *descriptors[static_cast<std::size_t>(i)]);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the new RandomBlocksDescriptor has the correct sizes")
            {
                REQUIRE_EQ(bd.getNumberOfDimensions(), 1);

                IndexVector_t correctSize = IndexVector_t::Constant(1, size);
                REQUIRE_EQ(bd.getNumberOfCoefficientsPerDimension(), correctSize);
                REQUIRE_EQ(bd.getNumberOfCoefficients(), correctSize.prod());

                REQUIRE_EQ(bd.getSpacingPerDimension().size(), 1);
                REQUIRE_EQ(bd.getSpacingPerDimension()[0], 1.0);
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getOffsetOfBlock(i), offsets[i]);

                REQUIRE_THROWS(bd.getOffsetOfBlock(blocks));
            }

            THEN("the block descriptor is different from a monolithic descriptor with the same "
                 "dimensions")
            {
                IndexVector_t size = IndexVector_t::Constant(1, bd.getNumberOfCoefficients());
                VolumeDescriptor dd(size);
                REQUIRE_NE(bd, dd);
                REQUIRE_NE(dd, bd);
            }

            THEN("the block descriptor is different from a RandomBlocksDescriptor with the same "
                 "size but a different number of blocks")
            {
                IndexVector_t size = IndexVector_t::Constant(1, bd.getNumberOfCoefficients());
                VolumeDescriptor dd(size);
                std::vector<std::unique_ptr<DataDescriptor>> vec;
                vec.push_back(dd.clone());
                RandomBlocksDescriptor bd2(vec);
                REQUIRE_NE(bd, bd2);
                REQUIRE_NE(bd2, bd);
            }
        }

        WHEN("creating a RandomBlocksDescriptor containing these descriptors by moving")
        {
            std::vector<std::unique_ptr<DataDescriptor>> tmp(0);
            for (const auto& desc : descriptors)
                tmp.push_back(desc->clone());

            RandomBlocksDescriptor bd(std::move(tmp));

            THEN("there are 5 blocks of the correct size")
            {
                REQUIRE_EQ(bd.getNumberOfBlocks(), blocks);

                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i),
                               *descriptors[static_cast<std::size_t>(i)]);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the new RandomBlocksDescriptor has the correct sizes")
            {
                REQUIRE_EQ(bd.getNumberOfDimensions(), 1);

                IndexVector_t correctSize = IndexVector_t::Constant(1, size);
                REQUIRE_EQ(bd.getNumberOfCoefficientsPerDimension(), correctSize);
                REQUIRE_EQ(bd.getNumberOfCoefficients(), correctSize.prod());

                REQUIRE_EQ(bd.getSpacingPerDimension().size(), 1);
                REQUIRE_EQ(bd.getSpacingPerDimension()[0], 1.0);
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getOffsetOfBlock(i), offsets[i]);

                REQUIRE_THROWS(bd.getOffsetOfBlock(blocks));
            }
        }
    }

    GIVEN("ten 2D descriptors")
    {
        index_t blocks = 10;
        std::vector<std::unique_ptr<DataDescriptor>> descriptors(0);

        index_t size = 0;
        IndexVector_t offsets(blocks);
        offsets[0] = 0;
        for (index_t i = 0; i < blocks; i++) {
            IndexVector_t coeffs(2);
            coeffs << 1 + std::rand() % 100, 1 + std::rand() % 100;
            descriptors.emplace_back(std::make_unique<VolumeDescriptor>(coeffs));
            size += coeffs.prod();
            if (i != blocks - 1)
                offsets[i + 1] = offsets[i] + coeffs.prod();
        }

        WHEN("creating a RandomBlocksDescriptor containing these descriptors")
        {
            RandomBlocksDescriptor bd(descriptors);

            THEN("there are 10 blocks of the correct size")
            {
                REQUIRE_EQ(bd.getNumberOfBlocks(), blocks);

                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i),
                               *descriptors[static_cast<std::size_t>(i)]);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the new RandomBlocksDescriptor has the correct sizes")
            {
                REQUIRE_EQ(bd.getNumberOfDimensions(), 1);

                IndexVector_t correctSize = IndexVector_t::Constant(1, size);
                REQUIRE_EQ(bd.getNumberOfCoefficientsPerDimension(), correctSize);
                REQUIRE_EQ(bd.getNumberOfCoefficients(), correctSize.prod());

                REQUIRE_EQ(bd.getSpacingPerDimension().size(), 1);
                REQUIRE_EQ(bd.getSpacingPerDimension()[0], 1.0);
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getOffsetOfBlock(i), offsets[i]);

                REQUIRE_THROWS(bd.getOffsetOfBlock(blocks));
            }

            THEN("the block descriptor is different from a monolithic descriptor with the same "
                 "dimensions")
            {
                IndexVector_t size = IndexVector_t::Constant(1, bd.getNumberOfCoefficients());
                VolumeDescriptor dd(size);
                REQUIRE_NE(bd, dd);
                REQUIRE_NE(dd, bd);
            }

            THEN("the block descriptor is different from a RandomBlocksDescriptor with the same "
                 "size but a different number of blocks")
            {
                IndexVector_t size = IndexVector_t::Constant(1, bd.getNumberOfCoefficients());
                VolumeDescriptor dd(size);
                std::vector<std::unique_ptr<DataDescriptor>> vec;
                vec.push_back(dd.clone());
                RandomBlocksDescriptor bd2(vec);
                REQUIRE_NE(bd, bd2);
                REQUIRE_NE(bd2, bd);
            }

            THEN("the block descriptor is different from a RandomBlocksDescriptor with the same "
                 "size and number of blocks but different individual block descriptors")
            {
                std::vector<std::unique_ptr<DataDescriptor>> descriptors2;
                for (const auto& desc : descriptors) {
                    auto linearized = std::make_unique<VolumeDescriptor>(
                        IndexVector_t::Constant(1, desc->getNumberOfCoefficients()));
                    descriptors2.push_back(std::move(linearized));
                }

                RandomBlocksDescriptor bd2(descriptors2);
                REQUIRE_NE(bd, bd2);
                REQUIRE_NE(bd2, bd);
            }
        }

        WHEN("creating a RandomBlocksDescriptor containing these descriptors by moving")
        {
            std::vector<std::unique_ptr<DataDescriptor>> tmp(0);
            for (const auto& desc : descriptors)
                tmp.push_back(desc->clone());

            RandomBlocksDescriptor bd(std::move(tmp));

            THEN("there are 10 blocks of the correct size")
            {
                REQUIRE_EQ(bd.getNumberOfBlocks(), blocks);

                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i),
                               *descriptors[static_cast<std::size_t>(i)]);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the new RandomBlocksDescriptor has the correct sizes")
            {
                REQUIRE_EQ(bd.getNumberOfDimensions(), 1);

                IndexVector_t correctSize = IndexVector_t::Constant(1, size);
                REQUIRE_EQ(bd.getNumberOfCoefficientsPerDimension(), correctSize);
                REQUIRE_EQ(bd.getNumberOfCoefficients(), correctSize.prod());

                REQUIRE_EQ(bd.getSpacingPerDimension().size(), 1);
                REQUIRE_EQ(bd.getSpacingPerDimension()[0], 1.0);
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getOffsetOfBlock(i), offsets[i]);

                REQUIRE_THROWS(bd.getOffsetOfBlock(blocks));
            }
        }
    }

    GIVEN("25 descriptors with arbitrary dimensions")
    {
        index_t blocks = 25;
        std::vector<std::unique_ptr<DataDescriptor>> descriptors(0);

        index_t size = 0;
        IndexVector_t offsets(blocks);
        offsets[0] = 0;
        for (index_t i = 0; i < blocks; i++) {
            IndexVector_t coeffs(2 + std::rand() % 4);
            coeffs.setRandom();
            for (int j = 0; j < coeffs.size(); j++)
                coeffs[j] = 1 + std::abs(coeffs[j]) % 100;

            descriptors.emplace_back(std::make_unique<VolumeDescriptor>(coeffs));
            size += coeffs.prod();

            if (i != blocks - 1)
                offsets[i + 1] = offsets[i] + coeffs.prod();
        }

        WHEN("creating a RandomBlocksDescriptor containing these descriptors")
        {
            RandomBlocksDescriptor bd(descriptors);

            THEN("there are 10 blocks of the correct size")
            {
                REQUIRE_EQ(bd.getNumberOfBlocks(), blocks);

                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i),
                               *descriptors[static_cast<std::size_t>(i)]);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the new RandomBlocksDescriptor has the correct sizes")
            {
                REQUIRE_EQ(bd.getNumberOfDimensions(), 1);

                IndexVector_t correctSize = IndexVector_t::Constant(1, size);
                REQUIRE_EQ(bd.getNumberOfCoefficientsPerDimension(), correctSize);
                REQUIRE_EQ(bd.getNumberOfCoefficients(), correctSize.prod());

                REQUIRE_EQ(bd.getSpacingPerDimension().size(), 1);
                REQUIRE_EQ(bd.getSpacingPerDimension()[0], 1.0);
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getOffsetOfBlock(i), offsets[i]);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the block descriptor is different from a monolithic descriptor with the same "
                 "dimensions")
            {
                IndexVector_t size = IndexVector_t::Constant(1, bd.getNumberOfCoefficients());
                VolumeDescriptor dd(size);
                REQUIRE_NE(bd, dd);
                REQUIRE_NE(dd, bd);
            }

            THEN("the block descriptor is different from a RandomBlocksDescriptor with the same "
                 "size but a different number of blocks")
            {
                IndexVector_t size = IndexVector_t::Constant(1, bd.getNumberOfCoefficients());
                VolumeDescriptor dd(size);
                std::vector<std::unique_ptr<DataDescriptor>> vec;
                vec.push_back(dd.clone());
                RandomBlocksDescriptor bd2(vec);
                REQUIRE_NE(bd, bd2);
                REQUIRE_NE(bd2, bd);
            }

            THEN("the block descriptor is different from a RandomBlocksDescriptor with the same "
                 "size and number of blocks but different individual block descriptors")
            {
                std::vector<std::unique_ptr<DataDescriptor>> descriptors2;
                for (const auto& desc : descriptors) {
                    auto linearized = std::make_unique<VolumeDescriptor>(
                        IndexVector_t::Constant(1, desc->getNumberOfCoefficients()));
                    descriptors2.push_back(std::move(linearized));
                }

                RandomBlocksDescriptor bd2(descriptors2);
                REQUIRE_NE(bd, bd2);
                REQUIRE_NE(bd2, bd);
            }
        }

        WHEN("creating a RandomBlocksDescriptor containing these descriptors by moving")
        {
            std::vector<std::unique_ptr<DataDescriptor>> tmp(0);
            for (const auto& desc : descriptors)
                tmp.push_back(desc->clone());

            RandomBlocksDescriptor bd(std::move(tmp));

            THEN("there are 10 blocks of the correct size")
            {
                REQUIRE_EQ(bd.getNumberOfBlocks(), blocks);

                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getDescriptorOfBlock(i),
                               *descriptors[static_cast<std::size_t>(i)]);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }

            THEN("the new RandomBlocksDescriptor has the correct sizes")
            {
                REQUIRE_EQ(bd.getNumberOfDimensions(), 1);

                IndexVector_t correctSize = IndexVector_t::Constant(1, size);
                REQUIRE_EQ(bd.getNumberOfCoefficientsPerDimension(), correctSize);
                REQUIRE_EQ(bd.getNumberOfCoefficients(), correctSize.prod());

                REQUIRE_EQ(bd.getSpacingPerDimension().size(), 1);
                REQUIRE_EQ(bd.getSpacingPerDimension()[0], 1.0);
            }

            THEN("the block offsets are correct")
            {
                for (index_t i = 0; i < blocks; ++i)
                    REQUIRE_EQ(bd.getOffsetOfBlock(i), offsets[i]);

                REQUIRE_THROWS(bd.getDescriptorOfBlock(blocks));
            }
        }
    }
}

TEST_CASE("RandomBlocksDescriptor: Testing clone()")
{
    GIVEN("a RandomBlocksDescriptor of 1D descriptors")
    {
        index_t blocks = 21;
        std::vector<std::unique_ptr<DataDescriptor>> descriptors(0);

        IndexVector_t offsets(blocks);
        offsets[0] = 0;
        for (index_t i = 0; i < blocks; i++) {
            index_t n = 1 + std::rand() % 100;
            descriptors.emplace_back(
                std::make_unique<VolumeDescriptor>(IndexVector_t::Constant(1, n)));
            if (i != blocks - 1)
                offsets[i + 1] = offsets[i] + n;
        }

        RandomBlocksDescriptor bd(descriptors);
        WHEN("cloning the descriptor")
        {
            auto bdClone = bd.clone();

            THEN("it's a real clone")
            {
                REQUIRE_NE(bdClone.get(), &bd);
                REQUIRE_UNARY(is<RandomBlocksDescriptor>(bdClone.get()));
                REQUIRE_EQ(*bdClone, bd);
            }
        }
    }

    GIVEN("a RandomBlocksDescriptor of 2D descriptors")
    {
        index_t blocks = 77;
        std::vector<std::unique_ptr<DataDescriptor>> descriptors(0);

        IndexVector_t offsets(blocks);
        offsets[0] = 0;
        for (index_t i = 0; i < blocks; i++) {
            IndexVector_t coeffs(2);
            coeffs << 1 + std::rand() % 100, 1 + std::rand() % 100;
            descriptors.emplace_back(std::make_unique<VolumeDescriptor>(coeffs));
            if (i != blocks - 1)
                offsets[i + 1] = offsets[i] + coeffs.prod();
        }

        RandomBlocksDescriptor bd(descriptors);
        WHEN("cloning the descriptor")
        {
            auto bdClone = bd.clone();

            THEN("it's a real clone")
            {
                REQUIRE_NE(bdClone.get(), &bd);
                REQUIRE_UNARY(is<RandomBlocksDescriptor>(bdClone.get()));
                REQUIRE_EQ(*bdClone, bd);
            }
        }
    }

    GIVEN("a RandomBlocksDescriptor of descriptors with arbitrary dimensions")
    {
        index_t blocks = 13;
        std::vector<std::unique_ptr<DataDescriptor>> descriptors(0);

        IndexVector_t offsets(blocks);
        offsets[0] = 0;
        for (index_t i = 0; i < blocks; i++) {
            IndexVector_t coeffs(1 + std::rand() % 5);
            coeffs.setRandom();
            for (int j = 0; j < coeffs.size(); j++)
                coeffs[j] = 1 + std::abs(coeffs[j]) % 100;

            descriptors.emplace_back(std::make_unique<VolumeDescriptor>(coeffs));

            if (i != blocks - 1)
                offsets[i + 1] = offsets[i] + coeffs.prod();
        }

        RandomBlocksDescriptor bd(descriptors);
        WHEN("cloning the descriptor")
        {
            auto bdClone = bd.clone();

            THEN("it's a real clone")
            {
                REQUIRE_NE(bdClone.get(), &bd);
                REQUIRE_UNARY(is<RandomBlocksDescriptor>(bdClone.get()));
                REQUIRE_EQ(*bdClone, bd);
            }
        }
    }
}

TEST_SUITE_END();
