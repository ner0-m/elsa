#include <catch2/catch.hpp>

#include "IdenticalBlocksDescriptor.h"
#include "PartitionDescriptor.h"
#include "RandomBlocksDescriptor.h"
#include "BlockLinearOperator.h"
#include "VolumeDescriptor.h"
#include "Identity.h"
#include "Scaling.h"

using namespace elsa;

TEMPLATE_TEST_CASE("Constructing a BlockLinearOperator ", "", float, double)
{
    using BlockType = typename BlockLinearOperator<TestType>::BlockType;
    using OperatoList = std::vector<std::unique_ptr<LinearOperator<TestType>>>;

    index_t rows = 6, cols = 8;
    IndexVector_t size2D(2);
    size2D << rows, cols;
    VolumeDescriptor dd{size2D};

    auto sizeBlock = size2D;
    sizeBlock[1] *= 2;
    VolumeDescriptor bdBase{sizeBlock};
    PartitionDescriptor bd{bdBase, 2};

    GIVEN("an empty operator list")
    {

        OperatoList ops;

        WHEN("creating a BlockLinearOperator from it")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(ops, BlockType::COL),
                                  InvalidArgumentError);

                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(ops, BlockType::ROW),
                                  InvalidArgumentError);

                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(dd, bd, ops, BlockType::ROW),
                                  InvalidArgumentError);

                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(bd, dd, ops, BlockType::COL),
                                  InvalidArgumentError);
            }
        }
    }

    GIVEN("a list of identical operators")
    {
        auto iop1 = std::make_unique<Identity<TestType>>(dd);
        std::vector<std::unique_ptr<LinearOperator<TestType>>> ops(0);
        ops.push_back(std::move(iop1->clone()));
        ops.push_back(std::move(iop1->clone()));
        WHEN("creating a BlockLinearOperator from it")
        {
            BlockLinearOperator<TestType> blockOp1{ops, BlockType::COL};
            BlockLinearOperator<TestType> blockOp2{ops, BlockType::ROW};

            THEN("the BlockLinearOperator contains the correct operators")
            {
                REQUIRE(blockOp1.numberOfOps() == 2);
                REQUIRE(blockOp1.getIthOperator(0) == *iop1);
                REQUIRE(blockOp1.getIthOperator(1) == *iop1);
                REQUIRE(blockOp2.numberOfOps() == 2);
                REQUIRE(blockOp2.getIthOperator(0) == *iop1);
                REQUIRE(blockOp2.getIthOperator(1) == *iop1);
            }

            THEN("the automatically generated operator descriptors are correct")
            {
                REQUIRE(blockOp1.getDomainDescriptor() == bd);
                REQUIRE(blockOp1.getRangeDescriptor() == dd);
                REQUIRE(blockOp2.getDomainDescriptor() == dd);
                REQUIRE(blockOp2.getRangeDescriptor() == bd);
            }
        }

        WHEN("creating a BlockLinearOperator with user specified descriptors from it")
        {
            IdenticalBlocksDescriptor bd2{2, dd};
            VolumeDescriptor ddLinearized{IndexVector_t::Constant(1, dd.getNumberOfCoefficients())};
            BlockLinearOperator<TestType> blockOp1{bd2, ddLinearized, ops, BlockType::COL};
            BlockLinearOperator<TestType> blockOp2{ddLinearized, bd2, ops, BlockType::ROW};

            THEN("the BlockLinearOperator contains the correct operators")
            {
                REQUIRE(blockOp1.numberOfOps() == 2);
                REQUIRE(blockOp1.getIthOperator(0) == *iop1);
                REQUIRE(blockOp1.getIthOperator(1) == *iop1);
                REQUIRE(blockOp2.numberOfOps() == 2);
                REQUIRE(blockOp2.getIthOperator(0) == *iop1);
                REQUIRE(blockOp2.getIthOperator(1) == *iop1);
            }

            THEN("the automatically generated operator descriptors are correct")
            {
                REQUIRE(blockOp1.getDomainDescriptor() == bd2);
                REQUIRE(blockOp1.getRangeDescriptor() == ddLinearized);
                REQUIRE(blockOp2.getDomainDescriptor() == ddLinearized);
                REQUIRE(blockOp2.getRangeDescriptor() == bd2);
            }
        }

        WHEN("creating a BlockLinearOperator with invalid user specified descriptors from it")
        {
            THEN("an exception is thrown")
            {
                IdenticalBlocksDescriptor blocksOfIncorrectSize{2, bd};

                // wrong number of coefficients
                REQUIRE_THROWS_AS(
                    BlockLinearOperator<TestType>(blocksOfIncorrectSize, dd, ops, BlockType::COL),
                    InvalidArgumentError);
                REQUIRE_THROWS_AS(
                    BlockLinearOperator<TestType>(dd, blocksOfIncorrectSize, ops, BlockType::ROW),
                    InvalidArgumentError);
                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(bd, bdBase, ops, BlockType::COL),
                                  InvalidArgumentError);
                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(bdBase, bd, ops, BlockType::ROW),
                                  InvalidArgumentError);

                // descriptor not of block type
                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(dd, bdBase, ops, BlockType::ROW),
                                  InvalidArgumentError);
                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(bdBase, dd, ops, BlockType::COL),
                                  InvalidArgumentError);
            }
        }
    }

    GIVEN("a list of operators with different number of dimensions")
    {
        auto iop1 = std::make_unique<Identity<TestType>>(dd);
        VolumeDescriptor ddLinearized{IndexVector_t::Constant(1, dd.getNumberOfCoefficients())};
        auto iop2 = std::make_unique<Identity<TestType>>(ddLinearized);
        std::vector<std::unique_ptr<LinearOperator<TestType>>> ops(0);
        ops.push_back(std::move(iop1->clone()));
        ops.push_back(std::move(iop2->clone()));

        std::vector<std::unique_ptr<DataDescriptor>> descVec;
        descVec.push_back(dd.clone());
        descVec.push_back(ddLinearized.clone());
        RandomBlocksDescriptor expectedBlocks(std::move(descVec));

        WHEN("creating a BlockLinearOperator from it")
        {
            BlockLinearOperator<TestType> blockOp1{ops, BlockType::COL};
            BlockLinearOperator<TestType> blockOp2{ops, BlockType::ROW};

            THEN("the BlockLinearOperator contains the correct operators")
            {
                REQUIRE(blockOp1.numberOfOps() == 2);
                REQUIRE(blockOp1.getIthOperator(0) == *iop1);
                REQUIRE(blockOp1.getIthOperator(1) == *iop2);
                REQUIRE(blockOp2.numberOfOps() == 2);
                REQUIRE(blockOp2.getIthOperator(0) == *iop1);
                REQUIRE(blockOp2.getIthOperator(1) == *iop2);
            }

            THEN("the automatically generated operator descriptors are correct")
            {
                REQUIRE(blockOp1.getDomainDescriptor() == expectedBlocks);
                REQUIRE(blockOp1.getRangeDescriptor() == ddLinearized);
                REQUIRE(blockOp2.getDomainDescriptor() == ddLinearized);
                REQUIRE(blockOp2.getRangeDescriptor() == expectedBlocks);
            }
        }

        WHEN("creating a BlockLinearOperator with user specified descriptors from it")
        {
            IdenticalBlocksDescriptor bd2{2, dd};
            VolumeDescriptor ddLinearized{IndexVector_t::Constant(1, dd.getNumberOfCoefficients())};
            BlockLinearOperator<TestType> blockOp1{bd2, dd, ops, BlockType::COL};
            BlockLinearOperator<TestType> blockOp2{dd, bd2, ops, BlockType::ROW};

            THEN("the BlockLinearOperator contains the correct operators")
            {
                REQUIRE(blockOp1.numberOfOps() == 2);
                REQUIRE(blockOp1.getIthOperator(0) == *iop1);
                REQUIRE(blockOp1.getIthOperator(1) == *iop2);
                REQUIRE(blockOp2.numberOfOps() == 2);
                REQUIRE(blockOp2.getIthOperator(0) == *iop1);
                REQUIRE(blockOp2.getIthOperator(1) == *iop2);
            }

            THEN("the automatically generated operator descriptors are correct")
            {
                REQUIRE(blockOp1.getDomainDescriptor() == bd2);
                REQUIRE(blockOp1.getRangeDescriptor() == dd);
                REQUIRE(blockOp2.getDomainDescriptor() == dd);
                REQUIRE(blockOp2.getRangeDescriptor() == bd2);
            }
        }

        WHEN("creating a BlockLinearOperator with invalid user specified descriptors from it")
        {
            THEN("an exception is thrown")
            {
                IdenticalBlocksDescriptor blocksOfIncorrectSize{2, bd};

                // wrong number of coefficients
                REQUIRE_THROWS_AS(
                    BlockLinearOperator<TestType>(blocksOfIncorrectSize, dd, ops, BlockType::COL),
                    InvalidArgumentError);
                REQUIRE_THROWS_AS(
                    BlockLinearOperator<TestType>(dd, blocksOfIncorrectSize, ops, BlockType::ROW),
                    InvalidArgumentError);
                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(bd, bdBase, ops, BlockType::COL),
                                  InvalidArgumentError);
                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(bdBase, bd, ops, BlockType::ROW),
                                  InvalidArgumentError);

                // descriptor not of block type
                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(dd, bdBase, ops, BlockType::ROW),
                                  InvalidArgumentError);
                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(bdBase, dd, ops, BlockType::COL),
                                  InvalidArgumentError);
            }
        }
    }

    GIVEN("a list of operators that cannot be stacked in the given fashion")
    {
        auto iop1 = std::make_unique<Identity<TestType>>(dd);
        auto iop2 = std::make_unique<Identity<TestType>>(bdBase);
        std::vector<std::unique_ptr<LinearOperator<TestType>>> ops(0);
        ops.push_back(std::move(iop1->clone()));
        ops.push_back(std::move(iop2->clone()));

        WHEN("creating a BlockLinearOperator from it")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(ops, BlockType::COL),
                                  InvalidArgumentError);
                REQUIRE_THROWS_AS(BlockLinearOperator<TestType>(ops, BlockType::ROW),
                                  InvalidArgumentError);
            }
        }
    }

    GIVEN("a list of operators whose descriptors have equal number of coefficients per dimension "
          "but different spacing")
    {
        auto iop1 = std::make_unique<Identity<TestType>>(dd);
        VolumeDescriptor dds2{size2D, dd.getSpacingPerDimension() * 2};
        auto iop2 = std::make_unique<Identity<TestType>>(dds2);
        std::vector<std::unique_ptr<LinearOperator<TestType>>> ops(0);
        ops.push_back(std::move(iop1->clone()));
        ops.push_back(std::move(iop2->clone()));

        WHEN("creating a BlockLinearOperator from it")
        {
            BlockLinearOperator<TestType> blockOp1{ops, BlockType::COL};
            BlockLinearOperator<TestType> blockOp2{ops, BlockType::ROW};

            THEN("the BlockLinearOperator contains the correct operators")
            {
                REQUIRE(blockOp1.numberOfOps() == 2);
                REQUIRE(blockOp1.getIthOperator(0) == *iop1);
                REQUIRE(blockOp1.getIthOperator(1) == *iop2);
                REQUIRE(blockOp2.numberOfOps() == 2);
                REQUIRE(blockOp2.getIthOperator(0) == *iop1);
                REQUIRE(blockOp2.getIthOperator(1) == *iop2);
            }

            THEN("the automatically generated operator descriptors are correct and have a uniform "
                 "spacing of one")
            {
                REQUIRE(blockOp1.getDomainDescriptor() == bd);
                REQUIRE(blockOp1.getRangeDescriptor() == dd);
                REQUIRE(blockOp2.getDomainDescriptor() == dd);
                REQUIRE(blockOp2.getRangeDescriptor() == bd);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: BlockLinearOperator apply", "", float, double)
{
    using BlockType = typename BlockLinearOperator<TestType>::BlockType;
    index_t rows = 4, cols = 8, numBlks = 3;
    GIVEN("a 2D volume")
    {
        IndexVector_t domainIndex(2);
        domainIndex << rows, cols;
        VolumeDescriptor domainDesc(domainIndex);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> vec(domainDesc.getNumberOfCoefficients());
        vec << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2;

        DataContainer<TestType> domain(domainDesc, vec);

        WHEN("3 Scaling operators are applied to it, in a ROW-Block fashion")
        {
            IndexVector_t rangeIndex(2);
            rangeIndex << rows, cols * numBlks;
            VolumeDescriptor rangeDesc(rangeIndex);
            DataContainer<TestType> range(rangeDesc);

            TestType scale1 = 2.f;
            TestType scale2 = 3.f;
            TestType scale3 = 4.f;

            std::vector<std::unique_ptr<LinearOperator<TestType>>> ops(0);
            ops.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale1)));
            ops.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale2)));
            ops.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale3)));

            BlockLinearOperator<TestType> blockOp(ops, BlockType::ROW);

            blockOp.apply(domain, range);

            THEN("the for the first block, x was multiplied by 2.")
            {
                for (int i = 0; i < rows * cols; i++)
                    REQUIRE(range[i] == vec[i] * scale1);
            }
            THEN("the for the second block, x was multiplied by 3.")
            {
                for (int i = 0; i < rows * cols; i++)
                    REQUIRE(range[i + rows * cols] == vec[i] * scale2);
            }
            THEN("the for the third block, x was multiplied by 4.")
            {
                for (int i = 0; i < rows * cols; i++)
                    REQUIRE(range[i + rows * cols * 2] == vec[i] * scale3);
            }
        }
    }

    GIVEN("a 2D volume with 3 blocks")
    {
        IndexVector_t blockIndex(2);
        blockIndex << rows, cols;
        VolumeDescriptor blockDesc(blockIndex);
        IdenticalBlocksDescriptor domainDesc(numBlks, blockDesc);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> vec(domainDesc.getNumberOfCoefficients());
        vec << 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,

            1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2,

            2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,

            1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2;

        DataContainer<TestType> domain(domainDesc, vec);

        WHEN("3 Scaling operators are applied to it in a COL-Block fashion")
        {
            IndexVector_t rangeIndex(2);
            rangeIndex << rows, cols;
            VolumeDescriptor rangeDesc(rangeIndex);
            DataContainer<TestType> range(rangeDesc);

            float scale1 = 2.f;
            float scale2 = 3.f;
            float scale3 = 4.f;

            std::vector<std::unique_ptr<LinearOperator<TestType>>> ops(0);
            ops.push_back(std::move(std::make_unique<Scaling<TestType>>(blockDesc, scale1)));
            ops.push_back(std::move(std::make_unique<Scaling<TestType>>(blockDesc, scale2)));
            ops.push_back(std::move(std::make_unique<Scaling<TestType>>(blockDesc, scale3)));

            BlockLinearOperator<TestType> blockOp(ops, BlockType::COL);

            blockOp.apply(domain, range);

            THEN("then the range data, is a component wise addition of the 3 blocks each "
                 "scaled with the corresponding factor")
            {
                auto size = rows * cols;
                for (int i = 0; i < size; i++) {
                    REQUIRE(range[i]
                            == vec[i] * scale1 + vec[i + size] * scale2
                                   + vec[i + size * 2] * scale3);
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: BlockLinearOperator applyAdjoint", "", float, double)
{
    using BlockType = typename BlockLinearOperator<TestType>::BlockType;
    index_t rows = 4, cols = 8, numBlks = 3;
    GIVEN("a 2D volume with 3 blocks")
    {
        IndexVector_t rangeIndex(2);
        rangeIndex << rows, cols * numBlks;
        VolumeDescriptor rangeDesc(rangeIndex);

        const index_t n = rangeDesc.getNumberOfCoefficients();
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> rangeVec(n);
        rangeVec.topRows(n / 3).setConstant(5);
        rangeVec.middleRows(n / 3, 2 * n / 3).setConstant(7);
        rangeVec.bottomRows(n / 3).setOnes();

        DataContainer<TestType> range(rangeDesc, rangeVec);

        WHEN("applying the adjoint of 3 Scaling operators ordered in a ROW-Block fashion")
        {
            IndexVector_t domainIndex(2);
            domainIndex << rows, cols;
            VolumeDescriptor domainDesc(domainIndex);

            TestType scale1 = 2.f;
            TestType scale2 = 3.f;
            TestType scale3 = 4.f;

            std::vector<std::unique_ptr<LinearOperator<TestType>>> ops(0);
            ops.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale1)));
            ops.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale2)));
            ops.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale3)));

            BlockLinearOperator<TestType> blockOp(ops, BlockType::ROW);

            auto result = blockOp.applyAdjoint(range);

            THEN("then the result is a component wise addition of the 3 blocks each "
                 "scaled with the corresponding factor")
            {
                for (int i = 0; i < rows * cols; i++)
                    REQUIRE(result[i] == 35);
            }
        }
    }

    GIVEN("a 2D volume")
    {
        IndexVector_t rangeIndex(2);
        rangeIndex << rows, cols;
        VolumeDescriptor rangeDesc(rangeIndex);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> vec(rangeDesc.getNumberOfCoefficients());
        vec << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2;

        DataContainer<TestType> range(rangeDesc, vec);

        WHEN("applying the adjoint of 3 Scaling operators ordered in a COL-Block fashion")
        {
            TestType scale1 = 2.f;
            TestType scale2 = 3.f;
            TestType scale3 = 4.f;

            std::vector<std::unique_ptr<LinearOperator<TestType>>> ops(0);
            ops.push_back(std::move(std::make_unique<Scaling<TestType>>(rangeDesc, scale1)));
            ops.push_back(std::move(std::make_unique<Scaling<TestType>>(rangeDesc, scale2)));
            ops.push_back(std::move(std::make_unique<Scaling<TestType>>(rangeDesc, scale3)));

            BlockLinearOperator<TestType> blockOp(ops, BlockType::COL);

            auto result = blockOp.applyAdjoint(range);

            THEN("the for the first block, x was multiplied by 2.")
            {
                for (int i = 0; i < rows * cols; i++)
                    REQUIRE(result[i] == vec[i] * scale1);
            }
            THEN("the for the second block, x was multiplied by 3.")
            {
                for (int i = 0; i < rows * cols; i++)
                    REQUIRE(result[i + rows * cols] == vec[i] * scale2);
            }
            THEN("the for the third block, x was multiplied by 4.")
            {
                for (int i = 0; i < rows * cols; i++)
                    REQUIRE(result[i + rows * cols * 2] == vec[i] * scale3);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Cloning BlockLinearOperator", "", float, double)
{
    using BlockType = typename BlockLinearOperator<TestType>::BlockType;
    index_t rows = 4, cols = 8;
    GIVEN("a ROW BlockLinearOperator")
    {
        IndexVector_t domainIndex(2);
        domainIndex << rows, cols;
        VolumeDescriptor domainDesc(domainIndex);

        TestType scale1 = 2.f;
        TestType scale2 = 3.f;
        TestType scale3 = 4.f;

        std::vector<std::unique_ptr<LinearOperator<TestType>>> ops(0);
        ops.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale1)));
        ops.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale2)));
        ops.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale3)));

        BlockLinearOperator<TestType> blockOp(ops, BlockType::ROW);

        WHEN("cloning the operator")
        {
            auto bloClone = blockOp.clone();

            THEN("it's a real clone")
            {
                REQUIRE(bloClone.get() != &blockOp);
                REQUIRE(dynamic_cast<BlockLinearOperator<TestType>*>(bloClone.get()));
                REQUIRE(blockOp == *bloClone);
            }
        }
    }

    GIVEN("a COL BlockLinearOperator")
    {
        IndexVector_t rangeIndex(2);
        rangeIndex << rows, cols;
        VolumeDescriptor rangeDesc(rangeIndex);

        TestType scale1 = 2.f;
        TestType scale2 = 3.f;
        TestType scale3 = 4.f;

        std::vector<std::unique_ptr<LinearOperator<TestType>>> ops(0);
        ops.push_back(std::move(std::make_unique<Scaling<TestType>>(rangeDesc, scale1)));
        ops.push_back(std::move(std::make_unique<Scaling<TestType>>(rangeDesc, scale2)));
        ops.push_back(std::move(std::make_unique<Scaling<TestType>>(rangeDesc, scale3)));

        BlockLinearOperator<TestType> blockOp(ops, BlockType::COL);

        WHEN("cloning the operator")
        {
            auto bloClone = blockOp.clone();

            THEN("it's a real clone")
            {
                REQUIRE(bloClone.get() != &blockOp);
                REQUIRE(dynamic_cast<BlockLinearOperator<TestType>*>(bloClone.get()));
                REQUIRE(blockOp == *bloClone);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Comparing BlockLinearOperators", "", float, double)
{
    using BlockType = typename BlockLinearOperator<TestType>::BlockType;
    index_t rows = 4, cols = 8;
    GIVEN("a BlockLinearOperator")
    {
        IndexVector_t domainIndex(2);
        domainIndex << rows, cols;
        VolumeDescriptor domainDesc(domainIndex);

        TestType scale1 = 2.f;
        TestType scale2 = 3.f;
        TestType scale3 = 4.f;

        std::vector<std::unique_ptr<LinearOperator<TestType>>> ops(0);
        ops.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale1)));
        ops.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale2)));
        ops.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale3)));

        BlockLinearOperator<TestType> blockOp(ops, BlockType::ROW);

        WHEN("comparing the operator to a leaf of itself")
        {
            auto blockOpLeaf = leaf(blockOp);

            THEN("they are not equal")
            {
                REQUIRE(blockOp != blockOpLeaf);
                REQUIRE(blockOpLeaf != blockOp);
            }
        }

        WHEN("comparing the operator to a BlockLinearOperator containing the same operators, but "
             "stacked differently")
        {
            BlockLinearOperator<TestType> blockOp2(ops, BlockType::COL);

            THEN("they are not equal")
            {
                REQUIRE(blockOp != blockOp2);
                REQUIRE(blockOp2 != blockOp);
            }
        }

        WHEN("comparing the operator to a BlockLinearOperator containing different operators")
        {
            std::vector<std::unique_ptr<LinearOperator<TestType>>> ops2(0);
            ops2.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale1)));
            ops2.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale2)));
            ops2.push_back(std::move(std::make_unique<Scaling<TestType>>(domainDesc, scale2)));
            BlockLinearOperator<TestType> blockOp2(ops2, BlockType::ROW);

            THEN("they are not equal")
            {
                REQUIRE(blockOp != blockOp2);
                REQUIRE(blockOp2 != blockOp);
            }
        }
    }
}
