/**
 * \file test_DataHandlerMap.cpp
 *
 * \brief Tests for DataHandlerMapCPU class
 *
 * \author David Frank - initial code
 * \author Tobias Lasser - rewrite and code coverage
 */

#include <catch2/catch.hpp>
#include "DataHandlerMapCPU.h"
#include "DataHandlerCPU.h"

template <typename data_t>
int elsa::useCount(const DataHandlerCPU<data_t>& dh)
{
    return dh._data.use_count();
}

using namespace elsa;

TEMPLATE_TEST_CASE("Scenario: Constructing DataHandlerMapCPU", "", float, double, index_t)
{
    GIVEN("a certain size")
    {
        index_t size = 314;

        WHEN("constructing with a given vector")
        {
            Eigen::VectorX<TestType> randVec{size * 2};
            randVec.setRandom();
            const DataHandlerCPU<TestType> dh{randVec};
            const auto dhMap = dh.getBlock(size / 3, size / 3);

            THEN("the DataHandlerMapCPU references the actual vector")
            {
                REQUIRE(dhMap->getSize() == size / 3);

                for (index_t i = 0; i < size / 3; ++i)
                    REQUIRE(&(*dhMap)[i] == &dh[i + size / 3]);
            }
        }

        WHEN("copy constructing")
        {
            Eigen::VectorX<TestType> randVec{size * 2};
            randVec.setRandom();
            const DataHandlerCPU<TestType> dh{randVec};
            const auto dhMap = dh.getBlock(size / 3, size / 3);
            const auto& dhMapRef = static_cast<const DataHandlerMapCPU<TestType>&>(*dhMap);
            const auto dhMapCopy = dhMapRef;

            THEN("the copy references the actual vector")
            {
                REQUIRE(dhMap->getSize() == size / 3);

                for (index_t i = 0; i < size / 3; ++i)
                    REQUIRE(&dhMapCopy[i] == &dh[i + size / 3]);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Testing equality operator on DataHandlerMapCPU", "", float, double,
                   index_t)
{
    GIVEN("some DataHandlerMapCPU")
    {
        index_t size = 314;
        Eigen::VectorX<TestType> randVec{size};
        randVec.setRandom();
        const DataHandlerCPU realDh{randVec};
        const auto dhPtr = realDh.getBlock(0, size);
        const auto& dh = *dhPtr;

        WHEN("comparing to a handler with a different size")
        {
            const DataHandlerCPU<TestType> dh2{size + 1};
            THEN("the result is false")
            {
                REQUIRE_FALSE(dh == dh2);
                REQUIRE_FALSE(dh == *dh2.getBlock(0, size + 1));
            }
        }

        WHEN("comparing to a shallow copy or view of the handler")
        {
            const auto dh2 = realDh;
            THEN("the result is true")
            {
                REQUIRE(dh == dh2);
                REQUIRE(dh == *realDh.getBlock(0, size));
            }
        }

        WHEN("comparing to a deep copy or a view of the deep copy")
        {
            const DataHandlerCPU dh2{randVec};
            THEN("the result is true")
            {
                REQUIRE(dh == dh2);
                REQUIRE(dh == *dh2.getBlock(0, size));
            }
        }

        WHEN("comparing to a handler or map with different data")
        {
            randVec[0] += 1;
            const DataHandlerCPU dh2{randVec};
            THEN("the result is false")
            {
                REQUIRE_FALSE(dh == dh2);
                REQUIRE_FALSE(dh == *dh2.getBlock(0, size));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Assigning to DataHandlerMapCPU", "", float, double, index_t)
{
    GIVEN("a partial DataHandlerMapCPU")
    {
        index_t size = 314;
        DataHandlerCPU<TestType> dh{2 * size};
        const auto dhMap = dh.getBlock(0, size);

        WHEN("copy assigning")
        {
            auto& dhMapRef = static_cast<DataHandlerMapCPU<TestType>&>(*dhMap);

            THEN("sizes must match")
            {
                const DataHandlerCPU<TestType> dh2{3 * size};
                const auto dh2Map = dh2.getBlock(0, 3 * size);
                const auto& dh2MapRef = static_cast<const DataHandlerMapCPU<TestType>&>(*dh2Map);
                REQUIRE_THROWS(dhMapRef = dh2MapRef);
            }

            THEN("a deep copy is performed")
            {
                const DataHandlerCPU<TestType> dh2{size};
                const auto dh2Map = dh2.getBlock(0, size);
                const auto& dh2MapRef = static_cast<const DataHandlerMapCPU<TestType>&>(*dh2Map);

                dhMapRef = dh2MapRef;
                REQUIRE(useCount(dh) == 1);
                REQUIRE(useCount(dh2) == 1);
                REQUIRE(&dh[0] == &dhMapRef[0]);
                REQUIRE(dhMapRef == dh2);
                REQUIRE(&dh[0] != &dh2MapRef[0]);
            }
        }

        WHEN("copy assigning a DataHandlerCPU through base pointers")
        {
            Eigen::VectorX<TestType> randVec{size};
            randVec.setRandom();
            const std::unique_ptr<const DataHandler<TestType>> dh2Ptr =
                std::make_unique<const DataHandlerCPU<TestType>>(randVec);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<TestType>> bigDh =
                    std::make_unique<DataHandlerCPU<TestType>>(2 * size);
                REQUIRE_THROWS(*dhMap = *bigDh);
            }

            const auto dhCopy = dh;
            *dhMap = *dh2Ptr;
            THEN("a deep copy is performed")
            {
                REQUIRE(useCount(dh) == 1);
                REQUIRE(useCount(dhCopy) == 1);

                for (index_t i = 0; i < size; i++)
                    REQUIRE(dh[i] == (*dh2Ptr)[i]);

                REQUIRE(*dhMap == *dh2Ptr);
                REQUIRE(&(*dhMap)[0] == &dh[0]);
            }
        }

        WHEN("copy assigning a partial DataHandlerMapCPU through base pointers")
        {
            DataHandler<TestType>* dhPtr = &dh;
            const auto dhCopy = dh;
            Eigen::VectorX<TestType> randVec{2 * size};
            randVec.setRandom();
            const DataHandlerCPU<TestType> dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const auto bigDh = dh2.getBlock(0, 2 * size);
                REQUIRE_THROWS(*dhMap = *bigDh);
            }

            *dhMap = *dh2Map;
            THEN("a deep copy is performed")
            {
                REQUIRE(useCount(dh) == 1);
                REQUIRE(useCount(dhCopy) == 1);

                for (index_t i = 0; i < size; i++)
                    REQUIRE(dh[i] == dh2[i]);

                REQUIRE(*dhMap == *dh2Map);
                REQUIRE(&(*dhMap)[0] == &dh[0]);
            }
        }

        WHEN("copy assigning a full DataHandlerMapCPU (aka a view) through base pointers")
        {
            DataHandler<TestType>* dhPtr = &dh;
            const auto dhCopy = dh;
            Eigen::VectorX<TestType> randVec{size};
            randVec.setRandom();
            const DataHandlerCPU<TestType> dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<TestType>> bigDh =
                    std::make_unique<DataHandlerCPU<TestType>>(2 * size);
                REQUIRE_THROWS(*dhMap = *bigDh->getBlock(0, 2 * size));
            }

            *dhMap = *dh2Map;
            THEN("a deep copy is performed")
            {
                REQUIRE(useCount(dh) == 1);
                REQUIRE(useCount(dhCopy) == 1);

                for (index_t i = 0; i < size; i++)
                    REQUIRE(dh[i] == dh2[i]);

                REQUIRE(&(*dhMap)[0] == &dh[0]);
            }
        }

        WHEN("\"move\" assigning a DataHandlerMapCPU through base pointers")
        {
            Eigen::VectorX<TestType> randVec{size};
            randVec.setRandom();
            const std::unique_ptr<DataHandler<TestType>> dh2Ptr =
                std::make_unique<DataHandlerCPU<TestType>>(randVec);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<TestType>> bigDh =
                    std::make_unique<DataHandlerCPU<TestType>>(2 * size);
                REQUIRE_THROWS(*dhMap = std::move(*bigDh));
            }

            const auto dhCopy = dh;
            *dhMap = std::move(*dh2Ptr);
            THEN("a deep copy is performed")
            {
                REQUIRE(useCount(dh) == 1);
                REQUIRE(useCount(dhCopy) == 1);

                for (index_t i = 0; i < size; i++)
                    REQUIRE(dh[i] == (*dh2Ptr)[i]);

                REQUIRE(*dhMap == *dh2Ptr);
                REQUIRE(&(*dhMap)[0] == &dh[0]);
            }
        }

        WHEN("\"move\" assigning a partial DataHandlerMapCPU through base pointers")
        {
            DataHandler<TestType>* dhPtr = &dh;
            const auto dhCopy = dh;
            Eigen::VectorX<TestType> randVec{2 * size};
            randVec.setRandom();
            DataHandlerCPU<TestType> dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const auto bigDh = dh2.getBlock(0, 2 * size);
                REQUIRE_THROWS(*dhMap = std::move(*bigDh));
            }

            *dhMap = std::move(*dh2Map);
            THEN("a deep copy is performed")
            {
                REQUIRE(useCount(dh) == 1);
                REQUIRE(useCount(dhCopy) == 1);

                for (index_t i = 0; i < size; i++)
                    REQUIRE(dh[i] == dh2[i]);

                REQUIRE(*dhMap == *dh2Map);
                REQUIRE(&(*dhMap)[0] == &dh[0]);
            }
        }

        WHEN("\"move\" assigning a full DataHandlerMapCPU (aka a view) through base pointers")
        {
            DataHandler<TestType>* dhPtr = &dh;
            const auto dhCopy = dh;
            Eigen::VectorX<TestType> randVec{size};
            randVec.setRandom();
            DataHandlerCPU<TestType> dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<TestType>> bigDh =
                    std::make_unique<DataHandlerCPU<TestType>>(2 * size);
                REQUIRE_THROWS(*dhMap = std::move(*bigDh->getBlock(0, 2 * size)));
            }

            *dhMap = std::move(*dh2Map);
            THEN("a deep copy is performed")
            {
                REQUIRE(useCount(dh) == 1);
                REQUIRE(useCount(dhCopy) == 1);

                for (index_t i = 0; i < size; i++)
                    REQUIRE(dh[i] == dh2[i]);

                REQUIRE(&(*dhMap)[0] == &dh[0]);
            }
        }
    }

    GIVEN("a full DataHandlerMapCPU (aka a view)")
    {
        index_t size = 314;
        DataHandlerCPU<TestType> dh{size};
        const auto dhMap = dh.getBlock(0, size);

        WHEN("copy assigning and both maps are views")
        {
            auto& dhMapRef = static_cast<DataHandlerMapCPU<TestType>&>(*dhMap);

            THEN("sizes must match")
            {
                const DataHandlerCPU<TestType> dh2{3 * size};
                const auto dh2Map = dh2.getBlock(0, 3 * size);
                const auto& dh2MapRef = static_cast<const DataHandlerMapCPU<TestType>&>(*dh2Map);
                REQUIRE_THROWS(dhMapRef = dh2MapRef);
            }

            THEN("a shallow copy is performed")
            {
                const DataHandlerCPU<TestType> dh2{size};
                const auto dh2Map = dh2.getBlock(0, size);
                const auto& dh2MapRef = static_cast<const DataHandlerMapCPU<TestType>&>(*dh2Map);
                dhMapRef = dh2MapRef;
                REQUIRE(useCount(dh) == 2);
                REQUIRE(dh == dh2);
                REQUIRE(dh == dhMapRef);
                REQUIRE(dh == dh2MapRef);
                dhMapRef[0] = 1;
                REQUIRE(&dhMapRef[0] == &dh[0]);
                REQUIRE(&dhMapRef[0] != &dh2MapRef[0]);
            }
        }

        WHEN("copy assigning a DataHandlerMapCPU through base pointers")
        {
            Eigen::VectorX<TestType> randVec{size};
            randVec.setRandom();
            const std::unique_ptr<DataHandler<TestType>> dh2Ptr =
                std::make_unique<DataHandlerCPU<TestType>>(randVec);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<TestType>> bigDh =
                    std::make_unique<DataHandlerCPU<TestType>>(2 * size);
                REQUIRE_THROWS(*dhMap = *bigDh);
            }

            *dhMap = *dh2Ptr;
            THEN("a shallow copy is performed")
            {
                REQUIRE(useCount(dh) == 2);
                REQUIRE(dh == *dh2Ptr);
                REQUIRE(*dhMap == *dh2Ptr);
                REQUIRE(&(*dhMap)[0] == &dh[0]);
                dh[0] = 1;
                REQUIRE(&(*dhMap)[0] == &dh[0]);
                REQUIRE(&dh[0] != &(*dh2Ptr)[0]);
            }
        }

        WHEN("copy assigning a partial DataHandlerMapCPU through base pointers")
        {
            const auto dhCopy = dh;
            Eigen::VectorX<TestType> randVec{2 * size};
            randVec.setRandom();
            const DataHandlerCPU<TestType> dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const auto bigDh = dh2.getBlock(0, 2 * size);
                REQUIRE_THROWS(*dhMap = *bigDh);
            }

            *dhMap = *dh2Map;
            THEN("a deep copy is performed")
            {
                REQUIRE(useCount(dh) == 1);
                REQUIRE(useCount(dhCopy) == 1);

                for (index_t i = 0; i < size; i++)
                    REQUIRE(dh[i] == dh2[i]);

                REQUIRE(*dhMap == *dh2Map);
                REQUIRE(&(*dhMap)[0] == &dh[0]);
            }
        }

        WHEN("copy assigning a full DataHandlerMapCPU (aka a view) through base pointers")
        {
            Eigen::VectorX<TestType> randVec{size};
            randVec.setRandom();
            const DataHandlerCPU<TestType> dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<TestType>> bigDh =
                    std::make_unique<DataHandlerCPU<TestType>>(2 * size);
                REQUIRE_THROWS(*dhMap = *bigDh->getBlock(0, 2 * size));
            }

            *dhMap = *dh2Map;
            THEN("a shallow copy is performed")
            {
                REQUIRE(useCount(dh) == 2);
                REQUIRE(dh == dh2);
                REQUIRE(*dhMap == *dh2Map);
                REQUIRE(&(*dhMap)[0] == &dh[0]);
                dh[0] = 1;
                REQUIRE(&(*dhMap)[0] == &dh[0]);
                REQUIRE(&dh[0] != &dh2[0]);
            }
        }

        WHEN("\"move\" assigning a DataHandlerCPU through base pointers")
        {
            Eigen::VectorX<TestType> randVec{size};
            randVec.setRandom();
            const std::unique_ptr<DataHandler<TestType>> dh2Ptr =
                std::make_unique<DataHandlerCPU<TestType>>(randVec);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<TestType>> bigDh =
                    std::make_unique<DataHandlerCPU<TestType>>(2 * size);
                REQUIRE_THROWS(*dhMap = std::move(*bigDh));
            }

            *dhMap = std::move(*dh2Ptr);
            THEN("a shallow copy is performed")
            {
                REQUIRE(useCount(dh) == 2);
                REQUIRE(dh == *dh2Ptr);
                REQUIRE(*dhMap == *dh2Ptr);
                REQUIRE(&(*dhMap)[0] == &dh[0]);
                dh[0] = 1;
                REQUIRE(&(*dhMap)[0] == &dh[0]);
                REQUIRE(&dh[0] != &(*dh2Ptr)[0]);
            }
        }

        WHEN("\"move\" assigning a partial DataHandlerMapCPU through base pointers")
        {
            const auto dhCopy = dh;
            Eigen::VectorX<TestType> randVec{2 * size};
            randVec.setRandom();
            DataHandlerCPU<TestType> dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const auto bigDh = dh2.getBlock(0, 2 * size);
                REQUIRE_THROWS(*dhMap = std::move(*bigDh));
            }

            *dhMap = std::move(*dh2Map);
            THEN("a deep copy is performed")
            {
                REQUIRE(useCount(dh) == 1);
                REQUIRE(useCount(dhCopy) == 1);

                for (index_t i = 0; i < size; i++)
                    REQUIRE(dh[i] == dh2[i]);

                REQUIRE(*dhMap == *dh2Map);
                REQUIRE(&(*dhMap)[0] == &dh[0]);
            }
        }

        WHEN("\"move\" assigning a full DataHandlerMapCPU (aka a view) through base pointers")
        {
            Eigen::VectorX<TestType> randVec{size};
            randVec.setRandom();
            DataHandlerCPU<TestType> dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<TestType>> bigDh =
                    std::make_unique<DataHandlerCPU<TestType>>(2 * size);
                REQUIRE_THROWS(*dhMap = std::move(*bigDh->getBlock(0, 2 * size)));
            }

            *dhMap = std::move(*dh2Map);
            THEN("a shallow copy is performed")
            {
                REQUIRE(useCount(dh) == 2);
                REQUIRE(dh == dh2);
                REQUIRE(*dhMap == *dh2Map);
                REQUIRE(&(*dhMap)[0] == &dh[0]);
                dh[0] = 1;
                REQUIRE(&(*dhMap)[0] == &dh[0]);
                REQUIRE(&dh[0] != &dh2[0]);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Cloning DataHandlerMapCPU", "", float, double, index_t)
{
    GIVEN("a full DataHandlerMapCPU (aka a view)")
    {
        index_t size = 728;
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> dataVec(size);
        dataVec.setRandom();
        DataHandlerCPU realDh(dataVec);
        auto dhPtr = realDh.getBlock(0, size);
        auto& dh = *dhPtr;

        WHEN("cloning")
        {
            auto dhClone = dh.clone();

            THEN("a shallow copy is produced")
            {
                REQUIRE(dhClone.get() != &dh);

                REQUIRE(dhClone->getSize() == dh.getSize());

                REQUIRE(useCount(realDh) == 2);

                REQUIRE(*dhClone == dh);

                dh[0] = 1;
                REQUIRE(*dhClone != dh);
            }
        }
    }

    GIVEN("a partial DataHandlerMapCPU")
    {
        index_t size = 728;
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> dataVec(size);
        dataVec.setRandom();
        DataHandlerCPU realDh(dataVec);
        auto dhPtr = realDh.getBlock(0, size / 2);
        auto& dh = *dhPtr;

        WHEN("a deep copy is produced")
        {
            auto dhClone = dh.clone();

            THEN("everything matches")
            {
                REQUIRE(dhClone.get() != &dh);

                REQUIRE(dhClone->getSize() == dh.getSize());

                REQUIRE(useCount(realDh) == 1);

                REQUIRE(dh == *dhClone);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Testing the reduction operations of DataHandlerMapCPU", "", float,
                   double, index_t)
{
    GIVEN("some DataHandlerMapCPU")
    {
        index_t size = 284;

        WHEN("putting in some random data")
        {
            Eigen::Matrix<TestType, Eigen::Dynamic, 1> randVec(size * 2);
            randVec.setRandom();
            DataHandlerCPU realDh(randVec);
            auto dhPtr = realDh.getBlock(size / 3, size);
            auto& dh = *dhPtr;

            THEN("the reductions work as expected")
            {
                REQUIRE(dh.sum() == Approx(randVec.middleRows(size / 3, size).sum()));
                REQUIRE(dh.l1Norm()
                        == Approx(randVec.middleRows(size / 3, size).array().abs().sum()));
                REQUIRE(dh.lInfNorm()
                        == Approx(randVec.middleRows(size / 3, size).array().abs().maxCoeff()));
                REQUIRE(dh.squaredL2Norm()
                        == Approx(randVec.middleRows(size / 3, size).squaredNorm()));

                Eigen::Matrix<TestType, Eigen::Dynamic, 1> randVec2(size);
                randVec2.setRandom();
                DataHandlerCPU realDh2(randVec2);
                auto dh2Ptr = realDh2.getBlock(0, size);
                auto& dh2 = *dh2Ptr;
                REQUIRE(dh.dot(dh2) == Approx(randVec.middleRows(size / 3, size).dot(randVec2)));

                DataHandlerCPU dhCPU(randVec2);
                REQUIRE(dh.dot(dhCPU) == Approx(randVec.middleRows(size / 3, size).dot(randVec2)));
            }

            THEN("the dot product expects correctly sized arguments")
            {
                index_t wrongSize = size - 1;
                Eigen::Matrix<TestType, Eigen::Dynamic, 1> randVec2(wrongSize);
                randVec2.setRandom();
                DataHandlerCPU dh2(randVec2);

                REQUIRE_THROWS_AS(dh.dot(dh2), std::invalid_argument);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Testing the element-wise operations of DataHandlerMapCPU", "", float,
                   double, index_t)
{
    GIVEN("some DataHandlerMapCPU")
    {
        index_t size = 567;

        WHEN("putting in some random data")
        {
            Eigen::Matrix<TestType, Eigen::Dynamic, 1> randVec(size);
            randVec.setRandom();
            DataHandlerCPU realDh(randVec);
            auto dhPtr = realDh.getBlock(0, size);
            auto& dh = static_cast<DataHandlerMapCPU<TestType>&>(*dhPtr);

            THEN("the element-wise unary operations work as expected")
            {
                auto dhSquared = dh.square();
                for (index_t i = 0; i < size; ++i)
                    REQUIRE((*dhSquared)[i] == Approx(randVec(i) * randVec(i)));

                auto dhSqrt = dh.sqrt();
                for (index_t i = 0; i < size; ++i)
                    if (randVec(i) >= 0)
                        REQUIRE((*dhSqrt)[i]
                                == Approx(static_cast<TestType>(std::sqrt(randVec(i)))));

                auto dhExp = dh.exp();
                for (index_t i = 0; i < size; ++i)
                    REQUIRE((*dhExp)[i] == Approx(static_cast<TestType>(std::exp(randVec(i)))));

                auto dhLog = dh.log();
                for (index_t i = 0; i < size; ++i)
                    if (randVec(i) > 0)
                        REQUIRE((*dhLog)[i] == Approx(static_cast<TestType>(log(randVec(i)))));
            }

            THEN("the element-wise binary vector operations work as expected")
            {
                DataHandlerCPU<TestType> bigDh{2 * size};

                REQUIRE_THROWS(dh += bigDh);
                REQUIRE_THROWS(dh -= bigDh);
                REQUIRE_THROWS(dh *= bigDh);
                REQUIRE_THROWS(dh /= bigDh);

                DataHandlerCPU realOldDh(randVec);
                auto oldDhPtr = realOldDh.getBlock(0, size);
                auto& oldDh = static_cast<DataHandlerMapCPU<TestType>&>(*oldDhPtr);

                Eigen::Matrix<TestType, Eigen::Dynamic, 1> randVec2(size);
                randVec2.setRandom();
                DataHandlerCPU dhCPU(randVec2);
                auto dh2Ptr = dhCPU.getBlock(0, size);
                auto& dh2 = *dh2Ptr;

                dh += dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] + dh2[i]);

                dh = oldDh;
                dh += dhCPU;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] + dhCPU[i]);

                dh = oldDh;
                dh -= dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] - dh2[i]);

                dh = oldDh;
                dh -= dhCPU;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] - dhCPU[i]);

                dh = oldDh;
                dh *= dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] * dh2[i]);

                dh = oldDh;
                dh *= dhCPU;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] * dhCPU[i]);

                dh = oldDh;
                dh /= dh2;
                for (index_t i = 0; i < size; ++i)
                    if (dh2[i] != 0)
                        REQUIRE(dh[i] == oldDh[i] / dh2[i]);

                dh = oldDh;
                dh /= dhCPU;
                for (index_t i = 0; i < size; ++i)
                    if (dhCPU[i] != 0)
                        REQUIRE(dh[i] == oldDh[i] / dhCPU[i]);
            }

            THEN("the element-wise binary scalar operations work as expected")
            {
                DataHandlerCPU realOldDh(randVec);
                auto oldDhPtr = realOldDh.getBlock(0, size);
                auto& oldDh = static_cast<DataHandlerMapCPU<TestType>&>(*oldDhPtr);
                TestType scalar = std::is_integral_v<TestType> ? 3 : 3.5;

                dh += scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] + scalar);

                dh = oldDh;
                dh -= scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] - scalar);

                dh = oldDh;
                dh *= scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] * scalar);

                dh = oldDh;
                dh /= scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] / scalar);
            }

            THEN("the element-wise assignment of a scalar works as expected")
            {
                TestType scalar = std::is_integral_v<TestType> ? 47 : 47.11;

                dh = scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == scalar);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Testing arithmetic operations with DataHandler arguments", "", float,
                   double, index_t)
{
    GIVEN("some DataHandlers")
    {
        index_t size = 1095;
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> randVec(size);
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> randVec2(size);
        randVec.setRandom();
        randVec2.setRandom();
        DataHandlerCPU realDh(randVec);
        auto dhPtr = realDh.getBlock(0, size);
        auto& dh = *dhPtr;
        DataHandlerCPU realDh2(randVec2);
        auto dh2Ptr = realDh2.getBlock(0, size);
        auto& dh2 = *dh2Ptr;
        DataHandlerCPU dhCPU(randVec2);

        THEN("the binary element-wise operations work as expected")
        {
            DataHandlerCPU<TestType> bigDh{2 * size};

            REQUIRE_THROWS(dh + bigDh);
            REQUIRE_THROWS(dh - bigDh);
            REQUIRE_THROWS(dh * bigDh);
            REQUIRE_THROWS(dh / bigDh);

            auto resultPlus = dh + dh2;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultPlus)[i] == dh[i] + dh2[i]);

            resultPlus = dh + dhCPU;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultPlus)[i] == dh[i] + dhCPU[i]);

            auto resultMinus = dh - dh2;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultMinus)[i] == dh[i] - dh2[i]);

            resultMinus = dh - dhCPU;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultMinus)[i] == dh[i] - dhCPU[i]);

            auto resultMult = dh * dh2;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultMult)[i] == dh[i] * dh2[i]);

            resultMult = dh * dhCPU;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultMult)[i] == dh[i] * dhCPU[i]);

            auto resultDiv = dh / dh2;
            for (index_t i = 0; i < size; ++i)
                if (dh2[i] != 0)
                    REQUIRE((*resultDiv)[i] == dh[i] / dh2[i]);

            resultDiv = dh / dhCPU;
            for (index_t i = 0; i < size; ++i)
                if (dh2[i] != 0)
                    REQUIRE((*resultDiv)[i] == dh[i] / dhCPU[i]);
        }

        THEN("the operations with a scalar work as expected")
        {
            TestType scalar = std::is_integral_v<TestType> ? 4 : 4.7;

            auto resultScalarPlus = scalar + dh;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultScalarPlus)[i] == scalar + dh[i]);

            auto resultPlusScalar = dh + scalar;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultPlusScalar)[i] == dh[i] + scalar);

            auto resultScalarMinus = scalar - dh;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultScalarMinus)[i] == scalar - dh[i]);

            auto resultMinusScalar = dh - scalar;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultMinusScalar)[i] == dh[i] - scalar);

            auto resultScalarMult = scalar * dh;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultScalarMult)[i] == scalar * dh[i]);

            auto resultMultScalar = dh * scalar;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultMultScalar)[i] == dh[i] * scalar);

            auto resultScalarDiv = scalar / dh;
            for (index_t i = 0; i < size; ++i)
                if (dh[i] != 0)
                    REQUIRE((*resultScalarDiv)[i] == scalar / dh[i]);

            auto resultDivScalar = dh / scalar;
            for (index_t i = 0; i < size; ++i)
                REQUIRE((*resultDivScalar)[i] == dh[i] / scalar);
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Referencing blocks of DataHandlerMapCPU", "", float, double, index_t)
{
    GIVEN("some DataHandlerMapCPU")
    {
        index_t size = 728;
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> dataVec(size);
        DataHandlerCPU realDh(dataVec);
        auto dhPtr = realDh.getBlock(0, size);
        auto& dh = *dhPtr;

        WHEN("getting the reference to a block")
        {
            REQUIRE_THROWS(dh.getBlock(size, 1));
            REQUIRE_THROWS(dh.getBlock(0, size + 1));

            auto dhBlock = dh.getBlock(size / 3, size / 2);

            THEN("returned data handler references the correct elements")
            {
                REQUIRE(dhBlock->getSize() == size / 2);

                for (index_t i = 0; i < size / 2; i++)
                    REQUIRE(&(*dhBlock)[i] == &dh[i + size / 3]);
            }
        }
    }

    GIVEN("a const DataHandlerMapCPU")
    {
        index_t size = 728;
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> dataVec(size);
        const DataHandlerCPU realDh(dataVec);
        auto dhPtr = realDh.getBlock(0, size);
        auto& dh = *dhPtr;

        WHEN("getting the reference to a block")
        {
            REQUIRE_THROWS(dh.getBlock(size, 1));
            REQUIRE_THROWS(dh.getBlock(0, size + 1));

            auto dhBlock = dh.getBlock(size / 3, size / 2);

            THEN("returned data handler references the correct elements")
            {
                REQUIRE(dhBlock->getSize() == size / 2);

                for (index_t i = 0; i < size / 2; i++)
                    REQUIRE(&(*dhBlock)[i] == &dh[i + size / 3]);
            }
        }
    }
}