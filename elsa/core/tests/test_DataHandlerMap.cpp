/**
 * @file test_DataHandlerMap.cpp
 *
 * @brief Tests for DataHandlerMaps - DataHandlerMapCPU and DataHandlerMapGPU
 *
 * @author David Frank - initial code
 * @author Tobias Lasser - rewrite and code coverage
 * @author Jens Petit - refactoring into TEMPLATE_PRODUCT_TEST_CASE
 */

#include <catch2/catch.hpp>
#include "DataHandlerMapCPU.h"
#include "DataHandlerCPU.h"
#include "testHelpers.h"

#ifdef ELSA_CUDA_VECTOR
#include "DataHandlerGPU.h"
#include "DataHandlerMapGPU.h"
#endif

using namespace elsa;

// for testing the copy-on-write mechanism
template <typename data_t>
long elsa::useCount(const DataHandlerCPU<data_t>& dh)
{
    return dh._data.use_count();
}

#ifdef ELSA_CUDA_VECTOR
// for testing the copy-on-write mechanism
template <typename data_t>
long elsa::useCount(const DataHandlerGPU<data_t>& dh)
{
    return dh._data.use_count();
}
#endif

// Helper to provide the correct map based on the handler type
template <typename Handler>
struct MapToHandler {
    using map =
        std::conditional_t<std::is_same_v<DataHandlerCPU<typename Handler::value_type>, Handler>,
                           DataHandlerMapCPU<typename Handler::value_type>,
#ifdef ELSA_CUDA_VECTOR
                           DataHandlerMapGPU<typename Handler::value_type>>;
#else
                           DataHandlerMapCPU<typename Handler::value_type>>;
#endif
};

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Constructing DataHandlerMap", "",
                           (DataHandlerCPU, DataHandlerGPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Constructing DataHandlerMap", "", (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::value_type;

    GIVEN("a certain size")
    {
        index_t size = 314;

        WHEN("constructing with a given vector")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size * 2};
            randVec.setRandom();
            const TestType dh{randVec};
            const auto dhMap = dh.getBlock(size / 3, size / 3);

            THEN("the DataHandlerMap references the actual vector")
            {
                REQUIRE(dhMap->getSize() == size / 3);

                for (index_t i = 0; i < size / 3; ++i)
                    REQUIRE(&(*dhMap)[i] == &dh[i + size / 3]);
            }
        }

        WHEN("copy constructing")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size * 2};
            randVec.setRandom();
            const TestType dh{randVec};
            const auto dhMap = dh.getBlock(size / 3, size / 3);

            const auto& dhMapRef = static_cast<const typename MapToHandler<TestType>::map&>(*dhMap);

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

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing equality operator on DataHandlerMap", "",
                           (DataHandlerCPU, DataHandlerGPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing equality operator on DataHandlerMap", "",
                           (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandlerMap")
    {
        index_t size = 314;
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
        randVec.setRandom();
        const TestType realDh{randVec};
        const auto dhPtr = realDh.getBlock(0, size);
        const auto& dh = *dhPtr;

        WHEN("comparing to a handler with a different size")
        {
            const TestType dh2{size + 1};
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
            const TestType dh2{randVec};
            THEN("the result is true")
            {
                REQUIRE(dh == dh2);
                REQUIRE(dh == *dh2.getBlock(0, size));
            }
        }

        WHEN("comparing to a handler or map with different data")
        {
            randVec[0] += 1;
            const TestType dh2{randVec};
            THEN("the result is false")
            {
                REQUIRE_FALSE(dh == dh2);
                REQUIRE_FALSE(dh == *dh2.getBlock(0, size));
            }
        }
    }
}

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Assigning to DataHandlerMap", "",
                           (DataHandlerCPU, DataHandlerGPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Assigning to DataHandlerMap", "", (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::value_type;

    GIVEN("a partial DataHandlerMap")
    {
        index_t size = 314;
        TestType dh{2 * size};
        dh = 0;
        const auto dhMap = dh.getBlock(0, size);

        WHEN("copy assigning")
        {
            auto& dhMapRef = static_cast<typename MapToHandler<TestType>::map&>(*dhMap);

            THEN("sizes must match")
            {
                const TestType dh2{3 * size};
                const auto dh2Map = dh2.getBlock(0, 3 * size);
                const auto& dh2MapRef =
                    static_cast<const typename MapToHandler<TestType>::map&>(*dh2Map);
                REQUIRE_THROWS(dhMapRef = dh2MapRef);
            }

            THEN("a deep copy is performed")
            {
                Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
                randVec.setRandom();
                const TestType dh2{randVec};
                const auto dh2Map = dh2.getBlock(0, size);
                const auto& dh2MapRef =
                    static_cast<const typename MapToHandler<TestType>::map&>(*dh2Map);

                dhMapRef = dh2MapRef;
                REQUIRE(useCount(dh) == 1);
                REQUIRE(useCount(dh2) == 1);
                REQUIRE(&dh[0] == &dhMapRef[0]);
                REQUIRE(dhMapRef == dh2);
                REQUIRE(&dh[0] != &dh2MapRef[0]);
            }
        }

        WHEN("copy assigning a DataHandlerMap through base pointers")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
            const std::unique_ptr<const DataHandler<data_t>> dh2Ptr =
                std::make_unique<const TestType>(randVec);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<TestType>(2 * size);
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

        WHEN("copy assigning a partial DataHandlerMap through base pointers")
        {
            const auto dhCopy = dh;
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{2 * size};
            randVec.setRandom();
            const TestType dh2{randVec};
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

        WHEN("copy assigning a full DataHandlerMap (aka a view) through base pointers")
        {
            const auto dhCopy = dh;
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
            const TestType dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<TestType>(2 * size);
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

        WHEN("\"move\" assigning a DataHandlerMap through base pointers")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
            const std::unique_ptr<DataHandler<data_t>> dh2Ptr = std::make_unique<TestType>(randVec);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<TestType>(2 * size);
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

        WHEN("\"move\" assigning a partial DataHandlerMap through base pointers")
        {
            const auto dhCopy = dh;
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{2 * size};
            randVec.setRandom();
            TestType dh2{randVec};
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

        WHEN("\"move\" assigning a full DataHandlerMap (aka a view) through base pointers")
        {
            const auto dhCopy = dh;
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
            TestType dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<TestType>(2 * size);
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

    GIVEN("a full DataHandlerMap (aka a view)")
    {
        index_t size = 314;
        TestType dh{size};
        const auto dhMap = dh.getBlock(0, size);

        WHEN("copy assigning and both maps are views")
        {
            auto& dhMapRef = static_cast<typename MapToHandler<TestType>::map&>(*dhMap);

            THEN("sizes must match")
            {
                const TestType dh2{3 * size};
                const auto dh2Map = dh2.getBlock(0, 3 * size);
                const auto& dh2MapRef =
                    static_cast<const typename MapToHandler<TestType>::map&>(*dh2Map);
                REQUIRE_THROWS(dhMapRef = dh2MapRef);
            }

            THEN("a shallow copy is performed")
            {
                const TestType dh2{size};
                const auto dh2Map = dh2.getBlock(0, size);
                const auto& dh2MapRef =
                    static_cast<const typename MapToHandler<TestType>::map&>(*dh2Map);
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

        WHEN("copy assigning a DataHandlerMap through base pointers")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
            const std::unique_ptr<DataHandler<data_t>> dh2Ptr = std::make_unique<TestType>(randVec);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<TestType>(2 * size);
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

        WHEN("copy assigning a partial DataHandlerMap through base pointers")
        {
            const auto dhCopy = dh;
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{2 * size};
            randVec.setRandom();
            const TestType dh2{randVec};
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

        WHEN("copy assigning a full DataHandlerMap (aka a view) through base pointers")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
            const TestType dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<TestType>(2 * size);
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

        WHEN("\"move\" assigning a DataHandler through base pointers")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
            const std::unique_ptr<DataHandler<data_t>> dh2Ptr = std::make_unique<TestType>(randVec);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<TestType>(2 * size);
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

        WHEN("\"move\" assigning a partial DataHandlerMap through base pointers")
        {
            const auto dhCopy = dh;
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{2 * size};
            randVec.setRandom();
            TestType dh2{randVec};
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

        WHEN("\"move\" assigning a full DataHandlerMap (aka a view) through base pointers")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
            TestType dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<TestType>(2 * size);
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

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Cloning DataHandlerMap", "", (DataHandlerCPU, DataHandlerGPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Cloning DataHandlerMap", "", (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::value_type;

    GIVEN("a full DataHandlerMap (aka a view)")
    {
        index_t size = 728;
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> dataVec(size);
        dataVec.setRandom();
        TestType realDh(dataVec);
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

    GIVEN("a partial DataHandlerMap")
    {
        index_t size = 728;
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> dataVec(size);
        dataVec.setRandom();
        TestType realDh(dataVec);
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

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing the reduction operations of DataHandlerMap", "",
                           (DataHandlerCPU, DataHandlerGPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing the reduction operations of DataHandlerMap", "",
                           (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandlerMap")
    {
        index_t size = 284;

        WHEN("putting in some random data")
        {
            auto randVec = generateRandomMatrix<data_t>(size * 2);
            TestType realDh(randVec);
            auto dhPtr = realDh.getBlock(size / 3, size);
            auto& dh = *dhPtr;

            THEN("the reductions work as expected")
            {
                REQUIRE(checkSameNumbers(dh.sum(), randVec.middleRows(size / 3, size).sum()));
                REQUIRE(dh.l0PseudoNorm()
                        == (randVec.middleRows(size / 3, size).array().cwiseAbs()
                            >= std::numeric_limits<GetFloatingPointType_t<data_t>>::epsilon())
                               .count());
                REQUIRE(checkSameNumbers(dh.l1Norm(),
                                         randVec.middleRows(size / 3, size).array().abs().sum()));
                REQUIRE(checkSameNumbers(
                    dh.lInfNorm(), randVec.middleRows(size / 3, size).array().abs().maxCoeff()));
                REQUIRE(checkSameNumbers(dh.squaredL2Norm(),
                                         randVec.middleRows(size / 3, size).squaredNorm()));
                REQUIRE(checkSameNumbers(dh.l2Norm(), randVec.middleRows(size / 3, size).norm()));

                auto randVec2 = generateRandomMatrix<data_t>(size);
                TestType realDh2(randVec2);
                auto dh2Ptr = realDh2.getBlock(0, size);
                auto& dh2 = *dh2Ptr;
                REQUIRE(checkSameNumbers(dh.dot(dh2),
                                         randVec.middleRows(size / 3, size).dot(randVec2)));

                TestType dhCPU(randVec2);
                REQUIRE(checkSameNumbers(dh.dot(dhCPU),
                                         randVec.middleRows(size / 3, size).dot(randVec2)));
            }

            THEN("the dot product expects correctly sized arguments")
            {
                index_t wrongSize = size - 1;
                Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec2(wrongSize);
                randVec2.setRandom();
                TestType dh2(randVec2);

                REQUIRE_THROWS_AS(dh.dot(dh2), InvalidArgumentError);
            }
        }
    }
}

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing the element-wise operations of DataHandlerMap", "",
                           (DataHandlerCPU, DataHandlerGPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing the element-wise operations of DataHandlerMap", "",
                           (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandlerMap")
    {
        index_t size = 567;

        WHEN("putting in some random data")
        {
            auto randVec = generateRandomMatrix<data_t>(size);
            TestType realDh(randVec);

            auto dhPtr = realDh.getBlock(0, size);
            auto& dh = static_cast<typename MapToHandler<TestType>::map&>(*dhPtr);

            THEN("the element-wise binary vector operations work as expected")
            {
                TestType bigDh{2 * size};

                REQUIRE_THROWS(dh += bigDh);
                REQUIRE_THROWS(dh -= bigDh);
                REQUIRE_THROWS(dh *= bigDh);
                REQUIRE_THROWS(dh /= bigDh);

                TestType realOldDh(randVec);
                auto oldDhPtr = realOldDh.getBlock(0, size);
                auto& oldDh = static_cast<typename MapToHandler<TestType>::map&>(*oldDhPtr);

                auto randVec2 = generateRandomMatrix<data_t>(size);

                TestType dhCPU(randVec2);
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
                    REQUIRE(checkSameNumbers(dh[i], oldDh[i] * dh2[i]));

                dh = oldDh;
                dh *= dhCPU;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(checkSameNumbers(dh[i], oldDh[i] * dhCPU[i]));

                dh = oldDh;
                dh /= dh2;
                for (index_t i = 0; i < size; ++i)
                    if (dh2[i] != data_t(0))
                        // due to floating point arithmetic less precision
                        REQUIRE(checkSameNumbers(dh[i], oldDh[i] / dh2[i]));

                dh = oldDh;
                dh /= dhCPU;
                for (index_t i = 0; i < size; ++i)
                    if (dhCPU[i] != data_t(0))
                        // due to floating point arithmetic less precision
                        REQUIRE(checkSameNumbers(dh[i], oldDh[i] / dhCPU[i]));
            }

            THEN("the element-wise binary scalar operations work as expected")
            {
                TestType realOldDh(randVec);
                auto oldDhPtr = realOldDh.getBlock(0, size);
                auto& oldDh = static_cast<typename MapToHandler<TestType>::map&>(*oldDhPtr);
                data_t scalar = std::is_integral_v<data_t> ? 3 : data_t(3.5f);

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
                    REQUIRE(checkSameNumbers(dh[i], oldDh[i] / scalar));
            }

            THEN("the element-wise assignment of a scalar works as expected")
            {
                auto scalar = std::is_integral_v<data_t> ? data_t(47) : data_t(47.11f);

                dh = scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == scalar);
            }
        }
    }
}

#ifdef ELSA_CUDA_VECTOR
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Referencing blocks of DataHandlerMap", "",
                           (DataHandlerCPU, DataHandlerGPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#else
TEMPLATE_PRODUCT_TEST_CASE("Scenario: Referencing blocks of DataHandlerMap", "", (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
#endif
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandlerMap")
    {
        index_t size = 728;
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> dataVec(size);
        TestType realDh(dataVec);
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

    GIVEN("a const DataHandlerMap")
    {
        index_t size = 728;
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> dataVec(size);
        const TestType realDh(dataVec);
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
