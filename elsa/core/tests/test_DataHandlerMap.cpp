/**
 * @file test_DataHandlerMap.cpp
 *
 * @brief Tests for DataHandlerMaps - DataHandlerMapGPU
 *
 * @author David Frank - initial code
 * @author Tobias Lasser - rewrite and code coverage
 * @author Jens Petit - refactoring into TEMPLATE_PRODUCT_TEST_CASE
 */

#include "doctest/doctest.h"
#include "DataHandlerCPU.h"
#include "testHelpers.h"

#ifdef ELSA_CUDA_VECTOR
#include "DataHandlerGPU.h"
#include "DataHandlerMapGPU.h"
#endif

using namespace elsa;
using namespace elsa;

// for testing the copy-on-write mechanism
template <typename data_t>
long elsa::useCount(const DataHandlerCPU<data_t>& dh)
{
    return dh.data_.use_count();
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
#ifdef ELSA_CUDA_VECTOR
                           DataHandlerMapGPU<typename Handler::value_type>>;
#else
                           void>;
#endif
};

using CPUTypeTuple =
    std::tuple<DataHandlerCPU<float>, DataHandlerCPU<double>, DataHandlerCPU<complex<float>>,
               DataHandlerCPU<complex<double>>, DataHandlerCPU<index_t>>;

TYPE_TO_STRING(DataHandlerCPU<float>);
TYPE_TO_STRING(DataHandlerCPU<double>);
TYPE_TO_STRING(DataHandlerCPU<index_t>);
TYPE_TO_STRING(DataHandlerCPU<complex<float>>);
TYPE_TO_STRING(DataHandlerCPU<complex<double>>);

#ifdef ELSA_CUDA_VECTOR
using GPUTypeTuple =
    std::tuple<DataHandlerGPU<float>, DataHandlerGPU<double>, DataHandlerGPU<complex<float>>,
               DataHandlerGPU<complex<double>>, DataHandlerGPU<index_t>>;

TYPE_TO_STRING(DataHandlerGPU<float>);
TYPE_TO_STRING(DataHandlerGPU<double>);
TYPE_TO_STRING(DataHandlerGPU<index_t>);
TYPE_TO_STRING(DataHandlerGPU<complex<float>>);
TYPE_TO_STRING(DataHandlerGPU<complex<double>>);
#endif

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE_DEFINE("DataHandlerMap: Testing construction", TestType,
                          datahandlermap_construction)
{
    using data_t = typename TestType::value_type;

    GIVEN("a certain size")
    {
        index_t size = 314;

        WHEN("constructing with a given vector")
        {
            Vector_t<data_t> randVec{size * 2};
            randVec.setRandom();
            const TestType dh{randVec};
            const auto dhMap = dh.getBlock(size / 3, size / 3);

            THEN("the DataHandlerMap references the actual vector")
            {
                REQUIRE_EQ(dhMap->getSize(), size / 3);

                for (index_t i = 0; i < size / 3; ++i)
                    REQUIRE_EQ(&(*dhMap)[i], &dh[i + size / 3]);
            }
        }

        WHEN("copy constructing")
        {
            Vector_t<data_t> randVec{size * 2};
            randVec.setRandom();
            const TestType dh{randVec};
            const auto dhMap = dh.getBlock(size / 3, size / 3);

            const auto& dhMapRef = static_cast<const typename MapToHandler<TestType>::map&>(*dhMap);

            const auto dhMapCopy = dhMapRef;

            THEN("the copy references the actual vector")
            {
                REQUIRE_EQ(dhMap->getSize(), size / 3);

                for (index_t i = 0; i < size / 3; ++i)
                    REQUIRE_EQ(&dhMapCopy[i], &dh[i + size / 3]);
            }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataHandlerMap: Testing equality operator", TestType,
                          datahandlermap_eqoperator)
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandlerMap")
    {
        index_t size = 314;
        Vector_t<data_t> randVec{size};
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
                REQUIRE_EQ(dh, dh2);
                REQUIRE_EQ(dh, *realDh.getBlock(0, size));
            }
        }

        WHEN("comparing to a deep copy or a view of the deep copy")
        {
            const TestType dh2{randVec};
            THEN("the result is true")
            {
                REQUIRE_EQ(dh, dh2);
                REQUIRE_EQ(dh, *dh2.getBlock(0, size));
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

TEST_CASE_TEMPLATE_DEFINE("DataHandlerMap: Testing assignment to DataHandlerMap", TestType,
                          datahandlermap_assign)
{
    using data_t = typename TestType::value_type;
    using MapType = typename MapToHandler<TestType>::map;

    const index_t size = 314;

    // Constructo DataHandler with 2x the size
    TestType dh{2 * size};
    dh = 0;

    // Create map to the first half of the DH
    const auto dhMap = dh.getBlock(0, size);

    REQUIRE_EQ(dhMap->getSize(), size);
    REQUIRE_EQ(dhMap->getSize(), dh.getSize() / 2);

    GIVEN("A reference to the concrete DataHandlerMap (TestType)")
    {
        auto& dhMapRef = static_cast<typename MapToHandler<TestType>::map&>(*dhMap);

        WHEN("Copy-assigning a DataHandler with the same size to the map")
        {
            Vector_t<data_t> randVec{size};
            randVec.setRandom();
            const TestType dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);
            const auto& dh2MapRef = static_cast<const MapType&>(*dh2Map);

            dhMapRef = dh2MapRef;

            THEN("a deep copy is performed")
            {
                REQUIRE_EQ(useCount(dh), 1);
                REQUIRE_EQ(useCount(dh2), 1);

                REQUIRE_EQ(dhMapRef, dh2);
                REQUIRE_UNARY(isCwiseApprox(dhMapRef, dh2MapRef));

                REQUIRE_EQ(&dh[0], &dhMapRef[0]);
                REQUIRE_NE(&dh[0], &dh2MapRef[0]);

                // Changing the original DataHandler, doesn't change the new one
                dh[0] *= 4;

                THEN("Changing the original DataHandler doesn't affect the new one")
                {
                    REQUIRE_UNARY(checkApproxEq(dh[0], dhMapRef[0]));
                    REQUIRE_UNARY(checkApproxNe(dh[0], dh2[0]));
                    REQUIRE_UNARY(checkApproxNe(dh[0], dh2MapRef[0]));
                }
            }
        }

        WHEN("Copy-assigning a DataHandler with a different size to the map")
        {
            const TestType dh2{3 * size};
            const auto dh2Map = dh2.getBlock(0, 3 * size);
            const auto& dh2MapRef =
                static_cast<const typename MapToHandler<TestType>::map&>(*dh2Map);

            THEN("the assignment throws") { REQUIRE_THROWS(dhMapRef = dh2MapRef); }
        }
    }

    GIVEN("Given the base pointer to the map")
    {
        Vector_t<data_t> randVec{size};
        randVec.setRandom();
        const std::unique_ptr<const DataHandler<data_t>> dh2Ptr =
            std::make_unique<const TestType>(randVec);

        WHEN("Copy-assigning a DataHandler base pointer of the same size")
        {
            // TODO: why do we need dhCopy?
            const auto dhCopy = dh;
            *dhMap = *dh2Ptr;
            THEN("a deep copy is performed")
            {
                REQUIRE_EQ(useCount(dh), 1);
                REQUIRE_EQ(useCount(dhCopy), 1);

                for (index_t i = 0; i < size; i++)
                    REQUIRE_EQ(dh[i], (*dh2Ptr)[i]);

                REQUIRE_EQ(*dhMap, *dh2Ptr);
                REQUIRE_EQ(&(*dhMap)[0], &dh[0]);

                // Changing the original DataHandler, doesn't change the new one
                dh[0] *= 2;

                THEN("Changing the original DataHandler doesn't affect the new one")
                {
                    REQUIRE_UNARY(checkApproxEq((*dhMap)[0], dh[0]));
                    REQUIRE_UNARY(checkApproxNe((*dhMap)[0], (*dh2Ptr)[0]));
                }
            }
        }

        WHEN("Copy-assigning a DataHandler base pointer of a different size")
        {
            const std::unique_ptr<DataHandler<data_t>> bigDh = std::make_unique<TestType>(2 * size);
            THEN("The assigning throws") { REQUIRE_THROWS(*dhMap = *bigDh); }
        }

        WHEN("Copy-assigning a block of a DataHandlerMap though the base pointer")
        {
            const auto dhCopy = dh;
            Vector_t<data_t> randVec{2 * size};
            randVec.setRandom();

            const TestType dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            WHEN("The sizes of the block and DataHandler are the same")
            {
                *dhMap = *dh2Map;
                THEN("a deep copy is performed")
                {
                    REQUIRE_EQ(useCount(dh), 1);
                    REQUIRE_EQ(useCount(dhCopy), 1);

                    for (index_t i = 0; i < size; i++)
                        REQUIRE_EQ(dh[i], dh2[i]);

                    REQUIRE_EQ(*dhMap, *dh2Map);
                    REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                }
            }

            WHEN("The sizes of the block and DataHandler are different")
            {
                const auto bigDh = dh2.getBlock(0, 2 * size);
                THEN("Assignment throws") { REQUIRE_THROWS(*dhMap = *bigDh); }
            }
        }

        WHEN("Copy-assigning a full DataHandlerMap (aka a view) through the base pointers")
        {
            const auto dhCopy = dh;
            Vector_t<data_t> randVec{size};
            randVec.setRandom();

            const TestType dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            WHEN("The sizes are the same")
            {
                *dhMap = *dh2Map;
                THEN("a deep copy is performed")
                {
                    REQUIRE_EQ(useCount(dh), 1);
                    REQUIRE_EQ(useCount(dhCopy), 1);

                    for (index_t i = 0; i < size; i++)
                        REQUIRE_UNARY(checkApproxEq(dh[i], dh2[i]));

                    REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                }
            }

            WHEN("The sizes are different")
            {
                const std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<TestType>(2 * size);
                THEN("sizes must match") { REQUIRE_THROWS(*dhMap = *bigDh->getBlock(0, 2 * size)); }
            }
        }

        WHEN("\"move\" assigning a DataHandlerMap through the base pointer")
        {
            Vector_t<data_t> randVec{size};
            randVec.setRandom();
            const std::unique_ptr<DataHandler<data_t>> dh2Ptr = std::make_unique<TestType>(randVec);

            WHEN("The sizes are the same")
            {
                const auto dhCopy = dh;

                *dhMap = std::move(*dh2Ptr);
                THEN("a deep copy is performed")
                {
                    REQUIRE_EQ(useCount(dh), 1);
                    REQUIRE_EQ(useCount(dhCopy), 1);

                    for (index_t i = 0; i < size; i++)
                        REQUIRE_UNARY(checkApproxEq(dh[i], (*dh2Ptr)[i]));

                    REQUIRE_EQ(*dhMap, *dh2Ptr);
                    REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                }
            }

            WHEN("The sizes are different")
            {
                const std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<TestType>(2 * size);
                THEN("the assignment throws") { REQUIRE_THROWS(*dhMap = std::move(*bigDh)); }
            }
        }

        WHEN("\"move\" assigning a block of a  DataHandlerMap through the base pointer")
        {
            const auto dhCopy = dh;
            Vector_t<data_t> randVec{2 * size};
            randVec.setRandom();
            TestType dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            WHEN("The sizes are the same")
            {
                *dhMap = std::move(*dh2Map);
                THEN("a deep copy is performed")
                {
                    REQUIRE_EQ(useCount(dh), 1);
                    REQUIRE_EQ(useCount(dhCopy), 1);

                    for (index_t i = 0; i < size; i++)
                        REQUIRE_UNARY(checkApproxEq(dh[i], dh2[i]));

                    REQUIRE_EQ(*dhMap, *dh2Map);
                    REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                }
            }

            WHEN("The sizes are different")
            {
                const auto bigDh = dh2.getBlock(0, 2 * size);
                THEN("the assignment throws") { REQUIRE_THROWS(*dhMap = std::move(*bigDh)); }
            }
        }

        WHEN("\"move\" assigning a full DataHandlerMap (aka a view) through the base pointer")
        {
            const auto dhCopy = dh;
            Vector_t<data_t> randVec{size};
            randVec.setRandom();
            TestType dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            WHEN("The sizes are the same")
            {
                *dhMap = std::move(*dh2Map);
                THEN("a deep copy is performed")
                {
                    REQUIRE_EQ(useCount(dh), 1);
                    REQUIRE_EQ(useCount(dhCopy), 1);

                    for (index_t i = 0; i < size; i++)
                        REQUIRE_UNARY(checkApproxEq(dh[i], dh2[i]));

                    REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                }
            }

            WHEN("The sizes are different")
            {
                const std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<TestType>(2 * size);
                THEN("the assignment throws") { REQUIRE_THROWS(*dhMap = std::move(*bigDh)); }
            }
        }
    }

    GIVEN("A reference to a full concrete DataHandlerMap (aka a view)")
    {
        index_t size = 314;
        TestType dh{size};
        const auto dhMap = dh.getBlock(0, size);
        auto& dhMapRef = static_cast<MapType&>(*dhMap);

        WHEN("Copy-assigning an equally sized view to the map")
        {
            const TestType dh2{size};
            const auto dh2Map = dh2.getBlock(0, size);
            const auto& dh2MapRef = static_cast<const MapType&>(*dh2Map);
            dhMapRef = dh2MapRef;

            THEN("a shallow copy is performed")
            {
                REQUIRE_EQ(useCount(dh), 2);
                REQUIRE_EQ(dh, dh2);
                REQUIRE_EQ(dh, dhMapRef);
                REQUIRE_EQ(dh, dh2MapRef);
                dhMapRef[0] = 1;
                REQUIRE_EQ(&dhMapRef[0], &dh[0]);
                REQUIRE_NE(&dhMapRef[0], &dh2MapRef[0]);
            }
        }
        WHEN("Copy-assigning an differently sized view to the map")
        {
            const TestType dh2{3 * size};
            const auto dh2Map = dh2.getBlock(0, 3 * size);
            const auto& dh2MapRef = static_cast<const MapType&>(*dh2Map);

            THEN("The assigning throws") { REQUIRE_THROWS(dhMapRef = dh2MapRef); }
        }
    }

    GIVEN("a full DataHandlerMap (aka a view)")
    {
        index_t size = 314;
        TestType dh{size};
        const auto dhMap = dh.getBlock(0, size);

        WHEN("copy assigning a DataHandlerMap through base pointers")
        {
            Vector_t<data_t> randVec{size};
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
                REQUIRE_EQ(useCount(dh), 2);
                REQUIRE_EQ(dh, *dh2Ptr);
                REQUIRE_EQ(*dhMap, *dh2Ptr);
                REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                dh[0] = 1;
                REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                REQUIRE_NE(&dh[0], &(*dh2Ptr)[0]);
            }
        }

        WHEN("copy assigning a partial DataHandlerMap through base pointers")
        {
            const auto dhCopy = dh;
            Vector_t<data_t> randVec{2 * size};
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
                REQUIRE_EQ(useCount(dh), 1);
                REQUIRE_EQ(useCount(dhCopy), 1);

                for (index_t i = 0; i < size; i++)
                    REQUIRE_UNARY(checkApproxEq(dh[i], dh2[i]));

                REQUIRE_EQ(*dhMap, *dh2Map);
                REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
            }
        }

        WHEN("copy assigning a full DataHandlerMap (aka a view) through base pointers")
        {
            Vector_t<data_t> randVec{size};
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
                REQUIRE_EQ(useCount(dh), 2);
                REQUIRE_EQ(dh, dh2);
                REQUIRE_EQ(*dhMap, *dh2Map);
                REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                dh[0] = 1;
                REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                REQUIRE_NE(&dh[0], &dh2[0]);
            }
        }

        WHEN("\"move\" assigning a DataHandler through base pointers")
        {
            Vector_t<data_t> randVec{size};
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
                REQUIRE_EQ(useCount(dh), 2);
                REQUIRE_EQ(dh, *dh2Ptr);
                REQUIRE_EQ(*dhMap, *dh2Ptr);
                REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                dh[0] = 1;
                REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                REQUIRE_NE(&dh[0], &(*dh2Ptr)[0]);
            }
        }

        WHEN("\"move\" assigning a partial DataHandlerMap through base pointers")
        {
            const auto dhCopy = dh;
            Vector_t<data_t> randVec{2 * size};
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
                REQUIRE_EQ(useCount(dh), 1);
                REQUIRE_EQ(useCount(dhCopy), 1);

                for (index_t i = 0; i < size; i++)
                    REQUIRE_UNARY(checkApproxEq(dh[i], dh2[i]));

                REQUIRE_EQ(*dhMap, *dh2Map);
                REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
            }
        }

        WHEN("\"move\" assigning a full DataHandlerMap (aka a view) through base pointers")
        {
            Vector_t<data_t> randVec{size};
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
                REQUIRE_EQ(useCount(dh), 2);
                REQUIRE_EQ(dh, dh2);
                REQUIRE_EQ(*dhMap, *dh2Map);
                REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                dh[0] = 1;
                REQUIRE_EQ(&(*dhMap)[0], &dh[0]);
                REQUIRE_NE(&dh[0], &dh2[0]);
            }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataHandlerMap: Testing clone()", TestType, datahandlermap_clone)
{
    using data_t = typename TestType::value_type;

    GIVEN("a full DataHandlerMap (aka a view)")
    {
        index_t size = 728;
        Vector_t<data_t> dataVec(size);
        dataVec.setRandom();
        TestType realDh(dataVec);
        auto dhPtr = realDh.getBlock(0, size);
        auto& dh = *dhPtr;

        WHEN("cloning")
        {
            auto dhClone = dh.clone();

            THEN("a shallow copy is produced")
            {
                REQUIRE_NE(dhClone.get(), &dh);

                REQUIRE_EQ(dhClone->getSize(), dh.getSize());

                REQUIRE_EQ(useCount(realDh), 2);

                REQUIRE_EQ(*dhClone, dh);

                dh[0] = 1;
                REQUIRE_NE(*dhClone, dh);
            }
        }
    }

    GIVEN("a partial DataHandlerMap")
    {
        index_t size = 728;
        Vector_t<data_t> dataVec(size);
        dataVec.setRandom();
        TestType realDh(dataVec);
        auto dhPtr = realDh.getBlock(0, size / 2);
        auto& dh = *dhPtr;

        WHEN("a deep copy is produced")
        {
            auto dhClone = dh.clone();

            THEN("everything matches")
            {
                REQUIRE_NE(dhClone.get(), &dh);

                REQUIRE_EQ(dhClone->getSize(), dh.getSize());

                REQUIRE_EQ(useCount(realDh), 1);

                REQUIRE_EQ(dh, *dhClone);
            }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataHandlerMap: Testing the reduction operations", TestType,
                          datahandlermap_reduction)
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
                REQUIRE_UNARY(checkApproxEq(dh.sum(), randVec.middleRows(size / 3, size).sum()));
                REQUIRE_EQ(dh.l0PseudoNorm(),
                           (randVec.middleRows(size / 3, size).array().cwiseAbs()
                            >= std::numeric_limits<GetFloatingPointType_t<data_t>>::epsilon())
                               .count());
                REQUIRE_UNARY(checkApproxEq(
                    dh.l1Norm(), randVec.middleRows(size / 3, size).array().abs().sum()));
                REQUIRE_UNARY(checkApproxEq(
                    dh.lInfNorm(), randVec.middleRows(size / 3, size).array().abs().maxCoeff()));
                REQUIRE_UNARY(checkApproxEq(dh.squaredL2Norm(),
                                            randVec.middleRows(size / 3, size).squaredNorm()));
                REQUIRE_UNARY(
                    checkApproxEq(dh.l2Norm(), randVec.middleRows(size / 3, size).norm()));

                auto randVec2 = generateRandomMatrix<data_t>(size);
                TestType realDh2(randVec2);
                auto dh2Ptr = realDh2.getBlock(0, size);
                auto& dh2 = *dh2Ptr;
                REQUIRE_UNARY(
                    checkApproxEq(dh.dot(dh2), randVec.middleRows(size / 3, size).dot(randVec2)));

                TestType dhCPU(randVec2);
                REQUIRE_UNARY(
                    checkApproxEq(dh.dot(dhCPU), randVec.middleRows(size / 3, size).dot(randVec2)));
            }

            THEN("the dot product expects correctly sized arguments")
            {
                index_t wrongSize = size - 1;
                Vector_t<data_t> randVec2(wrongSize);
                randVec2.setRandom();
                TestType dh2(randVec2);

                REQUIRE_THROWS_AS(dh.dot(dh2), InvalidArgumentError);
            }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataHandlerMap: Testing the element-wise operations", TestType,
                          datahandlermap_elemwise)
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
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] + dh2[i]));

                dh = oldDh;
                dh += dhCPU;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] + dhCPU[i]));

                dh = oldDh;
                dh -= dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] - dh2[i]));

                dh = oldDh;
                dh -= dhCPU;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] - dhCPU[i]));

                dh = oldDh;
                dh *= dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] * dh2[i]));

                dh = oldDh;
                dh *= dhCPU;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] * dhCPU[i]));

                dh = oldDh;
                dh /= dh2;
                for (index_t i = 0; i < size; ++i)
                    if (dh2[i] != data_t(0))
                        // due to floating point arithmetic less precision
                        REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] / dh2[i]));

                dh = oldDh;
                dh /= dhCPU;
                for (index_t i = 0; i < size; ++i)
                    if (dhCPU[i] != data_t(0))
                        // due to floating point arithmetic less precision
                        REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] / dhCPU[i]));
            }

            THEN("the element-wise binary scalar operations work as expected")
            {
                TestType realOldDh(randVec);
                auto oldDhPtr = realOldDh.getBlock(0, size);
                auto& oldDh = static_cast<typename MapToHandler<TestType>::map&>(*oldDhPtr);
                data_t scalar = std::is_integral_v<data_t> ? 3 : data_t(3.5f);

                dh += scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] + scalar));

                dh = oldDh;
                dh -= scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] - scalar));

                dh = oldDh;
                dh *= scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] * scalar));

                dh = oldDh;
                dh /= scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] / scalar));
            }

            THEN("the element-wise assignment of a scalar works as expected")
            {
                auto scalar = std::is_integral_v<data_t> ? data_t(47) : data_t(47.11f);

                dh = scalar;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], scalar));
            }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataHandlerMap: Testing referencing blocks", TestType,
                          datahandlermap_blockref)
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandlerMap")
    {
        index_t size = 728;
        Vector_t<data_t> dataVec(size);
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
                REQUIRE_EQ(dhBlock->getSize(), size / 2);

                for (index_t i = 0; i < size / 2; i++)
                    REQUIRE_EQ(&(*dhBlock)[i], &dh[i + size / 3]);
            }
        }
    }

    GIVEN("a const DataHandlerMap")
    {
        index_t size = 728;
        Vector_t<data_t> dataVec(size);
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
                REQUIRE_EQ(dhBlock->getSize(), size / 2);

                for (index_t i = 0; i < size / 2; i++)
                    REQUIRE_EQ(&(*dhBlock)[i], &dh[i + size / 3]);
            }
        }
    }
}

// "instantiate" the test templates for CPU types
TEST_CASE_TEMPLATE_APPLY(datahandlermap_construction, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandlermap_eqoperator, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandlermap_assign, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandlermap_clone, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandlermap_reduction, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandlermap_elemwise, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandlermap_blockref, CPUTypeTuple);

#ifdef ELSA_CUDA_VECTOR
// "instantiate" the test templates for GPU types
TEST_CASE_TEMPLATE_APPLY(datahandlermap_construction, GPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandlermap_eqoperator, GPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandlermap_assign, GPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandlermap_clone, GPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandlermap_reduction, GPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandlermap_elemwise, GPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandlermap_blockref, GPUTypeTuple);
#endif

TEST_SUITE_END();
