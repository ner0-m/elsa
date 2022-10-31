/**
 * @file test_DataHandlers.cpp
 *
 * @brief Common tests for DataHandlers class
 *
 * @author David Frank - initial code
 * @author Tobias Lasser - rewrite and code coverage
 * @author Jens Petit - refactoring to general DataHandler test
 */

#include "doctest/doctest.h"
#include "DataHandlerCPU.h"
#include "DataHandlerMapCPU.h"
#include "testHelpers.h"

#ifdef ELSA_CUDA_VECTOR
#include "DataHandlerGPU.h"
#include "DataHandlerMapGPU.h"
#endif

template <typename data_t>
long elsa::useCount(const DataHandlerCPU<data_t>& dh)
{
    return dh._data.use_count();
}

#ifdef ELSA_CUDA_VECTOR
template <typename data_t>
long elsa::useCount(const DataHandlerGPU<data_t>& dh)
{
    return dh._data.use_count();
}
#endif

using namespace elsa;
using namespace doctest;

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

TEST_CASE_TEMPLATE_DEFINE("DataHandlers: Testing Construction", TestType, datahandler_construction)
{
    using data_t = typename TestType::value_type;

    GIVEN("a certain size")
    {
        index_t size = 314;

        WHEN("constructing")
        {
            const TestType dh{size};

            THEN("it has the correct size")
            {
                REQUIRE_EQ(size, dh.getSize());
            }
        }

        WHEN("constructing with a given vector")
        {
            auto randVec = generateRandomMatrix<data_t>(size);
            const TestType dh{randVec};

            for (index_t i = 0; i < size; ++i)
                REQUIRE_UNARY(checkApproxEq(dh[i], randVec(i)));
        }

        WHEN("copy constructing")
        {
            auto randVec = generateRandomMatrix<data_t>(size);
            const TestType dh{randVec};
            const auto dhView = dh.getBlock(0, size);

            TestType dh2 = dh;

            THEN("a shallow copy is created")
            {
                REQUIRE_EQ(dh2, dh);
                REQUIRE_EQ(useCount(dh), 2);

                const auto dh2View = dh2.getBlock(0, size);
                AND_THEN("associated maps are not transferred")
                {
                    dh2[0] = data_t(1);
                    REQUIRE_UNARY(checkApproxEq(dh2[0], 1));
                    REQUIRE_UNARY(checkApproxEq((*dhView)[0], randVec[0]));
                    REQUIRE_UNARY(checkApproxEq((*dh2View)[0], 1));
                }
            }
        }

        WHEN("move constructing")
        {
            auto randVec = generateRandomMatrix<data_t>(size);
            TestType dh{randVec};
            const auto dhView = dh.getBlock(0, size);
            TestType testDh{randVec};

            const TestType dh2 = std::move(dh);

            THEN("data and associated maps are moved to the new handler")
            {
                REQUIRE_EQ(useCount(dh2), 1);
                REQUIRE_EQ(dh2, testDh);
                REQUIRE_EQ(&(*dhView)[0], &dh2[0]);
            }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataHandlers: Testing equality operator", TestType, datahandler_equality)
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandler")
    {
        const index_t size = 314;
        auto randVec = generateRandomMatrix<data_t>(size);
        const TestType dh{randVec};

        WHEN("comparing to a handler with a different size")
        {
            const TestType dh2{size + 1};
            THEN("the result is false")
            {
                REQUIRE_NE(dh, dh2);
                REQUIRE_NE(dh, *dh2.getBlock(0, size + 1));
            }
        }

        WHEN("comparing to a shallow copy or view of the handler")
        {
            const auto dh2 = dh;
            THEN("the result is true")
            {
                REQUIRE_EQ(dh, dh2);
                REQUIRE_EQ(dh, *dh.getBlock(0, size));
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
                REQUIRE_NE(dh, dh2);
                REQUIRE_NE(dh, *dh2.getBlock(0, size));
            }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataHandlers: Assigning to DataHandlerCPU", TestType,
                          datahandler_assigncpu)
{
    using data_t = typename TestType::value_type;

    GIVEN("a DataHandlerCPU with an associated map")
    {
        const index_t size = 314;
        DataHandlerCPU<data_t> dh{size};
        auto dhMap = dh.getBlock(size / 2, size / 3);

        WHEN("copy assigning")
        {

            auto randVec = generateRandomMatrix<data_t>(size);
            const DataHandlerCPU dh2{randVec};
            const auto dh2Map = dh2.getBlock(size / 2, size / 3);

            THEN("sizes must match")
            {
                const DataHandlerCPU<data_t> bigDh{2 * size};
                REQUIRE_THROWS(dh = bigDh);
            }

            dh = dh2;
            THEN("a shallow copy is performed and associated Maps are updated")
            {
                REQUIRE_EQ(useCount(dh), 2);
                REQUIRE_EQ(dh, dh2);
                REQUIRE_EQ(*dhMap, *dh2Map);
            }
        }

        WHEN("move assigning")
        {
            auto randVec = generateRandomMatrix<data_t>(size);
            DataHandlerCPU dh2{randVec};
            const auto dh2View = dh2.getBlock(0, size);
            DataHandlerCPU testDh{randVec};

            THEN("sizes must match")
            {
                DataHandlerCPU<data_t> bigDh{2 * size};
                REQUIRE_THROWS(dh = std::move(bigDh));
            }

            dh = std::move(dh2);
            THEN("data is moved, associated maps are merged")
            {
                REQUIRE_EQ(useCount(dh), 1);
                REQUIRE_EQ(dh, testDh);
                REQUIRE_EQ(&(*dhMap)[0], &dh[size / 2]);
                REQUIRE_EQ(dhMap->getSize(), size / 3);
                REQUIRE_EQ(&(*dh2View)[0], &dh[0]);
                REQUIRE_EQ(dh2View->getSize(), size);
            }
        }

        WHEN("copy assigning a DataHandlerCPU through base pointers")
        {
            DataHandler<data_t>* dhPtr = &dh;

            auto randVec = generateRandomMatrix<data_t>(size);
            const auto dh2Ptr = std::make_unique<const DataHandlerCPU<data_t>>(randVec);
            const auto dh2Map = dh2Ptr->getBlock(size / 2, size / 3);

            THEN("sizes must match")
            {
                std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<DataHandlerCPU<data_t>>(2 * size);
                REQUIRE_THROWS(*dhPtr = *bigDh);
            }

            *dhPtr = *dh2Ptr;
            THEN("a shallow copy is performed and associated Maps are updated")
            {
                REQUIRE_EQ(useCount(dh), 2);
                REQUIRE_EQ(dh, *dh2Ptr);
                REQUIRE_EQ(*dhMap, *dh2Map);
                dh[0] = 1;
                REQUIRE_NE(&dh[0], &(*dh2Ptr)[0]);
                REQUIRE_EQ(*dhMap, *dh2Map);
                REQUIRE_EQ(&(*dhMap)[0], &dh[size / 2]);
            }
        }

        WHEN("copy assigning a partial DataHandlerMapCPU through base pointers")
        {
            DataHandler<data_t>* dhPtr = &dh;
            const auto dhCopy = dh;

            auto randVec = generateRandomMatrix<data_t>(2 * size);
            const DataHandlerCPU<data_t> dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                const auto bigDh = dh2.getBlock(0, size + 1);
                REQUIRE_THROWS(*dhPtr = *bigDh);
            }

            *dhPtr = *dh2Map;
            THEN("a deep copy is performed")
            {
                REQUIRE_EQ(useCount(dh), 1);
                REQUIRE_EQ(useCount(dhCopy), 1);
                REQUIRE_EQ(dh, *dh2Map);
                REQUIRE_EQ(&(*dhMap)[0], &dh[size / 2]);
                REQUIRE_EQ(dhMap->getSize(), size / 3);
            }
        }

        WHEN("copy assigning a full DataHandlerMapCPU (aka a view) through base pointers")
        {
            DataHandler<data_t>* dhPtr = &dh;

            auto randVec = generateRandomMatrix<data_t>(size);
            const DataHandlerCPU<data_t> dh2{randVec};
            const auto dh2View = dh2.getBlock(0, size);
            const auto dh2Map = dh2.getBlock(size / 2, size / 3);

            THEN("sizes must match")
            {
                std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<DataHandlerCPU<data_t>>(2 * size);
                auto bigDhView = bigDh->getBlock(0, 2 * size);
                REQUIRE_THROWS(*dhPtr = *bigDhView);
            }

            *dhPtr = *dh2View;
            THEN("a shallow copy is performed and associated maps are updated")
            {
                REQUIRE_EQ(useCount(dh), 2);
                REQUIRE_EQ(dh, *dh2View);
                REQUIRE_EQ(*dhMap, *dh2Map);
                dh[0] = 1;
                REQUIRE_NE(&dh[0], &(*dh2View)[0]);
                REQUIRE_EQ(*dhMap, *dh2Map);
                REQUIRE_EQ(&(*dhMap)[0], &dh[size / 2]);
            }
        }

        WHEN("move assigning a DataHandlerCPU through base pointers")
        {
            DataHandler<data_t>* dhPtr = &dh;

            auto randVec = generateRandomMatrix<data_t>(size);
            std::unique_ptr<DataHandler<data_t>> dh2Ptr =
                std::make_unique<DataHandlerCPU<data_t>>(randVec);
            const auto dh2View = dh2Ptr->getBlock(0, size);
            DataHandlerCPU<data_t> testDh{randVec};

            THEN("sizes must match")
            {
                std::unique_ptr<DataHandler<data_t>> bigDh =
                    std::make_unique<DataHandlerCPU<data_t>>(2 * size);
                REQUIRE_THROWS(*dhPtr = std::move(*bigDh));
            }

            *dhPtr = std::move(*dh2Ptr);
            THEN("data is moved and associated Maps are updated")
            {
                REQUIRE_EQ(useCount(dh), 1);
                REQUIRE_EQ(dh, testDh);
                REQUIRE_EQ(&(*dhMap)[0], &dh[size / 2]);
                REQUIRE_EQ(dhMap->getSize(), size / 3);
                REQUIRE_EQ(&(*dh2View)[0], &dh[0]);
                REQUIRE_EQ(dh2View->getSize(), size);
            }
        }

        WHEN("\"move\" assigning a partial DataHandlerMapCPU through base pointers")
        {
            DataHandler<data_t>* dhPtr = &dh;
            const auto dhCopy = dh;

            auto randVec = generateRandomMatrix<data_t>(2 * size);
            DataHandlerCPU<data_t> dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                REQUIRE_THROWS(*dhPtr = std::move(*dh2.getBlock(0, 2 * size)));
            }

            *dhPtr = std::move(*dh2Map);
            THEN("a deep copy is performed")
            {
                REQUIRE_EQ(useCount(dh), 1);
                REQUIRE_EQ(useCount(dhCopy), 1);
                REQUIRE_EQ(dh, *dh2.getBlock(0, size));
                REQUIRE_EQ(&(*dhMap)[0], &dh[size / 2]);
                REQUIRE_EQ(dhMap->getSize(), size / 3);
            }
        }

        WHEN("\"move\" assigning a full DataHandlerMapCPU (aka a view) through base pointers")
        {
            DataHandler<data_t>* dhPtr = &dh;

            auto randVec = generateRandomMatrix<data_t>(size);
            DataHandlerCPU<data_t> dh2{randVec};
            const auto dh2View = dh2.getBlock(0, size);
            const auto dh2Map = dh2.getBlock(size / 2, size / 3);

            THEN("sizes must match")
            {
                const std::unique_ptr<const DataHandler<data_t>> bigDh =
                    std::make_unique<const DataHandlerCPU<data_t>>(2 * size);
                REQUIRE_THROWS(*dhPtr = std::move(*bigDh->getBlock(0, 2 * size)));
            }

            *dhPtr = std::move(*dh2View);
            THEN("a shallow copy is performed and associated maps are updated")
            {
                REQUIRE_EQ(useCount(dh), 2);
                REQUIRE_EQ(dh, *dh2View);
                REQUIRE_EQ(*dhMap, *dh2Map);
                dh[0] = 1;
                REQUIRE_NE(&dh[0], &dh2[0]);
                REQUIRE_EQ(*dhMap, *dh2Map);
                REQUIRE_EQ(&(*dhMap)[0], &dh[size / 2]);
            }
        }
    }
}

#ifdef ELSA_CUDA_VECTOR
TEST_CASE_TEMPLATE("DataHandlers: Testing clone()", TestType, DataHandlerCPU<float>,
                   DataHandlerGPU<float>)
#else
TEST_CASE_TEMPLATE("DataHandlers: Testing clone()", TestType, DataHandlerCPU<float>)
#endif
{
    GIVEN("some DataHandler")
    {
        index_t size = 728;
        TestType dh(size);
        dh = 1.0f;

        WHEN("cloning")
        {
            auto dhClone = dh.clone();

            THEN("a shallow copy is produced")
            {
                REQUIRE_NE(dhClone.get(), &dh);

                REQUIRE_EQ(useCount(dh), 2);
                REQUIRE_EQ(*dhClone, dh);

                REQUIRE_EQ(dhClone->getSize(), dh.getSize());

                dh[0] = 2.f;
                REQUIRE_NE(dh, *dhClone);
            }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataHandlers: Testing the reduction operations", TestType,
                          datahandler_reduction)
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandler")
    {
        index_t size = 16;

        WHEN("putting in some random data")
        {
            auto randVec = generateRandomMatrix<data_t>(size);
            TestType dh(randVec);

            THEN("the reductions work as expected")
            {
                auto eps = std::numeric_limits<GetFloatingPointType_t<data_t>>::epsilon();
                REQUIRE_UNARY(checkApproxEq(dh.sum(), randVec.sum()));
                REQUIRE_UNARY(
                    checkApproxEq(dh.l0PseudoNorm(), (randVec.array().cwiseAbs() >= eps).count()));
                REQUIRE_UNARY(checkApproxEq(dh.l1Norm(), randVec.array().abs().sum()));
                REQUIRE_UNARY(checkApproxEq(dh.lInfNorm(), randVec.array().abs().maxCoeff()));
                REQUIRE_UNARY(checkApproxEq(dh.squaredL2Norm(), randVec.squaredNorm()));
                REQUIRE_UNARY(checkApproxEq(dh.l2Norm(), randVec.norm()));

                auto randVec2 = generateRandomMatrix<data_t>(size);
                TestType dh2(randVec2);

                REQUIRE_UNARY(checkApproxEq(dh.dot(dh2), randVec.dot(randVec2)));

                auto dhMap = dh2.getBlock(0, dh2.getSize());

                REQUIRE_UNARY(checkApproxEq(dh.dot(*dhMap), randVec.dot(randVec2)));
            }

            THEN("the dot product expects correctly sized arguments")
            {
                index_t wrongSize = size - 1;

                auto randVec2 = generateRandomMatrix<data_t>(wrongSize);
                TestType dh2(randVec2);

                REQUIRE_THROWS_AS(dh.dot(dh2), InvalidArgumentError);
            }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataHandlers: Testing the element-wise operations", TestType,
                          datahandler_elementwise)
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandler")
    {
        index_t size = 567;

        WHEN("putting in some random data")
        {
            auto randVec = generateRandomMatrix<data_t>(size);
            TestType dh(randVec);

            THEN("the element-wise binary vector operations work as expected")
            {
                TestType oldDh = dh;

                auto randVec2 = generateRandomMatrix<data_t>(size);
                TestType dh2(randVec2);

                auto dhMap = dh2.getBlock(0, dh2.getSize());

                TestType bigDh{size + 1};
                REQUIRE_THROWS(dh += bigDh);
                REQUIRE_THROWS(dh -= bigDh);
                REQUIRE_THROWS(dh *= bigDh);
                REQUIRE_THROWS(dh /= bigDh);

                dh += dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] + dh2[i]));

                dh = oldDh;
                dh += *dhMap;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] + dh2[i]));

                dh = oldDh;
                dh -= dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] - dh2[i]));

                dh = oldDh;
                dh -= *dhMap;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] - dh2[i]));

                dh = oldDh;
                dh *= dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] * dh2[i]));

                dh = oldDh;
                dh *= *dhMap;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] * dh2[i]));

                dh = oldDh;
                dh /= dh2;
                for (index_t i = 0; i < size; ++i)
                    if (dh2[i] != data_t(0))
                        // due to floating point arithmetic less precision
                        REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] / dh2[i]));

                dh = oldDh;
                dh /= *dhMap;
                for (index_t i = 0; i < size; ++i)
                    if (dh2[i] != data_t(0))
                        // due to floating point arithmetic less precision
                        REQUIRE_UNARY(checkApproxEq(dh[i], oldDh[i] / dh2[i]));
            }

            THEN("the element-wise binary scalar operations work as expected")
            {
                TestType oldDh = dh;
                data_t scalar = std::is_integral_v<data_t> ? 3 : 3.5;

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

TEST_CASE_TEMPLATE_DEFINE("DataHandlers: Testing referencing blocks", TestType,
                          datahandler_blockreferencing)
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandler")
    {
        index_t size = 728;
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> dataVec(size);
        TestType dh(dataVec);

        WHEN("getting the reference to a block")
        {
            REQUIRE_THROWS(dh.getBlock(size, 1));
            REQUIRE_THROWS(dh.getBlock(0, size + 1));

            auto dhBlock = dh.getBlock(size / 3, size / 2);

            THEN("returned data handler references the correct elements")
            {
                REQUIRE_EQ(dhBlock->getSize(), size / 2);

                for (index_t i = 0; i < size / 2; i++)
                    REQUIRE_UNARY(checkApproxEq(&(*dhBlock)[i], &dh[i + size / 3]));
            }
        }

        WHEN("the whole volume is referenced")
        {
            auto dhBlock = dh.getBlock(0, size);

            THEN("the referenced volume and the actual volume are equal")
            {

                REQUIRE_EQ(dh, *dhBlock);
            }
        }
    }
}

TEST_CASE_TEMPLATE_DEFINE("DataHandlers: Testing the copy-on-write mechanism", TestType,
                          datahandler_copyonwrite)
{
    using data_t = typename TestType::value_type;

    const index_t size = 42;

    GIVEN("A random DataContainer")
    {
        auto randVec = generateRandomMatrix<data_t>(size);
        TestType dh{randVec};

        WHEN("const manipulating a copy constructed shallow copy")
        {
            TestType dh2 = dh;

            THEN("the data is the same")
            {
                REQUIRE_EQ(dh, dh2);
                REQUIRE_EQ(useCount(dh), 2);
            }
        }

        WHEN("non-const manipulating a copy constructed shallow copy")
        {
            TestType dh2 = dh;
            REQUIRE_EQ(useCount(dh), 2);
            REQUIRE_EQ(useCount(dh2), 2);

            THEN("copy-on-write is invoked")
            {
                dh2 += 2;
                REQUIRE_NE(dh2, dh);
                REQUIRE_EQ(useCount(dh2), 1);
                REQUIRE_EQ(useCount(dh), 1);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 += dh;
                REQUIRE_NE(dh2, dh);
                REQUIRE_EQ(useCount(dh2), 1);
                REQUIRE_EQ(useCount(dh), 1);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 -= 2;
                REQUIRE_NE(dh2, dh);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 -= dh;
                REQUIRE_NE(dh2, dh);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 /= 2;
                REQUIRE_NE(dh2, dh);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 /= dh;
                REQUIRE_NE(dh2, dh);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 *= 2;
                REQUIRE_NE(dh2, dh);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 *= dh;
                REQUIRE_NE(dh2, dh);
            }

            THEN("copy-on-write is invoked")
            {
                dh[0] += 2;
                REQUIRE_NE(dh2, dh);
            }
        }

        WHEN("manipulating a non-shallow-copied container")
        {
            for (index_t i = 0; i < dh.getSize(); ++i) {
                dh[i] += 2;
            }

            THEN("copy-on-write should not be invoked")
            {
                REQUIRE_EQ(useCount(dh), 1);
            }
        }
    }
}

// "instantiate" the test templates for cpu types
TEST_CASE_TEMPLATE_APPLY(datahandler_construction, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandler_equality, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandler_assigncpu, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandler_reduction, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandler_elementwise, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandler_blockreferencing, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandler_copyonwrite, CPUTypeTuple);

#ifdef ELSA_CUDA_VECTOR
// "instantiate" the test templates for GPU types
TEST_CASE_TEMPLATE_APPLY(datahandler_construction, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandler_equality, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandler_assigncpu, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandler_reduction, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandler_elementwise, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandler_blockreferencing, CPUTypeTuple);
TEST_CASE_TEMPLATE_APPLY(datahandler_copyonwrite, CPUTypeTuple);
#endif

TEST_SUITE_END();
