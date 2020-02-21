/**
 * \file test_DataHandlers.cpp
 *
 * \brief Common tests for DataHandlers class
 *
 * \author David Frank - initial code
 * \author Tobias Lasser - rewrite and code coverage
 * \author Jens Petit - refactoring to general DataHandler test
 */

#include <catch2/catch.hpp>
#include "DataHandlerCPU.h"
#include "DataHandlerMapCPU.h"
#include "testHelpers.h"

template <typename data_t>
long elsa::useCount(const DataHandlerCPU<data_t>& dh)
{
    return dh._data.use_count();
}

using namespace elsa;

TEMPLATE_PRODUCT_TEST_CASE("Scenario: Constructing DataHandler", "", (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
{
    using data_t = typename TestType::value_type;

    GIVEN("a certain size")
    {
        index_t size = 314;

        WHEN("constructing with zeros")
        {
            const TestType dh{size};

            THEN("it has the correct size") { REQUIRE(size == dh.getSize()); }

            THEN("it has a zero vector")
            {
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == static_cast<data_t>(0));
            }
        }

        WHEN("constructing with a given vector")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
            const TestType dh{randVec};

            for (index_t i = 0; i < size; ++i)
                REQUIRE(dh[i] == randVec(i));
        }

        WHEN("copy constructing")
        {
            const TestType dh{size};
            const auto dhView = dh.getBlock(0, size);

            TestType dh2 = dh;

            THEN("a shallow copy is created")
            {
                REQUIRE(dh2 == dh);
                REQUIRE(useCount(dh) == 2);

                const auto dh2View = dh2.getBlock(0, size);
                AND_THEN("associated maps are not transferred")
                {
                    dh2[0] = data_t(1);
                    REQUIRE(dh2[0] == data_t(1));
                    REQUIRE((*dhView)[0] == data_t(0));
                    REQUIRE((*dh2View)[0] == data_t(1));
                }
            }
        }

        WHEN("move constructing")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
            TestType dh{randVec};
            const auto dhView = dh.getBlock(0, size);
            TestType testDh{randVec};

            const TestType dh2 = std::move(dh);

            THEN("data and associated maps are moved to the new handler")
            {
                REQUIRE(useCount(dh2) == 1);
                REQUIRE(dh2 == testDh);
                REQUIRE(&(*dhView)[0] == &dh2[0]);
            }
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing equality operator on DataHandler", "",
                           (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandler")
    {
        index_t size = 314;
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
        randVec.setRandom();
        const TestType dh{randVec};

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
            const auto dh2 = dh;
            THEN("the result is true")
            {
                REQUIRE(dh == dh2);
                REQUIRE(dh == *dh.getBlock(0, size));
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

TEMPLATE_PRODUCT_TEST_CASE("Scenario: Assigning to DataHandlerCPU", "", (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
{
    using data_t = typename TestType::value_type;

    GIVEN("a DataHandlerCPU with an associated map")
    {
        index_t size = 314;
        DataHandlerCPU<data_t> dh{size};
        auto dhMap = dh.getBlock(size / 2, size / 3);

        WHEN("copy assigning")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
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
                REQUIRE(useCount(dh) == 2);
                REQUIRE(dh == dh2);
                REQUIRE(*dhMap == *dh2Map);
            }
        }

        WHEN("move assigning")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
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
                REQUIRE(useCount(dh) == 1);
                REQUIRE(dh == testDh);
                REQUIRE(&(*dhMap)[0] == &dh[size / 2]);
                REQUIRE(dhMap->getSize() == size / 3);
                REQUIRE(&(*dh2View)[0] == &dh[0]);
                REQUIRE(dh2View->getSize() == size);
            }
        }

        WHEN("copy assigning a DataHandlerCPU through base pointers")
        {
            DataHandler<data_t>* dhPtr = &dh;
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
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
                REQUIRE(useCount(dh) == 2);
                REQUIRE(dh == *dh2Ptr);
                REQUIRE(*dhMap == *dh2Map);
                dh[0] = 1;
                REQUIRE(&dh[0] != &(*dh2Ptr)[0]);
                REQUIRE(*dhMap == *dh2Map);
                REQUIRE(&(*dhMap)[0] == &dh[size / 2]);
            }
        }

        WHEN("copy assigning a partial DataHandlerMapCPU through base pointers")
        {
            DataHandler<data_t>* dhPtr = &dh;
            const auto dhCopy = dh;
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{2 * size};
            randVec.setRandom();
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
                REQUIRE(useCount(dh) == 1);
                REQUIRE(useCount(dhCopy) == 1);
                REQUIRE(dh == *dh2Map);
                REQUIRE(&(*dhMap)[0] == &dh[size / 2]);
                REQUIRE(dhMap->getSize() == size / 3);
            }
        }

        WHEN("copy assigning a full DataHandlerMapCPU (aka a view) through base pointers")
        {
            DataHandler<data_t>* dhPtr = &dh;
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
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
                REQUIRE(useCount(dh) == 2);
                REQUIRE(dh == *dh2View);
                REQUIRE(*dhMap == *dh2Map);
                dh[0] = 1;
                REQUIRE(&dh[0] != &(*dh2View)[0]);
                REQUIRE(*dhMap == *dh2Map);
                REQUIRE(&(*dhMap)[0] == &dh[size / 2]);
            }
        }

        WHEN("move assigning a DataHandlerCPU through base pointers")
        {
            DataHandler<data_t>* dhPtr = &dh;
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
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
                REQUIRE(useCount(dh) == 1);
                REQUIRE(dh == testDh);
                REQUIRE(&(*dhMap)[0] == &dh[size / 2]);
                REQUIRE(dhMap->getSize() == size / 3);
                REQUIRE(&(*dh2View)[0] == &dh[0]);
                REQUIRE(dh2View->getSize() == size);
            }
        }

        WHEN("\"move\" assigning a partial DataHandlerMapCPU through base pointers")
        {
            DataHandler<data_t>* dhPtr = &dh;
            const auto dhCopy = dh;
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{2 * size};
            randVec.setRandom();
            DataHandlerCPU<data_t> dh2{randVec};
            const auto dh2Map = dh2.getBlock(0, size);

            THEN("sizes must match")
            {
                REQUIRE_THROWS(*dhPtr = std::move(*dh2.getBlock(0, 2 * size)));
            }

            *dhPtr = std::move(*dh2Map);
            THEN("a deep copy is performed")
            {
                REQUIRE(useCount(dh) == 1);
                REQUIRE(useCount(dhCopy) == 1);
                REQUIRE(dh == *dh2.getBlock(0, size));
                REQUIRE(&(*dhMap)[0] == &dh[size / 2]);
                REQUIRE(dhMap->getSize() == size / 3);
            }
        }

        WHEN("\"move\" assigning a full DataHandlerMapCPU (aka a view) through base pointers")
        {
            DataHandler<data_t>* dhPtr = &dh;
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
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
                REQUIRE(useCount(dh) == 2);
                REQUIRE(dh == *dh2View);
                REQUIRE(*dhMap == *dh2Map);
                dh[0] = 1;
                REQUIRE(&dh[0] != &dh2[0]);
                REQUIRE(*dhMap == *dh2Map);
                REQUIRE(&(*dhMap)[0] == &dh[size / 2]);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Cloning DataHandler", "", DataHandlerCPU<float>)
{
    GIVEN("some DataHandler")
    {
        index_t size = 728;
        TestType dh(size);

        WHEN("cloning")
        {
            auto dhClone = dh.clone();

            THEN("a shallow copy is produced")
            {
                REQUIRE(dhClone.get() != &dh);

                REQUIRE(useCount(dh) == 2);
                REQUIRE(*dhClone == dh);

                REQUIRE(dhClone->getSize() == dh.getSize());

                dh[0] = 1;
                REQUIRE(dh != *dhClone);
            }
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing the reduction operatios of DataHandler", "",
                           (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandler")
    {
        index_t size = 284;

        WHEN("putting in some random data")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec{size};
            randVec.setRandom();
            TestType dh(randVec);

            THEN("the reductions work as expected")
            {
                REQUIRE(dh.sum() == randVec.sum());
                REQUIRE(dh.l1Norm() == Approx(randVec.array().abs().sum()));
                REQUIRE(dh.lInfNorm() == Approx(randVec.array().abs().maxCoeff()));
                REQUIRE(dh.squaredL2Norm() == Approx(randVec.squaredNorm()));

                Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec2 =
                    Eigen::Matrix<data_t, Eigen::Dynamic, 1>::Random(size);
                TestType dh2(randVec2);

                REQUIRE(checkSameNumbers(dh.dot(dh2), randVec.dot(randVec2)));

                auto dhMap = dh2.getBlock(0, dh2.getSize());

                REQUIRE(checkSameNumbers(dh.dot(*dhMap), randVec.dot(randVec2)));
            }

            THEN("the dot product expects correctly sized arguments")
            {
                index_t wrongSize = size - 1;
                Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec2 =
                    Eigen::Matrix<data_t, Eigen::Dynamic, 1>::Random(wrongSize);
                TestType dh2(randVec2);

                REQUIRE_THROWS_AS(dh.dot(dh2), std::invalid_argument);
            }
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing the element-wise operations of DataHandler", "",
                           (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
{
    using data_t = typename TestType::value_type;

    GIVEN("some DataHandler")
    {
        index_t size = 567;

        WHEN("putting in some random data")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec =
                Eigen::Matrix<data_t, Eigen::Dynamic, 1>::Random(size);
            TestType dh(randVec);

            THEN("the element-wise binary vector operations work as expected")
            {
                TestType oldDh = dh;

                Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec2 =
                    Eigen::Matrix<data_t, Eigen::Dynamic, 1>::Random(size);
                TestType dh2(randVec2);

                auto dhMap = dh2.getBlock(0, dh2.getSize());

                TestType bigDh{size + 1};
                REQUIRE_THROWS(dh += bigDh);
                REQUIRE_THROWS(dh -= bigDh);
                REQUIRE_THROWS(dh *= bigDh);
                REQUIRE_THROWS(dh /= bigDh);

                dh += dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] + dh2[i]);

                dh = oldDh;
                dh += *dhMap;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] + dh2[i]);

                dh = oldDh;
                dh -= dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] - dh2[i]);

                dh = oldDh;
                dh -= *dhMap;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] - dh2[i]);

                dh = oldDh;
                dh *= dh2;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] * dh2[i]);

                dh = oldDh;
                dh *= *dhMap;
                for (index_t i = 0; i < size; ++i)
                    REQUIRE(dh[i] == oldDh[i] * dh2[i]);

                dh = oldDh;
                dh /= dh2;
                for (index_t i = 0; i < size; ++i)
                    if (dh2[i] != data_t(0))
                        REQUIRE(checkSameNumbers(dh[i], oldDh[i] / dh2[i]));

                dh = oldDh;
                dh /= *dhMap;
                for (index_t i = 0; i < size; ++i)
                    if (dh2[i] != data_t(0))
                        REQUIRE(checkSameNumbers(dh[i], oldDh[i] / dh2[i]));
            }

            THEN("the element-wise binary scalar operations work as expected")
            {
                TestType oldDh = dh;
                data_t scalar = std::is_integral_v<data_t> ? 3 : 3.5;

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

TEMPLATE_PRODUCT_TEST_CASE("Scenario: Referencing blocks of DataHandler", "", (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
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
                REQUIRE(dhBlock->getSize() == size / 2);

                for (index_t i = 0; i < size / 2; i++)
                    REQUIRE(&(*dhBlock)[i] == &dh[i + size / 3]);
            }
        }

        WHEN("the whole volume is referenced")
        {
            auto dhBlock = dh.getBlock(0, size);

            THEN("the referenced volume and the actual volume are equal")
            {

                REQUIRE(dh == *dhBlock);
            }
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Scenario: Testing the copy-on-write mechanism", "", (DataHandlerCPU),
                           (float, double, std::complex<float>, std::complex<double>, index_t))
{
    using data_t = typename TestType::value_type;

    GIVEN("A random DataContainer")
    {
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec =
            Eigen::Matrix<data_t, Eigen::Dynamic, 1>::Random(42);
        TestType dh{randVec};

        WHEN("const manipulating a copy constructed shallow copy")
        {
            TestType dh2 = dh;

            THEN("the data is the same")
            {
                REQUIRE(dh == dh2);
                REQUIRE(useCount(dh) == 2);
            }
        }

        WHEN("non-const manipulating a copy constructed shallow copy")
        {
            TestType dh2 = dh;
            REQUIRE(useCount(dh) == 2);
            REQUIRE(useCount(dh2) == 2);

            THEN("copy-on-write is invoked")
            {
                dh2 += 2;
                REQUIRE(dh2 != dh);
                REQUIRE(useCount(dh2) == 1);
                REQUIRE(useCount(dh) == 1);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 += dh;
                REQUIRE(dh2 != dh);
                REQUIRE(useCount(dh2) == 1);
                REQUIRE(useCount(dh) == 1);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 -= 2;
                REQUIRE(dh2 != dh);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 -= dh;
                REQUIRE(dh2 != dh);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 /= 2;
                REQUIRE(dh2 != dh);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 /= dh;
                REQUIRE(dh2 != dh);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 *= 2;
                REQUIRE(dh2 != dh);
            }

            THEN("copy-on-write is invoked")
            {
                dh2 *= dh;
                REQUIRE(dh2 != dh);
            }

            THEN("copy-on-write is invoked")
            {
                dh[0] += 2;
                REQUIRE(dh2 != dh);
            }
        }

        WHEN("manipulating a non-shallow-copied container")
        {
            for (index_t i = 0; i < dh.getSize(); ++i) {
                dh[i] += 2;
            }

            THEN("copy-on-write should not be invoked") { REQUIRE(useCount(dh) == 1); }
        }
    }
}
