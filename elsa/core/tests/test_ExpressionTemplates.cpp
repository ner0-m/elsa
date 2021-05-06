/**
 * @file test_ExpressionTemplates.cpp
 *
 * @brief Tests for Expression Templates
 *
 * @author Jens Petit
 */

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "doctest/doctest.h"
#include "DataContainer.h"
#include "IdenticalBlocksDescriptor.h"
#include <typeinfo>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <cxxabi.h>

using namespace elsa;
using namespace doctest;
static const index_t dimension = 16;

// helper to print out the type
template <class T>
std::string type_name()
{
    using typeT = typename std::remove_reference<T>::type;
    std::unique_ptr<char, void (*)(void*)> own(
        abi::__cxa_demangle(typeid(typeT).name(), nullptr, nullptr, nullptr), std::free);

    std::string r = own != nullptr ? own.get() : typeid(typeT).name();
    if (std::is_const<typeT>::value)
        r += " const";
    if (std::is_volatile<typeT>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("Expression templates", TestType, float, double)
{
    GIVEN("Three random data containers")
    {
        srand(static_cast<unsigned>(time(nullptr)));

        IndexVector_t numCoeff(3);
        numCoeff << dimension, dimension, dimension;
        VolumeDescriptor desc(numCoeff);
        DataContainer<TestType> dc(desc);
        DataContainer<TestType> dc2(desc);
        DataContainer<TestType> dc3(desc);
        DataContainer<TestType> result(desc);

        for (index_t i = 0; i < dc.getSize(); ++i) {
            dc[i] = static_cast<TestType>(rand()) / (static_cast<TestType>(RAND_MAX / 100.0));
            dc2[i] = static_cast<TestType>(rand()) / (static_cast<TestType>(RAND_MAX / 100.0));
            dc3[i] = static_cast<TestType>(rand()) / (static_cast<TestType>(RAND_MAX / 100.0));
        }

        WHEN("using auto with an expression")
        {
            auto exp = dc * dc2;
            auto exp2 = dc * dc2 - dc;

            THEN("the type is an (nested) expression type")
            {
                INFO(type_name<decltype(exp)>());
                INFO(type_name<decltype(exp2)>());
            }
        }

        WHEN("writing into a result DataContainer")
        {
            result = square(dc * dc2 - dc);

            THEN("the type is a DataContainer again") { INFO(type_name<decltype(result)>()); }
        }

        WHEN("Mixing expression and DataContainers")
        {
            Expression exp = elsa::exp(sqrt(dc) * log(dc2));
            result = dc * dc2 - dc / exp;

            THEN("the type is a DataContainer again")
            {
                INFO(type_name<decltype(exp)>());
                INFO(type_name<decltype(result)>());
            }
        }

        WHEN("Constructing a new DataContainer out of an expression")
        {
            THEN("the type is a DataContainer again")
            {
                DataContainer newDC = dc * dc2 + dc3 / dc2;
                INFO(type_name<decltype(newDC)>());
            }

            THEN("the type is a DataContainer again")
            {
                DataContainer newDC2 = TestType(2.8) * dc2;
                INFO(type_name<decltype(newDC2)>());
            }

            THEN("the type is a DataContainer again")
            {
                DataContainer newDC2 = dc2 * TestType(2.8);
                INFO(type_name<decltype(newDC2)>());
            }
        }
    }

    GIVEN("Three DataContainers")
    {
        IndexVector_t numCoeff(3);
        numCoeff << dimension, dimension, dimension;
        VolumeDescriptor desc(numCoeff);
        DataContainer<TestType> dc1(desc);
        DataContainer<TestType> dc2(desc);
        DataContainer<TestType> dc3(desc);

        for (index_t i = 0; i < dc1.getSize(); ++i) {
            dc1[i] = float(i) + 1;
            dc2[i] = float(i) + 2;
            dc3[i] = float(i) + 3;
        }

        WHEN("Performing calculations")
        {
            DataContainer result = dc1 + dc1 * sqrt(dc2);

            THEN("the results have to be correct")
            {
                for (index_t i = 0; i < result.getSize(); ++i) {
                    REQUIRE_EQ(Approx(result[i]), dc1[i] + dc1[i] * std::sqrt(dc2[i]));
                }
            }
        }

        WHEN("Performing calculations")
        {
            DataContainer result = dc3 / dc1 * dc2 - dc3;

            THEN("the results have to be correct")
            {
                for (index_t i = 0; i < result.getSize(); ++i) {
                    REQUIRE_EQ(Approx(result[i]).epsilon(0.001), dc3[i] / dc1[i] * dc2[i] - dc3[i]);
                }
            }
        }

        WHEN("Performing in-place calculations")
        {
            THEN("then the element-wise in-place addition works as expected")
            {
                DataContainer dc1Before = dc1;
                dc1 += dc3 * 2 / dc2;
                for (index_t i = 0; i < dc1.getSize(); ++i) {
                    REQUIRE_EQ(Approx(dc1[i]).epsilon(0.001), dc1Before[i] + (dc3[i] * 2 / dc2[i]));
                }
            }

            THEN("then the element-wise in-place multiplication works as expected")
            {
                DataContainer dc1Before = dc1;
                dc1 *= dc3 * 2 / dc2;
                for (index_t i = 0; i < dc1.getSize(); ++i) {
                    REQUIRE_EQ(Approx(dc1[i]).epsilon(0.001), dc1Before[i] * (dc3[i] * 2 / dc2[i]));
                }
            }

            THEN("then the element-wise in-place division works as expected")
            {
                DataContainer dc1Before = dc1;
                dc1 /= dc3 * 2 / dc2;
                for (index_t i = 0; i < dc1.getSize(); ++i) {
                    REQUIRE_EQ(Approx(dc1[i]).epsilon(0.001), dc1Before[i] / (dc3[i] * 2 / dc2[i]));
                }
            }

            THEN("then the element-wise in-place subtraction works as expected")
            {
                DataContainer dc1Before = dc1;
                dc1 -= dc3 * 2 / dc2;
                for (index_t i = 0; i < dc1.getSize(); ++i) {
                    REQUIRE_EQ(Approx(dc1[i]).epsilon(0.001), dc1Before[i] - (dc3[i] * 2 / dc2[i]));
                }
            }
        }
    }

    GIVEN("A single blocked DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 52, 29;
        VolumeDescriptor desc(numCoeff);

        index_t numBlocks = 7;
        IdenticalBlocksDescriptor blockDesc(numBlocks, desc);

        DataContainer<TestType> dc(blockDesc);

        for (index_t i = 0; i < dc.getSize(); ++i) {
            dc[i] = float(i) + 1;
        }

        const DataContainer constDc(blockDesc);

        WHEN("Performing regular arithmetic")
        {
            THEN("the calculations are correct")
            {
                DataContainer result = TestType(1.8) * dc + dc;
                for (index_t i = 0; i < dc.getSize(); i++) {
                    REQUIRE_EQ(Approx(result[i]), TestType(1.8) * dc[i] + dc[i]);
                }
            }
        }

        WHEN("Performing blocked arithmetic")
        {
            THEN("the calculations are correct")
            {
                auto dcBlock = dc.getBlock(0);
                auto dcBlock2 = dc.getBlock(1);
                auto dcBlock3 = dc.getBlock(2);

                DataContainer result =
                    TestType(1.8) * dcBlock + dcBlock2 / dcBlock - square(dcBlock3);
                for (index_t i = 0; i < result.getSize(); i++) {
                    REQUIRE_EQ(Approx(result[i]), TestType(1.8) * dcBlock[i]
                                                      + dcBlock2[i] / dcBlock[i]
                                                      - dcBlock3[i] * dcBlock3[i]);
                }
            }
        }

        WHEN("Performing blocked arithmetic on a const block")
        {
            THEN("the calculations are correct")
            {
                const auto dcBlock = dc.getBlock(0);

                DataContainer result =
                    TestType(1.8) * dcBlock + dcBlock / dcBlock - square(dcBlock);
                for (index_t i = 0; i < result.getSize(); i++) {
                    REQUIRE_EQ(Approx(result[i]),
                               TestType(1.8) * dc[i] + dc[i] / dc[i] - dc[i] * dc[i]);
                }
            }
        }
    }

    GIVEN("A non-blocked container")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 52, 7, 29;
        VolumeDescriptor desc(numCoeff);

        DataContainer<TestType> dc(desc);

        for (index_t i = 0; i < dc.getSize(); ++i) {
            dc[i] = float(i) + 1;
        }

        WHEN("Creating a view and doing arithmetic")
        {

            IndexVector_t numCoeff(1);
            numCoeff << desc.getNumberOfCoefficients();
            VolumeDescriptor linearDesc(numCoeff);
            auto linearDc = dc.viewAs(linearDesc);

            DataContainer result =
                TestType(1.8) * linearDc + linearDc / linearDc - square(linearDc);

            THEN("the calculations are correct")
            {
                for (index_t i = 0; i < result.getSize(); i++) {
                    REQUIRE_EQ(Approx(result[i]), TestType(1.8) * linearDc[i]
                                                      + linearDc[i] / linearDc[i]
                                                      - linearDc[i] * linearDc[i]);
                }
            }
        }
    }

#ifdef ELSA_CUDA_VECTOR
    cudaDeviceReset();
#endif
}

#ifdef ELSA_CUDA_VECTOR
TEST_CASE_TEMPLATE("Expression templates: Testing on GPU", TestType, float, double)
{
    GIVEN("Three random data containers")
    {
        srand(static_cast<unsigned>(time(nullptr)));

        IndexVector_t numCoeff(3);
        numCoeff << dimension, dimension, dimension;
        VolumeDescriptor desc(numCoeff);
        DataContainer<TestType> dc(desc, DataHandlerType::GPU);
        DataContainer<TestType> dc2(desc, DataHandlerType::GPU);
        DataContainer<TestType> dc3(desc, DataHandlerType::GPU);
        DataContainer<TestType> result(desc, DataHandlerType::GPU);

        for (index_t i = 0; i < dc.getSize(); ++i) {
            dc[i] = static_cast<TestType>(rand()) / (static_cast<TestType>(RAND_MAX / 100.0));
            dc2[i] = static_cast<TestType>(rand()) / (static_cast<TestType>(RAND_MAX / 100.0));
            dc3[i] = static_cast<TestType>(rand()) / (static_cast<TestType>(RAND_MAX / 100.0));
        }

        WHEN("using auto with an expression")
        {
            auto exp = dc * dc2;
            auto exp2 = dc * dc2 - dc;

            THEN("the type is an (nested) expression type")
            {
                INFO(type_name<decltype(exp)>());
                INFO(type_name<decltype(exp2)>());
            }
        }

        WHEN("writing into a result DataContainer")
        {
            result = square(dc * dc2 - dc);

            THEN("the type is a DataContainer again") { INFO(type_name<decltype(result)>()); }
        }

        WHEN("Mixing expression and DataContainers")
        {
            Expression exp = elsa::exp(sqrt(dc) * log(dc2));
            result = dc * dc2 - dc / exp;

            THEN("the type is a DataContainer again")
            {
                INFO(type_name<decltype(exp)>());
                INFO(type_name<decltype(result)>());
            }
        }

        WHEN("Constructing a new DataContainer out of an expression")
        {
            THEN("the type is a DataContainer again")
            {
                DataContainer newDC = dc * dc2 + dc3 / dc2;
                INFO(type_name<decltype(newDC)>());
                static_assert(std::is_same_v<typename decltype(newDC)::value_type, TestType>);
            }

            THEN("the type is a DataContainer again")
            {
                DataContainer newDC2 = TestType(2.8) * dc2;
                INFO(type_name<decltype(newDC2)>());
                static_assert(std::is_same_v<typename decltype(newDC2)::value_type, TestType>);
            }

            THEN("the type is a DataContainer again")
            {
                DataContainer newDC2 = dc2 * TestType(2.8);
                INFO(type_name<decltype(newDC2)>());
                static_assert(std::is_same_v<typename decltype(newDC2)::value_type, TestType>);
            }
        }
    }

    GIVEN("Three DataContainers")
    {
        IndexVector_t numCoeff(3);
        numCoeff << dimension, dimension, dimension;
        VolumeDescriptor desc(numCoeff);
        DataContainer dc1(desc, DataHandlerType::GPU);
        DataContainer dc2(desc, DataHandlerType::GPU);
        DataContainer dc3(desc, DataHandlerType::GPU);

        for (index_t i = 0; i < dc1.getSize(); ++i) {
            dc1[i] = float(i) + 1;
            dc2[i] = float(i) + 2;
            dc3[i] = float(i) + 3;
        }

        WHEN("Performing calculations")
        {
            DataContainer result = dc1 + dc1 * sqrt(dc2);

            THEN("the results have to be correct")
            {
                for (index_t i = 0; i < result.getSize(); ++i) {
                    REQUIRE_EQ(Approx(result[i]), dc1[i] + dc1[i] * std::sqrt(dc2[i]));
                }
            }
        }

        WHEN("Performing calculations")
        {
            DataContainer result = dc3 / dc1 * dc2 - dc3;

            THEN("the results have to be correct")
            {
                for (index_t i = 0; i < result.getSize(); ++i) {
                    REQUIRE_EQ(Approx(result[i]).epsilon(0.001), dc3[i] / dc1[i] * dc2[i] - dc3[i]);
                }
            }
        }

        WHEN("Performing in-place calculations")
        {
            THEN("then the element-wise in-place addition works as expected")
            {
                DataContainer dc1Before = dc1;
                dc1 += dc3 * 2 / dc2;
                for (index_t i = 0; i < dc1.getSize(); ++i) {
                    REQUIRE_EQ(Approx(dc1[i]).epsilon(0.001), dc1Before[i] + (dc3[i] * 2 / dc2[i]));
                }
            }

            THEN("then the element-wise in-place multiplication works as expected")
            {
                DataContainer dc1Before = dc1;
                dc1 *= dc3 * 2 / dc2;
                for (index_t i = 0; i < dc1.getSize(); ++i) {
                    REQUIRE_EQ(Approx(dc1[i]).epsilon(0.001), dc1Before[i] * (dc3[i] * 2 / dc2[i]));
                }
            }

            THEN("then the element-wise in-place division works as expected")
            {
                DataContainer dc1Before = dc1;
                dc1 /= dc3 * 2 / dc2;
                for (index_t i = 0; i < dc1.getSize(); ++i) {
                    REQUIRE_EQ(Approx(dc1[i]).epsilon(0.001), dc1Before[i] / (dc3[i] * 2 / dc2[i]));
                }
            }

            THEN("then the element-wise in-place subtraction works as expected")
            {
                DataContainer dc1Before = dc1;
                dc1 -= dc3 * 2 / dc2;
                for (index_t i = 0; i < dc1.getSize(); ++i) {
                    REQUIRE_EQ(Approx(dc1[i]).epsilon(0.001), dc1Before[i] - (dc3[i] * 2 / dc2[i]));
                }
            }
        }
    }

    cudaDeviceReset();
}
#endif

TEST_SUITE_END();
