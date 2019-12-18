/**
 * \file test_ExpressionTemplates.cpp
 *
 * \brief Tests for Expression Templates
 *
 * \author Jens Petit
 */

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>
#include <iostream>
#include "DataContainer.h"
#include "BlockDescriptor.h"
#include <typeinfo>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <cxxabi.h>

using namespace elsa;
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

SCENARIO("Expression templates")
{
    GIVEN("Three random data containers")
    {
        srand(static_cast<unsigned>(time(nullptr)));

        IndexVector_t numCoeff(3);
        numCoeff << dimension, dimension, dimension;
        DataDescriptor desc(numCoeff);
        DataContainer dc(desc);
        DataContainer dc2(desc);
        DataContainer dc3(desc);
        DataContainer result(desc);

        for (index_t i = 0; i < dc.getSize(); ++i) {
            dc[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
            dc2[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
            dc3[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
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
            DataContainer newDC = dc * dc2 + dc3 / dc2;

            THEN("the type is a DataContainer again") { INFO(type_name<decltype(newDC)>()); }
        }
    }

    GIVEN("Three DataContainers")
    {
        IndexVector_t numCoeff(3);
        numCoeff << dimension, dimension, dimension;
        DataDescriptor desc(numCoeff);
        DataContainer dc1(desc);
        DataContainer dc2(desc);
        DataContainer dc3(desc);

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
                    REQUIRE(Approx(result[i]) == dc1[i] + dc1[i] * std::sqrt(dc2[i]));
                }
            }
        }

        WHEN("Performing calculations")
        {
            DataContainer result = dc3 / dc1 * dc2 - dc3;

            THEN("the results have to be correct")
            {
                for (index_t i = 0; i < result.getSize(); ++i) {
                    REQUIRE(Approx(result[i]).epsilon(0.001) == dc3[i] / dc1[i] * dc2[i] - dc3[i]);
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
                    REQUIRE(Approx(dc1[i]).epsilon(0.001) == dc1Before[i] + (dc3[i] * 2 / dc2[i]));
                }
            }

            THEN("then the element-wise in-place multiplication works as expected")
            {
                DataContainer dc1Before = dc1;
                dc1 *= dc3 * 2 / dc2;
                for (index_t i = 0; i < dc1.getSize(); ++i) {
                    REQUIRE(Approx(dc1[i]).epsilon(0.001) == dc1Before[i] * (dc3[i] * 2 / dc2[i]));
                }
            }

            THEN("then the element-wise in-place division works as expected")
            {
                DataContainer dc1Before = dc1;
                dc1 /= dc3 * 2 / dc2;
                for (index_t i = 0; i < dc1.getSize(); ++i) {
                    REQUIRE(Approx(dc1[i]).epsilon(0.001) == dc1Before[i] / (dc3[i] * 2 / dc2[i]));
                }
            }

            THEN("then the element-wise in-place subtraction works as expected")
            {
                DataContainer dc1Before = dc1;
                dc1 -= dc3 * 2 / dc2;
                for (index_t i = 0; i < dc1.getSize(); ++i) {
                    REQUIRE(Approx(dc1[i]).epsilon(0.001) == dc1Before[i] - (dc3[i] * 2 / dc2[i]));
                }
            }
        }
    }

    GIVEN("A single blocked DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 52, 29;
        DataDescriptor desc(numCoeff);

        index_t numBlocks = 7;
        BlockDescriptor blockDesc(numBlocks, desc);

        DataContainer dc(blockDesc);

        for (index_t i = 0; i < dc.getSize(); ++i) {
            dc[i] = float(i) + 1;
        }

        const DataContainer constDc(blockDesc);

        WHEN("Performing regular arithmetic")
        {
            THEN("the calculations are correct")
            {
                DataContainer result = 1.8 * dc + dc;
                for (index_t i = 0; i < dc.getSize(); i++) {
                    REQUIRE(Approx(result[i]) == 1.8 * dc[i] + dc[i]);
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

                DataContainer result = 1.8 * dcBlock + dcBlock2 / dcBlock - square(dcBlock3);
                for (index_t i = 0; i < result.getSize(); i++) {
                    REQUIRE(Approx(result[i])
                            == 1.8 * dcBlock[i] + dcBlock2[i] / dcBlock[i]
                                   - dcBlock3[i] * dcBlock3[i]);
                }
            }
        }

        WHEN("Performing blocked arithmetic on a const block")
        {
            THEN("the calculations are correct")
            {
                const auto dcBlock = dc.getBlock(0);

                DataContainer result = 1.8 * dcBlock + dcBlock / dcBlock - square(dcBlock);
                for (index_t i = 0; i < result.getSize(); i++) {
                    REQUIRE(Approx(result[i]) == 1.8 * dc[i] + dc[i] / dc[i] - dc[i] * dc[i]);
                }
            }
        }
    }

    GIVEN("A non-blocked container")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 52, 7, 29;
        DataDescriptor desc(numCoeff);

        DataContainer dc(desc);

        for (index_t i = 0; i < dc.getSize(); ++i) {
            dc[i] = float(i) + 1;
        }

        WHEN("Creating a view and doing arithmetic")
        {

            IndexVector_t numCoeff(1);
            numCoeff << desc.getNumberOfCoefficients();
            DataDescriptor linearDesc(numCoeff);
            auto linearDc = dc.viewAs(linearDesc);

            DataContainer result = 1.8 * linearDc + linearDc / linearDc - square(linearDc);

            THEN("the calculations are correct")
            {
                for (index_t i = 0; i < result.getSize(); i++) {
                    REQUIRE(Approx(result[i])
                            == 1.8 * linearDc[i] + linearDc[i] / linearDc[i]
                                   - linearDc[i] * linearDc[i]);
                }
            }
        }
    }
}
