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

SCENARIO("Testing expression templates")
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
                std::cout << type_name<decltype(exp)>() << std::endl;
                std::cout << type_name<decltype(exp2)>() << std::endl;
            }
        }

        WHEN("writing into a result DataContainer")
        {
            result = dc * dc2 - dc;

            THEN("the type is a DataContainer again")
            {
                std::cout << type_name<decltype(result)>() << std::endl;
            }
        }

        WHEN("Mixing expression and DataContainers")
        {
            auto exp = dc * dc2;
            result = dc * dc2 - dc / exp;

            THEN("the type is a DataContainer again")
            {
                std::cout << type_name<decltype(result)>() << std::endl;
            }
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
        DataContainer result(desc);

        for (index_t i = 0; i < dc1.getSize(); ++i) {
            dc1[i] = i;
            dc2[i] = i + 1;
            dc3[i] = i + 2;
        }

        WHEN("Performing calculations")
        {
            result = dc1 + dc1 * dc2;

            THEN("the results have to be correct")
            {

                for (index_t i = 0; i < result.getSize(); ++i) {
                    REQUIRE(result[i] == dc1[i] + dc1[i] * dc2[i]);
                }
            }
        }

        WHEN("Performing calculations")
        {
            result = dc3 / dc1 * dc2 - dc3;

            THEN("the results have to be correct")
            {
                for (index_t i = 0; i < result.getSize(); ++i) {
                    REQUIRE(result[i] == dc3[i] / dc1[i] * dc2[i] - dc3[i]);
                }
            }
        }
    }
}
