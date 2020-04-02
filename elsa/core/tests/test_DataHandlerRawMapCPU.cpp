/**
 * \file test_DataHandlerRawMapCPU.cpp
 *
 * \brief Test for DataHandlerRawMapCPU
 *
 * \author David Tellenbach - initial code
 */

#include <catch2/catch.hpp>
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "DataHandlerRawMapCPU.h"
#include "testHelpers.h"

using namespace elsa;

TEMPLATE_TEST_CASE("DataHandlerRawMapCPU", "", float, double, std::complex<double>,
                   std::complex<float>)
{
    IndexVector_t vec(1);
    vec << 5;

    DataDescriptor desc(vec);

    Eigen::VectorX<TestType> coeff(5);
    coeff.setRandom();

    TestType* arr = coeff.data();

    DataContainer<TestType> dc(desc, arr);

    for (int i = 0; i < 10; ++i)
        REQUIRE(checkSameNumbers(dc[i], arr[i]));

    dc[2] = static_cast<TestType>(10);

    REQUIRE(checkSameNumbers(dc[2], static_cast<TestType>(10)));
    REQUIRE(checkSameNumbers(arr[2], static_cast<TestType>(10)));
}
