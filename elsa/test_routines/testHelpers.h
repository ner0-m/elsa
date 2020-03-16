#pragma once

#include <type_traits>
#include <complex>
#include "elsaDefines.h"

/**
 * \brief comparing two number types for approximate equality for complex and regular number
 *
 * \tparam T - arithmetic data type
 * \return true if same number
 *
 * Use example in test case: REQUIRE(checkSameNumbers(a, b));
 * The CHECK(...) assertion in the function ensures that the values are reported when the test fails
 */
template <typename T>
bool checkSameNumbers(T left, T right, int epsilonFactor = 1)
{
    using numericalBaseType = elsa::GetFloatingPointType_t<T>;

    numericalBaseType eps = std::numeric_limits<numericalBaseType>::epsilon()
                            * static_cast<numericalBaseType>(epsilonFactor)
                            * static_cast<numericalBaseType>(100);

    if constexpr (std::is_same_v<T,
                                 std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        CHECK(Approx(left.real()).epsilon(eps) == right.real());
        CHECK(Approx(left.imag()).epsilon(eps) == right.imag());
        return Approx(left.real()).epsilon(eps) == right.real()
               && Approx(left.imag()).epsilon(eps) == right.imag();
    } else {
        CHECK(Approx(left).epsilon(eps) == right);
        return Approx(left).epsilon(eps) == right;
    }
}
