#pragma once

#include <type_traits>
#include <complex.h>

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
bool checkSameNumbers(T right, T left)
{
    if constexpr (std::is_same_v<T,
                                 std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        CHECK(Approx(right.real()) == left.real());
        CHECK(Approx(right.imag()) == left.imag());
        return Approx(right.real()) == left.real() && Approx(right.imag()) == left.imag();
    } else {
        CHECK(Approx(right) == left);
        return Approx(right) == left;
    }
}
