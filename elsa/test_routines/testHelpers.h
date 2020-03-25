#pragma once

#include <type_traits>
#include <complex>
#include <random>
#include "elsaDefines.h"

#include <iomanip>
#include <limits>
#include <cassert>

namespace elsa
{
    /**
     * \brief Epsilon value for our test suit
     */
    static constexpr elsa::real_t epsilon = static_cast<elsa::real_t>(0.0001);

    /**
     * \brief comparing two number types for approximate equality for complex and regular number
     *
     * \tparam T - arithmetic data type
     * \return true if same number
     *
     * Use example in test case: REQUIRE(checkSameNumbers(a, b));
     * The CHECK(...) assertion in the function ensures that the values are reported when the test
     * fails
     */
    template <typename T>
    bool checkSameNumbers(T left, T right, int epsilonFactor = 1)
    {
        using numericalBaseType = elsa::GetFloatingPointType_t<T>;

        numericalBaseType eps = std::numeric_limits<numericalBaseType>::epsilon()
                                * static_cast<numericalBaseType>(epsilonFactor)
                                * static_cast<numericalBaseType>(100);

        if constexpr (std::is_same_v<
                          T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            CHECK(Approx(left.real()).epsilon(eps) == right.real());
            CHECK(Approx(left.imag()).epsilon(eps) == right.imag());
            return Approx(left.real()).epsilon(eps) == right.real()
                   && Approx(left.imag()).epsilon(eps) == right.imag();
        } else {
            CHECK(Approx(left).epsilon(eps) == right);
            return Approx(left).epsilon(eps) == right;
        }
    }

    /**
     * \brief Generates a random Eigen matrix for different data_t types with integer values limited
     * to a certain range
     *
     * \param[in] size the number of elements in the vector like matrix
     *
     * \tparam data_t the numerical type to use
     *
     * The integer range is chosen to be small, to allow multiplication with the values without
     * running into overflow issues.
     */
    template <typename data_t>
    auto generateRandomMatrix(elsa::index_t size)
    {
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec(size);

        if constexpr (std::is_integral_v<data_t>) {
            std::random_device rd;
            std::mt19937 eng(rd());
            std::uniform_int_distribution<> distr(-100, 100);

            for (elsa::index_t i = 0; i < size; ++i) {
                data_t num = distr(eng);
                // remove zeros as this leads to errors when dividing
                if (num == 0)
                    num = 1;
                randVec[i] = num;
            }
        } else {
            randVec.setRandom();
        }

        return randVec;
    }

    /**
     * \brief Compares two DataContainers using their norm. Computes \f$ \sqrt{\| x - y \|_{2}^2}
     * \f$ and compares it to \f$ prec * \sqrt{min(\| x \|_{2}^2, \| y \|_{2}^2)} \f$. If the first
     * is smaller or equal to the second, we can assume the vectors are approximate equal
     *
     * @tparam data_t Value type of DataContainers
     * @param x First DataContainer
     * @param y Second DataContainer
     * @param prec Precision to compare, the smaller the closer both have to be
     * @return true if the norms of the containers is approximate equal
     */
    template <typename data_t>
    bool isApprox(const elsa::DataContainer<data_t>& x, const elsa::DataContainer<data_t>& y,
                  real_t prec = Eigen::NumTraits<real_t>::dummy_precision())
    {
        // Check size is the same, don't throw an expection, as it is a programming error to pass
        // containers with different size
        assert(x.getSize() == y.getSize());

        DataContainer<data_t> z = x;
        z -= y;

        data_t lhs = std::sqrt(z.squaredL2Norm());
        data_t rhs = prec * std::sqrt(std::min(x.squaredL2Norm(), y.squaredL2Norm()));
        return lhs <= rhs;
    }
} // namespace elsa
