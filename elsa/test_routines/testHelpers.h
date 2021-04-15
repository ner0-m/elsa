#pragma once

#include <type_traits>
#include <complex>
#include <random>
#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DataContainer.h"

#include <iomanip>
#include <limits>
#include <cassert>

namespace elsa
{
    /**
     * @brief Epsilon value for our test suit
     */
    static constexpr elsa::real_t epsilon = static_cast<elsa::real_t>(0.0001);

    /**
     * @brief comparing two number types for approximate equality for complex and regular number
     *
     * @tparam T - arithmetic data type
     * @return true if same number
     *
     * Use example in test case: REQUIRE(checkSameNumbers(a, b));
     * The CHECK(...) assertion in the function ensures that the values are reported when the test
     * fails
     */
    template <typename T>
    bool checkSameNumbers(T left, T right)
    {
        if constexpr (std::is_same_v<
                          T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            CHECK(Approx(left.real()).margin(epsilon) == right.real());
            CHECK(Approx(left.imag()).margin(epsilon) == right.imag());
            return Approx(left.real()).margin(epsilon) == right.real()
                   && Approx(left.imag()).margin(epsilon) == right.imag();
        } else {
            CHECK(Approx(left).margin(epsilon) == right);
            return Approx(left).margin(epsilon) == right;
        }
    }

    /**
     * @brief Generates a random Eigen matrix for different data_t types with integer values limited
     * to a certain range
     *
     * @param[in] size the number of elements in the vector like matrix
     *
     * @tparam data_t the numerical type to use
     *
     * The integer range is chosen to be small, to allow multiplication with the values without
     * running into overflow issues.
     */
    template <typename data_t>
    auto generateRandomMatrix(index_t size)
    {
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> randVec(size);

        if constexpr (std::is_integral_v<data_t>) {
            // Define range depending on signed or unsigned type
            const auto [rangeBegin, rangeEnd] = []() -> std::tuple<data_t, data_t> {
                if constexpr (std::is_signed_v<data_t>) {
                    return {-100, 100};
                } else {
                    return {1, 100};
                }
            }();

            std::random_device rd;
            std::mt19937 eng(rd());
            std::uniform_int_distribution<data_t> distr(rangeBegin, rangeEnd);

            for (index_t i = 0; i < size; ++i) {
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
     * @brief generate a random eigen vector and a DataContainer with the same data. Specifically
     * take index_t into consideration and scale the random eigen vector, to not generate overflows
     *
     * @tparam data_t Value type of DataContainers
     * @param desc First DataContainer
     * @param handlerType Second DataContainer
     *
     * @return a pair of a DataContainer and eigen vector, of same size and the same values
     */
    template <typename data_t>
    std::tuple<DataContainer<data_t>, Eigen::Matrix<data_t, Eigen::Dynamic, 1>>
        generateRandomContainer(const DataDescriptor& desc, DataHandlerType handlerType)
    {
        auto containerSize = desc.getNumberOfCoefficients();

        auto randVec = generateRandomMatrix<data_t>(containerSize);

        auto dc = DataContainer<data_t>(desc, randVec, handlerType);

        return {dc, randVec};
    }
    /**
     * @brief Compares two DataContainers using their norm. Computes \f$ \sqrt{\| x - y \|_{2}^2}
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
    bool isApprox(const DataContainer<data_t>& x, const DataContainer<data_t>& y,
                  real_t prec = Eigen::NumTraits<real_t>::dummy_precision());

    /**
     * @brief Wrapper to remove const, volatile and reference of a type
     */
    template <typename T>
    using UnqualifiedType_t =
        typename std::remove_cv<typename std::remove_reference<T>::type>::type;

    /**
     * @brief Helper to give types a name, this is used to print information during testing
     *
     * @tparam T type that should be given a name
     * @tparam Dummy dummy, to be used to enable or disable specific specializations
     */
    template <typename T, typename Dummy = void>
    struct TypeName;

    /**
     * @brief specialization to specify a name for index_t
     * @tparam T [const] [volatile] index_t[&] should be accepted
     */
    template <typename T>
    struct TypeName<T, std::enable_if_t<std::is_same_v<index_t, UnqualifiedType_t<T>>>> {
        static constexpr char name[] = "index_t";
    };

    /**
     * @brief specialization to specify a name for float
     * @tparam T [const] [volatile] float[&] should be accepted
     */
    template <typename T>
    struct TypeName<T, std::enable_if_t<std::is_same_v<float, UnqualifiedType_t<T>>>> {
        static constexpr char name[] = "float";
    };

    /**
     * @brief specialization to specify a name for double
     * @tparam T [const] [volatile] double[&] should be accepted
     */
    template <typename T>
    struct TypeName<T, std::enable_if_t<std::is_same_v<double, UnqualifiedType_t<T>>>> {
        static constexpr char name[] = "double";
    };

    /**
     * @brief specialization to specify a name for complex<float>
     * @tparam T [const] [volatile] complex<float>[&] should be accepted
     */
    template <typename T>
    struct TypeName<T,
                    std::enable_if_t<std::is_same_v<std::complex<float>, UnqualifiedType_t<T>>>> {
        static constexpr char name[] = "complex<float>";
    };

    /**
     * @brief specialization to specify a name for complex<double>
     * @tparam T [const] [volatile] complex<double>[&] should be accepted
     */
    template <typename T>
    struct TypeName<T,
                    std::enable_if_t<std::is_same_v<std::complex<double>, UnqualifiedType_t<T>>>> {
        static constexpr char name[] = "complex<double>";
    };

    /**
     * @brief Quick access to TypeName<UnqualifiedType>::name
     * @tparam T a type
     */
    template <typename T>
    static constexpr auto TypeName_v = TypeName<UnqualifiedType_t<T>>::name;
} // namespace elsa
