#pragma once

#include "Logger.h"
#include "Timer.h"
#include "TypeCasts.hpp"

#include <array>

#define DEFAULT_LUT_SIZE 101

namespace elsa
{
    namespace detail
    {
        template <typename data_t, index_t N>
        constexpr std::array<data_t, N> generate_lut(std::function<data_t(data_t)> gen,
                                                     data_t radius)
        {
            Logger::get("generate_lut")->debug("Calculating lut");

            std::array<data_t, N> lut;

            auto t = static_cast<data_t>(0);
            const auto step = radius / (N - 1);

            for (std::size_t i = 0; i < N; ++i) {
                lut[i] = gen(t);
                t += step;
            }

            return lut;
        }

    } // namespace detail

    template <typename data_t, std::size_t N>
    class Lut
    {
    public:
        constexpr Lut(std::function<data_t(data_t)> gen, data_t support = 1)
            : _support(support), data_(detail::generate_lut<data_t, N>(gen, support))
        {
        }

        constexpr Lut(std::array<data_t, N>&& data, data_t support = 1)
            : _support(support), data_(data)
        {
        }

        template <typename T, std::enable_if_t<std::is_integral_v<T>, index_t> = 0>
        constexpr data_t operator()(T index) const
        {
            if (index < 0 || index > asSigned(N)) {
                return 0;
            }

            return data_[index];
        }

        /// TODO: Handle boundary conditions
        /// lerp(last, last+1, t), for some t > 0, yields f(last) / 2, as f(last + 1) = 0,
        /// this should be handled
        template <typename T, std::enable_if_t<std::is_floating_point_v<T>, index_t> = 0>
        constexpr data_t operator()(T distance) const
        {
            T index = (distance / _support) * (N - 1);
            if (index < 0 || index > asSigned(N - 1)) {
                return 0;
            }

            // Get the two closes indices
            const auto a = static_cast<std::size_t>(std::floor(index));
            const auto b = static_cast<std::size_t>(std::ceil(index));

            // Get the function values
            const auto fa = data_[a];
            const auto fb = data_[b];

            const auto t = index - static_cast<data_t>(a);
            return t * fb + (1 - t) * fa;
        }

        constexpr auto data() const { return data_; }

        constexpr auto size() const { return N; }

        constexpr auto support() const { return _support; }

    private:
        const data_t _support;
        const std::array<data_t, N> data_;
    };
} // namespace elsa
