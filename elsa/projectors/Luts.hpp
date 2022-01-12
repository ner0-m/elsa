#pragma once

#include "Blobs.h"
#include "Logger.h"

#include <array>

namespace elsa
{
    namespace detail
    {
        template <typename data_t, index_t N>
        constexpr std::array<data_t, N> blob_lut(ProjectedBlob<data_t> blob)
        {
            std::array<data_t, N> lut;

            auto t = static_cast<data_t>(0);
            const auto step = blob.radius() / N;

            for (std::size_t i = 0; i < N; ++i) {
                lut[i] = blob(t);
                t += step;
            }

            return lut;
        }

        template <typename data_t>
        data_t lerp(data_t a, SelfType_t<data_t> b, SelfType_t<data_t> t)
        {
            if ((a <= 0 && b >= 0) || (a >= 0 && b <= 0))
                return t * b + (1 - t) * a;

            if (t == 1)
                return b;

            const data_t x = a + t * (b - a);

            if ((t > 1) == (b > a))
                return b < x ? x : b;
            else
                return x < b ? x : b;
        }
    } // namespace detail

    template <typename data_t, std::size_t N>
    class Lut
    {
    public:
        Lut(std::array<data_t, N> data) : data_(std::move(data)) {}

        template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
        data_t operator()(T index) const
        {
            if (index < 0 || index > N) {
                return 0;
            }

            return data_[index];
        }

        /// TODO: Handle boundary conditions
        /// lerp(last, last+1, t), for some t > 0, yields f(last) / 2, as f(last + 1) = 0,
        /// this should be handled
        template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
        data_t operator()(T index) const
        {
            if (index < 0 || index > N) {
                return 0;
            }

            // Get the two closes indices
            const auto a = static_cast<std::size_t>(std::floor(index));
            const auto b = static_cast<std::size_t>(std::ceil(index));

            // Get the function values
            const auto fa = data_[a];
            const auto fb = data_[b];

            // Bilinear interpolation
            return detail::lerp(fa, fb, index - static_cast<data_t>(a));
        }

    private:
        std::array<data_t, N> data_;
    };

    // User defined deduction guide
    template <typename data_t, std::size_t N>
    Lut(std::array<data_t, N>) -> Lut<data_t, N>;

    template <typename data_t, index_t N>
    class ProjectedBlobLut
    {
    public:
        constexpr ProjectedBlobLut(data_t radius, SelfType_t<data_t> alpha,
                                   SelfType_t<data_t> order)
            : blob_(radius, alpha, order), lut_(detail::blob_lut<data_t, N>(blob_))
        {
        }

        data_t operator()(data_t distance) const { return lut_((distance / blob_.radius()) * N); }

    private:
        ProjectedBlob<data_t> blob_;
        Lut<data_t, N> lut_;
    };
} // namespace elsa
