#pragma once

#include "Blobs.h"
#include "BSplines.h"
#include "Logger.h"
#include "Timer.h"

#include <array>

namespace elsa
{
    namespace detail
    {
        template <typename data_t, index_t N>
        constexpr std::array<data_t, N + 1> generate_lut(ProjectedBlob<data_t> blob,
                                                         std::function<data_t(data_t)> gen)
        {
            Logger::get("generate_lut")->debug("Calculating lut");

            std::array<data_t, N + 1> lut;

            auto t = static_cast<data_t>(0);
            const auto step = blob.radius() / N;

            for (std::size_t i = 0; i <= N; ++i) {
                lut[i] = gen(t);
                t += step;
            }

            return lut;
        }

        template <typename data_t, index_t N>
        constexpr std::array<data_t, N + 1> bspline_lut(ProjectedBSpline<data_t> bspline)
        {
            Logger::get("bspline_lut")->debug("Calculating lut");

            std::array<data_t, N + 1> lut;

            auto t = static_cast<data_t>(0);
            const auto step = 2. / N;

            for (std::size_t i = 0; i <= N; ++i) {
                lut[i] = bspline(t);
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
        constexpr Lut(std::array<data_t, N + 1> data) : data_(std::move(data)) {}

        template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
        constexpr data_t operator()(T index) const
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
        constexpr data_t operator()(T index) const
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

            const auto t = index - static_cast<data_t>(a);
            return t * fb + (1 - t) * fa;
        }

    private:
        std::array<data_t, N + 1> data_;
    };

    // User defined deduction guide
    template <typename data_t, std::size_t N>
    Lut(std::array<data_t, N>) -> Lut<data_t, N - 1>;

    template <typename data_t, index_t N>
    class ProjectedBlobLut
    {
    public:
        constexpr ProjectedBlobLut(data_t radius, SelfType_t<data_t> alpha,
                                   SelfType_t<data_t> order)
            : blob_(radius, alpha, order),
              lut_(detail::generate_lut<data_t, N>(blob_, [this](data_t s) { return blob_(s); }))
        {
        }

        constexpr data_t radius() const { return blob_.radius(); }

        constexpr data_t alpha() const { return blob_.alpha(); }

        constexpr data_t order() const { return blob_.order(); }

        constexpr data_t operator()(data_t distance) const
        {
            return lut_((distance / blob_.radius()) * N);
        }

    private:
        ProjectedBlob<data_t> blob_;
        Lut<data_t, N> lut_;
    };

    template <typename data_t, index_t N>
    class ProjectedBlobDerivativeLut
    {
    public:
        constexpr ProjectedBlobDerivativeLut(data_t radius, SelfType_t<data_t> alpha,
                                             SelfType_t<data_t> order)
            : blob_(radius, alpha, order),
              lut_(detail::generate_lut<data_t, N>(
                  blob_, [this](data_t s) { return blob_.derivative(s); }))
        {
        }

        constexpr data_t radius() const { return blob_.radius(); }

        constexpr data_t alpha() const { return blob_.alpha(); }

        constexpr data_t order() const { return blob_.order(); }

        constexpr data_t operator()(data_t distance) const
        {
            return lut_((distance / blob_.radius()) * N);
        }

    private:
        ProjectedBlob<data_t> blob_;
        Lut<data_t, N> lut_;
    };

    template <typename data_t, index_t N>
    class ProjectedBlobGradientHelperLut
    {
    public:
        constexpr ProjectedBlobGradientHelperLut(data_t radius, SelfType_t<data_t> alpha,
                                                 SelfType_t<data_t> order)
            : blob_(radius, alpha, order),
              lut_(detail::generate_lut<data_t, N>(
                  blob_, [this](data_t s) { return blob_.gradient_helper(s); }))
        {
        }

        constexpr data_t radius() const { return blob_.radius(); }

        constexpr data_t alpha() const { return blob_.alpha(); }

        constexpr data_t order() const { return blob_.order(); }

        constexpr data_t operator()(data_t distance) const
        {
            return lut_((distance / blob_.radius()) * N);
        }

    private:
        ProjectedBlob<data_t> blob_;
        Lut<data_t, N> lut_;
    };

    template <typename data_t, index_t N>
    class ProjectedBSplineLut
    {
    public:
        constexpr ProjectedBSplineLut(int dim, int degree)
            : bspline_(dim, degree), lut_(detail::bspline_lut<data_t, N>(bspline_))
        {
        }

        constexpr data_t order() const { return bspline_.order(); }

        constexpr data_t operator()(data_t distance) const
        {
            return lut_((std::abs(distance) / 2.) * N);
        }

    private:
        ProjectedBSpline<data_t> bspline_;
        Lut<data_t, N> lut_;
    };
} // namespace elsa
