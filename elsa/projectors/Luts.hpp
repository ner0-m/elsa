#pragma once

#include "Blobs.h"
#include "BSplines.h"
#include "Logger.h"
#include "Timer.h"
#include "TypeCasts.hpp"

#include <array>

namespace elsa
{
    namespace detail
    {
        template <typename data_t, index_t N>
        constexpr std::array<data_t, N + 1> generate_lut(data_t radius,
                                                         std::function<data_t(data_t)> gen)
        {
            Logger::get("generate_lut")->debug("Calculating lut");

            std::array<data_t, N + 1> lut;

            auto t = static_cast<data_t>(0);
            const auto step = radius / N;

            for (std::size_t i = 0; i <= N; ++i) {
                lut[i] = gen(t);
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
        constexpr Lut(std::array<data_t, N + 1>&& data) : data_(std::move(data)) {}

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
        constexpr data_t operator()(T index) const
        {
            if (index < 0 || index > asSigned(N)) {
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
        constexpr ProjectedBlobLut(data_t radius, SelfType_t<data_t> alpha, int order)
            : blob_(radius, alpha, order),
              lut_(detail::generate_lut<data_t, N>(blob_.radius(),
                                                   [this](data_t s) { return blob_(s); }))
        {
        }

        constexpr data_t radius() const { return blob_.radius(); }

        constexpr data_t alpha() const { return blob_.alpha(); }

        constexpr index_t order() const { return blob_.order(); }

        constexpr data_t operator()(data_t distance) const
        {
            return lut_((distance / blob_.radius()) * N);
        }

        constexpr auto data() const { return lut_.data(); }

    private:
        ProjectedBlob<data_t> blob_;
        Lut<data_t, N> lut_;
    };

    template <typename data_t, index_t N>
    class ProjectedBlobDerivativeLut
    {
    public:
        constexpr ProjectedBlobDerivativeLut(data_t radius, SelfType_t<data_t> alpha, int order)
            : blob_(radius, alpha, order),
              lut_(detail::generate_lut<data_t, N>(
                  blob_.radius(), [this](data_t s) { return blob_.derivative(s); }))
        {
        }

        constexpr data_t radius() const { return blob_.radius(); }

        constexpr data_t alpha() const { return blob_.alpha(); }

        constexpr data_t order() const { return blob_.order(); }

        constexpr data_t operator()(data_t distance) const
        {
            return lut_((distance / blob_.radius()) * N);
        }

        constexpr auto data() const { return lut_.data(); }

    private:
        ProjectedBlob<data_t> blob_;
        Lut<data_t, N> lut_;
    };

    template <typename data_t, index_t N>
    class ProjectedBlobNormalizedGradientLut
    {
    public:
        constexpr ProjectedBlobNormalizedGradientLut(data_t radius, SelfType_t<data_t> alpha,
                                                     int order)
            : blob_(radius, alpha, order),
              lut_(detail::generate_lut<data_t, N>(
                  blob_.radius(), [this](data_t s) { return blob_.normalized_gradient(s); }))
        {
        }

        constexpr data_t radius() const { return blob_.radius(); }

        constexpr data_t alpha() const { return blob_.alpha(); }

        constexpr data_t order() const { return blob_.order(); }

        constexpr data_t operator()(data_t distance) const
        {
            return lut_((distance / blob_.radius()) * N);
        }

        constexpr auto data() const { return lut_.data(); }

    private:
        ProjectedBlob<data_t> blob_;
        Lut<data_t, N> lut_;
    };

    template <typename data_t, index_t N>
    class ProjectedBSplineLut
    {
    public:
        constexpr ProjectedBSplineLut(int dim, int degree)
            : bspline_(dim, degree),
              lut_(detail::generate_lut<data_t, N>(2, [this](data_t s) { return bspline_(s); }))
        {
        }

        constexpr data_t order() const { return bspline_.order(); }

        constexpr data_t radius() const { return 2; }

        constexpr data_t operator()(data_t distance) const
        {
            return lut_((std::abs(distance) / radius()) * N);
        }

        constexpr auto data() const { return lut_.data(); }

    private:
        ProjectedBSpline<data_t> bspline_;
        Lut<data_t, N> lut_;
    };

    template <typename data_t, index_t N>
    class ProjectedBSplineDerivativeLut
    {
    public:
        constexpr ProjectedBSplineDerivativeLut(int dim, int degree)
            : bspline_(dim, degree),
              lut_(detail::generate_lut<data_t, N>(
                  radius(), [this](data_t s) { return bspline_.derivative(s); }))
        {
        }

        constexpr data_t order() const { return bspline_.order(); }

        constexpr data_t radius() const { return 2; }

        constexpr data_t operator()(data_t distance) const
        {
            return lut_((std::abs(distance) / radius()) * N);
        }

        constexpr auto data() const { return lut_.data(); }

    private:
        ProjectedBSpline<data_t> bspline_;
        Lut<data_t, N> lut_;
    };

    template <typename data_t, index_t N>
    class ProjectedBSplineNormalizedGradientLut
    {
    public:
        constexpr ProjectedBSplineNormalizedGradientLut(int dim, int degree)
            : bspline_(dim, degree),
              lut_(detail::generate_lut<data_t, N>(
                  radius(), [this](data_t s) { return bspline_.normalized_gradient(s); }))
        {
        }

        constexpr data_t radius() const { return 2; }

        constexpr data_t order() const { return bspline_.order(); }

        constexpr data_t operator()(data_t distance) const
        {
            return lut_((distance / radius()) * N);
        }

        constexpr auto data() const { return lut_.data(); }

    private:
        ProjectedBSpline<data_t> bspline_;
        Lut<data_t, N> lut_;
    };

    template <typename data_t, index_t N>
    class ProjectedBlobIntegralLut
    {
    public:
        constexpr ProjectedBlobIntegralLut(data_t radius, SelfType_t<data_t> alpha, int order)
            : blob_(2, 10.83f, 2),
              lut_({0.0,
                    0.01999582745226908,
                    0.03996663668708255,
                    0.05988749473735544,
                    0.07973363869741683,
                    0.09948055965827621,
                    0.11910408533639767,
                    0.13858046097381055,
                    0.1578864280986796,
                    0.176999300749408,
                    0.19589703878185413,
                    0.2145583178981765,
                    0.23296259605703826,
                    0.2510901759482483,
                    0.2689222632401957,
                    0.28644102033548724,
                    0.30362961539878636,
                    0.3204722664507877,
                    0.3369542803533164,
                    0.3530620865424669,
                    0.3687832653992891,
                    0.3841065711805187,
                    0.39902194946501945,
                    0.41352054910469527,
                    0.42759472870142917,
                    0.44123805766385354,
                    0.4544453119292471,
                    0.46721246446635645,
                    0.47953667070427086,
                    0.4914162490603811,
                    0.5028506567668497,
                    0.5138404612195974,
                    0.5243873070965821,
                    0.5344938795127993,
                    0.5441638634980116,
                    0.5534019000994952,
                    0.5622135394260536,
                    0.5706051909611212,
                    0.578584071481907,
                    0.5861581509281876,
                    0.5933360965685677,
                    0.6001272158137795,
                    0.606541398025917,
                    0.6125890556694965,
                    0.6182810651448981,
                    0.6236287076372484,
                    0.6286436103041741,
                    0.6333376881142686,
                    0.6377230866346649,
                    0.6418121260509684,
                    0.6456172466860923,
                    0.6491509562664619,
                    0.6524257791647572,
                    0.6554542078280072,
                    0.6582486565786742,
                    0.6608214179544891,
                    0.6631846217304422,
                    0.6653501967436776,
                    0.6673298356192741,
                    0.6691349624721584,
                    0.6707767036379354,
                    0.6722658614633253,
                    0.6736128911653764,
                    0.6748278807478248,
                    0.6759205339430076,
                    0.6769001561287634,
                    0.6777756431518853,
                    0.6785554729730219,
                    0.67924770003254,
                    0.6798599522228912,
                    0.6803994303404358,
                    0.6808729098786335,
                    0.6812867450149535,
                    0.6816468746358587,
                    0.681958830237772,
                    0.6822277455370334,
                    0.682458367618474,
                    0.6826550694503613,
                    0.6828218635930252,
                    0.6829624169294476,
                    0.6830800662483654,
                    0.6831778345140078,
                    0.6832584476612613,
                    0.683324351760862,
                    0.6833777304059753,
                    0.6834205221791454,
                    0.6834544380670345,
                    0.6834809786994005,
                    0.6835014512984184,
                    0.6835169862344698,
                    0.6835285530949162,
                    0.6835369761829577,
                    0.683542949374378,
                    0.683547050270665,
                    0.6835497535976234,
                    0.6835514438089612,
                    0.6835524268645214,
                    0.6835529411625452,
                    0.6835531676147164,
                    0.6835532388615568,
                    0.6835532476340136})
        {
        }

        constexpr data_t radius() const { return blob_.radius(); }

        constexpr data_t alpha() const { return blob_.alpha(); }

        constexpr data_t order() const { return blob_.order(); }

        constexpr data_t operator()(data_t distance) const
        {
            if (distance >= 2)
                distance = 2.0;
            return lut_((distance / blob_.radius()) * N);
        }

        constexpr auto data() const { return lut_.data(); }

    private:
        ProjectedBlob<data_t> blob_;
        Lut<data_t, N> lut_;
    };
} // namespace elsa
