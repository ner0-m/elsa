#pragma once

#include "elsaDefines.h"

namespace elsa
{
    template <class T>
    struct SelfType {
        using type = T;
    };

    template <class T>
    using SelfType_t = typename SelfType<T>::type;

    namespace blobs
    {
        template <typename data_t>
        constexpr data_t blob_evaluate(data_t r, SelfType_t<data_t> a, SelfType_t<data_t> alpha,
                                       SelfType_t<data_t> m)
        {
            const auto w = static_cast<data_t>(1) - std::pow(r / a, static_cast<data_t>(2));
            if (w >= 0) {
                const data_t Im1 = std::cyl_bessel_i(m, alpha);
                const data_t arg = std::sqrt(w);
                const data_t Im2 = std::cyl_bessel_i(m, alpha * arg);

                return Im2 / Im1 * std::pow(arg, m);
            }
            return 0;
        }

        /// @brief Compute line integral of blob through a straight line
        /// @param distance distance of blob center to straight line, in literature often referred
        /// to as `r`
        /// @param radius radius of blob, often referred to as `a`
        /// @param alpha smoothness factor of blob
        /// @param order order of Bessel function, often referred to as `m`
        /// Ref:
        /// https://github.com/I2PC/xmipp/blob/3d4cc3f430cbc99a337635edbd95ebbcef33fc44/src/xmipp/libraries/data/blobs.cpp#L91A
        /// Distance-Driven Projection and Backprojection for Spherically Symmetric Basis Functions
        /// in CT - Levakhina
        /// Spherically symmetric volume elements as basis functions for image reconstructions in
        /// computed laminography - P. Trampert
        /// Semi-Discrete Iteration Methods in X-Ray Tomography - Jonas Vogelgesang
        template <typename data_t>
        constexpr data_t blob_projected(data_t s, SelfType_t<data_t> a, SelfType_t<data_t> alpha,
                                        SelfType_t<data_t> m)
        {
            // Equation derived in Lewitt 1990
            const data_t w = static_cast<data_t>(1) - ((s * s) / (a * a));

            // If `w` is close to zero or negative, `s` > `a`, and therefore just return 0
            if (w > 1e-10) {
                const data_t root = std::sqrt(w);

                // First three terms of equation
                const data_t q1 = a / std::cyl_bessel_i(m, alpha);
                const data_t q2 = std::sqrt(2 * pi<data_t> / alpha);
                const data_t q3 = std::pow(root, m + static_cast<data_t>(0.5));

                const data_t q4 = std::cyl_bessel_i(m + static_cast<data_t>(0.5), alpha * root);

                return q1 * q2 * q3 * q4;
            }
            return 0;
        }

        template <typename data_t>
        constexpr data_t blob_projected(data_t s)
        {
            return blob_projected(s, 2.f, 10.83f, 2);
        }
    } // namespace blobs

    template <typename data_t>
    class Blob
    {
    public:
        constexpr Blob(data_t radius, SelfType_t<data_t> alpha, SelfType_t<data_t> order);

        constexpr data_t operator()(data_t s);

        constexpr data_t radius() const;

        constexpr data_t alpha() const;

        constexpr data_t order() const;

    private:
        data_t radius_;
        data_t alpha_;
        data_t order_;
    };

    template <typename data_t>
    class ProjectedBlob
    {
    public:
        constexpr ProjectedBlob(data_t radius, SelfType_t<data_t> alpha, SelfType_t<data_t> order)
            : radius_(radius), alpha_(alpha), order_(order)
        {
        }

        constexpr data_t operator()(data_t s)
        {
            return blobs::blob_projected(s, radius_, alpha_, order_);
        }
#
        data_t radius() const { return radius_; }

        data_t alpha() const { return alpha_; }

        data_t order() const { return order_; }

    private:
        data_t radius_;
        data_t alpha_;
        data_t order_;
    };

} // namespace elsa
