#pragma once

#include "Error.h"
#include "elsaDefines.h"
#include "Bessel.h"

namespace elsa
{
    namespace blobs
    {
        // The devinition given by Lewitt (1992):
        // \f$
        // b_{m, alpha, a}(r) = \frac{I_m(\alpha * \sqrt{1 - (r/a)^2)}{I_m(\alpha)} * (\sqrt{1 -
        // (r/a)^2})^m \f$ for any \f$ 0 <= r <= a \f$.
        template <typename data_t>
        constexpr data_t blob_evaluate(data_t r, SelfType_t<data_t> a, SelfType_t<data_t> alpha,
                                       index_t m) noexcept
        {
            const auto w = static_cast<data_t>(1) - std::pow(r / a, static_cast<data_t>(2));
            if (w >= 0) {
                const data_t Im1 = static_cast<data_t>(math::bessi(m, alpha));
                const data_t arg = std::sqrt(w);
                const data_t Im2 = static_cast<data_t>(math::bessi(m, alpha * arg));

                return (Im2 / Im1) * static_cast<data_t>(std::pow(arg, m));
            }
            return 0;
        }

        /// @brief Compute line integral of blob through a straight line
        /// The exact formulations are quite easily derived using the fact, that the line
        /// integral for a single blob is given by:
        ///
        /// \f$
        /// \frac{a}{I_m(\alpha)} \sqrt{\frac{2 \pi}{\alpha}} \left(\sqrt{1 -
        /// \left(\frac{s}{a}\right)^2}\right)^{m + 0.5} I_{m + 0.5}\left(\alpha \sqrt{1 -
        /// \left(\frac{s}{a}\right)^2}\right)
        /// \f$
        ///
        /// Then for a given order substitute I_{m + 0.5}(x) with the elementary function
        /// representations, as they can be found in Abramowitz and Stegun's Handbook of
        /// mathematical functions (1972):
        /// \f$ I_{0.5}(x) = \sqrt{\frac{2}{\pi x}} \sinh(x)\$f
        /// \f$ I_{1.5}(x) = \sqrt{\frac{2}{\pi x}} \left( \cosh(x) - \frac{\sinh(x)}{x} \right) \$f
        /// \f$ I_{2.5}(x) = \sqrt{\frac{2}{\pi x}} \left(\left(\frac{3}{x^2} +
        /// \frac{1}{x}\right)\sinh(x) - \frac{3}{x^2} \cosh(x)\right)\$f
        ///
        /// Which will result in the below formulations. In theory using the recurrent relations,
        /// this could be extended to further orders, but is deemed unnecessary for now.
        ///
        /// TODO: What about alpha = 0? Check if you find anything in the literature for that.
        ///
        /// @param distance distance of blob center to straight line, in literature often referred
        /// to as `r`
        /// @param radius radius of blob, often referred to as `a`
        /// @param alpha smoothness factor of blob, expected to be larger than 0
        /// @param order order of Bessel function, often referred to as `m`
        ///
        /// Ref:
        /// Distance-Driven Projection and Backprojection for Spherically Symmetric Basis Functions
        /// in CT - Levakhina
        /// Spherically symmetric volume elements as basis functions for image reconstructions in
        /// computed laminography - P. Trampert
        /// Semi-Discrete Iteration Methods in X-Ray Tomography - Jonas Vogelgesang
        template <typename data_t>
        constexpr data_t blob_projected(data_t s, SelfType_t<data_t> a, SelfType_t<data_t> alpha,
                                        index_t m)
        {
            // expect alpha > 0
            using namespace elsa::math;

            const data_t sda = s / a;
            const data_t sdas = std::pow(sda, 2.f);
            const data_t w = 1.f - sdas;

            if (w > 1.0e-10) {
                const auto arg = alpha * std::sqrt(w);
                if (m == 0) {
                    return (2.f * a / alpha) * std::sinh(arg)
                           / static_cast<data_t>(math::bessi0(alpha));
                } else if (m == 1) {
                    return (2.f * a / alpha) * std::sqrt(w)
                           * (std::cosh(arg) - std::sinh(arg) / arg)
                           / static_cast<data_t>(math::bessi1(alpha));

                } else if (m == 2) {
                    return (2.f * a / alpha) * w
                           * ((3.f / (arg * arg) + 1.f) * std::sinh(arg)
                              - (3.f / arg) * std::cosh(arg))
                           / static_cast<data_t>(math::bessi2(alpha));
                } else {
                    throw Error("m out of range in blob_projected()");
                }
            }
            return 0.0f;
        }

        template <typename data_t>
        constexpr data_t blob_projected(data_t s)
        {
            return blob_projected(s, 2.f, 10.83f, 2);
        }

        /// @brief Compute line integral of blob derivative through a straight line
        /// The exact formulations are quite easily derived using the fact, that the line
        /// integral for the derivative of a single blob is given by:
        ///
        /// \f$
        /// - \frac{\sqrt{2 \pi \alpha}}{I_m(\alpha)} \frac{s}{a} \left(\sqrt{1 -
        /// \left(\frac{s}{a}\right)^2}\right)^{m - 0.5} I_{m - 0.5}\left(\alpha \sqrt{1 -
        /// \left(\frac{s}{a}\right)^2}\right)
        /// \f$
        ///
        /// By substitution with the above formulas (blob_projected) one can derive the below
        /// formulations. In theory using the recurrent relations, this could be extended to
        /// further orders, but is deemed unnecessary for now.
        ///
        /// @param distance distance of blob center to straight line, in literature often referred
        /// to as `r`
        /// @param radius radius of blob, often referred to as `a`
        /// @param alpha smoothness factor of blob, expected to be larger than 0
        /// @param order order of Bessel function, often referred to as `m`
        ///
        /// Ref:
        /// Investigation of discrete imaging models and iterative image reconstruction
        /// in differential X-ray phase-contrast tomography - Qiaofeng Xu (Appendix B)
        template <typename data_t>
        constexpr data_t blob_derivative_projected(data_t s, SelfType_t<data_t> a,
                                                   SelfType_t<data_t> alpha, int m)
        {
            // expect alpha > 0
            using namespace elsa::math;

            const data_t sda = s / a;
            const data_t sdas = std::pow(sda, 2);
            const data_t w = 1.0 - sdas;

            if (w > 1.0e-10) {
                const auto arg = alpha * std::sqrt(w);
                if (m == 1) {
                    return (-2.0 * s / a) * std::sinh(arg) / bessi1(alpha);

                } else if (m == 2) {
                    return (-2.0 * s / alpha / a) * (std::cosh(arg) * arg - std::sinh(arg))
                           / bessi2(alpha);
                } else {
                    throw Error("m out of range in blob_projected()");
                }
            }
            return 0.0;
        }

        /// @brief For 3D objects we need to compute the directional gradient given by
        /// \f$ \frac{-g(\left\lVert \vec x\right\rVert)}{\left\lVert \vec x\right\rVert} \vec x^T
        /// \f$
        /// where g is the derivative computed below
        ///
        /// As we divide by the argument we have a potential divide by zero. This can be solved by
        /// moving the division inside g as it cancels out
        ///
        /// Compute line integral of blob derivative through a straight line
        /// The exact formulations are quite easily derived using the fact, that the line
        /// integral for the derivative of a single blob is given by:
        ///
        /// \f$
        /// - \frac{\sqrt{2 \pi \alpha}}{I_m(\alpha)} \frac{s}{a} \left(\sqrt{1 -
        /// \left(\frac{s}{a}\right)^2}\right)^{m - 0.5} I_{m - 0.5}\left(\alpha \sqrt{1 -
        /// \left(\frac{s}{a}\right)^2}\right)
        /// \f$
        ///
        /// By substitution with the above formulas (blob_projected) one can derive the below
        /// formulations. In theory using the recurrent relations, this could be extended to
        /// further orders, but is deemed unnecessary for now.
        ///
        /// @param distance distance of blob center to straight line, in literature often referred
        /// to as `r`
        /// @param radius radius of blob, often referred to as `a`
        /// @param alpha smoothness factor of blob, expected to be larger than 0
        /// @param order order of Bessel function, often referred to as `m`
        ///
        /// Ref:
        /// Investigation of discrete imaging models and iterative image reconstruction
        /// in differential X-ray phase-contrast tomography - Qiaofeng Xu (Appendix B)
        template <typename data_t>
        constexpr data_t blob_normalized_derivative_projected(data_t s, SelfType_t<data_t> a,
                                                              SelfType_t<data_t> alpha, int m)
        {
            // expect alpha > 0
            using namespace elsa::math;

            const data_t sda = s / a;
            const data_t sdas = std::pow(sda, 2);
            const data_t w = 1.0 - sdas;

            if (w > 1.0e-10) {
                const auto arg = alpha * std::sqrt(w);
                if (m == 1) {
                    return (-2.0 / a) * std::sinh(arg) / bessi1(alpha);

                } else if (m == 2) {
                    return (-2.0 / alpha / a) * (std::cosh(arg) * arg - std::sinh(arg))
                           / bessi2(alpha);
                } else {
                    throw Error("m out of range in blob_projected()");
                }
            }
            return 0.0;
        }

        template <typename data_t>
        constexpr data_t blob_derivative_projected(data_t s)
        {
            return blob_derivative_projected(s, 2.f, 10.83f, 2);
        }

    } // namespace blobs

    template <typename data_t>
    class Blob
    {
    public:
        constexpr Blob(data_t radius, SelfType_t<data_t> alpha, index_t order)
            : radius_(radius), alpha_(alpha), order_(order)
        {
        }
        constexpr data_t operator()(data_t s)
        {
            return blobs::blob_evaluate(s, radius_, alpha_, order_);
        }

        constexpr data_t radius() const { return radius_; }

        constexpr data_t alpha() const { return alpha_; }

        constexpr index_t order() const { return order_; }

    private:
        data_t radius_;
        data_t alpha_;
        index_t order_;
    };

    template <typename data_t>
    class ProjectedBlob
    {
    public:
        constexpr ProjectedBlob(data_t radius, SelfType_t<data_t> alpha, index_t order)
            : radius_(radius), alpha_(alpha), order_(order)
        {
        }

        constexpr data_t operator()(data_t s)
        {
            return blobs::blob_projected(s, radius_, alpha_, order_);
        }

        constexpr data_t derivative(data_t s)
        {
            return blobs::blob_derivative_projected(s, radius_, alpha_, order_);
        }

        constexpr data_t normalized_gradient(data_t s)
        {
            return blobs::blob_normalized_derivative_projected(s, radius_, alpha_, order_);
        }

        constexpr data_t radius() const { return radius_; }

        constexpr data_t alpha() const { return alpha_; }

        constexpr index_t order() const { return order_; }

    private:
        data_t radius_;
        data_t alpha_;
        index_t order_;
    };

} // namespace elsa
