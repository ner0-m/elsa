#pragma once

#include "LinearOperator.h"

// TODO extend from DiscreteShearletTransform?, do we gain anything from it? probably not
namespace elsa
{
    /**
     * @brief Class representing a (regular) Cone Adapted Discrete Shearlet Transform
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * References:
     * https://www.math.uh.edu/~dlabate/SHBookIntro.pdf
     * https://www.math.uh.edu/~dlabate/Athens.pdf
     */
    template <typename data_t = real_t>
    class ConeAdaptedDiscreteShearletTransform : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for a (regular) cone adapted discrete shearlet transform.
         *
         * @param[in] descriptor DataDescriptor describing the domain and the range of the operator
         * @param[in] scaleFactor the scalar factor to scale with
         */
        // TODO what are the shearlet system inputs? infer here max of these values and span those
        //  index spaces (use different inputs here)
        ConeAdaptedDiscreteShearletTransform(index_t width, index_t height);

        /// default destructor
        ~ConeAdaptedDiscreteShearletTransform() override = default;

    protected:
        // image to wavefront?
        // SH: R ^ n^2 -> R ^ J x n x n
        // dot product of the image and psi(j, k, m)?
        /// apply the (regular) discrete shearlet transform
        void applyImpl(const DataContainer<data_t>& f, DataContainer<data_t>& SHf) const override;

        // wavefront to image?
        // SH^-1: R ^ J x n x n -> R ^ n^2 // TODO is SH orthogonal i.e. SH^-1 = SH^T?
        /// apply the adjoint of the (regular) discrete shearlet transform // TODO rename y to g?
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& SHty) const override;

        /// implement the polymorphic clone operation
        ConeAdaptedDiscreteShearletTransform<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        real_t v(data_t x) const
        {
            if (x < 0) {
                return 0;
            } else if (0 <= x <= 1) {
                return 35 * std::pow(x, 4) - 84 * std::pow(x, 5) + 70 * std::pow(x, 6)
                       - 20 * std::pow(x, 7);
            } else {
                return 1;
            }
        }

        // TODO pi<real_t>, reconsider
        real_t b(data_t w) const
        {
            if (1 <= std::abs(w) <= 2) {
                return std::sin(pi_t / 2 * v(std::abs(w) - 1));
            } else if (2 < std::abs(w) <= 4) {
                return std::cos(pi_t / 2 * v(1 / 2 * std::abs(w) - 1));
            } else {
                return 0;
            }
        }

        real_t phi(data_t w) const
        {
            if (std::abs(w) <= 1 / 2) {
                return 1;
            } else if (1 / 2 < std::abs(w) < 1) {
                return std::cos(pi_t / 2 * v(2 * std::abs(w) - 1));
            } else {
                return 0;
            }
        }

        real_t phiHat(data_t w1, data_t w2) const
        {
            if (std::abs(w2) <= std::abs(w1)) {
                return phi(w1);
            } else {
                return phi(w2);
            }
        }

        real_t psiHat1(data_t w) const
        {
            return std::sqrt(std::pow(b(2 * w), 2) + std::pow(b(w), 2));
        }

        real_t psiHat2(data_t w) const
        {
            if (w <= 0) {
                return std::sqrt(v(1 + w));
            } else {
                return std::sqrt(v(1 - w));
            }
        }

        real_t psiHat(data_t w1, data_t w2) const
        {
            if (w1 == 0) {
                return 0;
            } else {
                return psiHat1(w1) * psiHat2(w2 / w1);
            }
        }
    };
} // namespace elsa
