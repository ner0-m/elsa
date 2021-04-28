#pragma once

#include "DiscreteShearletTransform.h"

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
     */
    template <typename data_t = real_t>
    class ConeAdaptedDiscreteShearletTransform
        : public DiscreteShearletTransform<data_t> // or : public LinearOperator<data_t>?
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
        // scale index j, the orientation index k, and the position index m.
        ConeAdaptedDiscreteShearletTransform(std::vector<int> mPrime, int j, int k,
                                             std::vector<int> m, int jTilde, int kTilde,
                                             std::vector<int> mTilde);

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
        DiscreteShearletTransform<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        // 3 functions of cone-adapted wavelets

        DataContainer<data_t> phi(std::vector<int> mPrime); // m should have size 2

        DataContainer<data_t> psi(int j, int k, std::vector<int> m);

        DataContainer<data_t> psiTilde(int jTilde, int kTilde, std::vector<int> mTilde);

        // Scone = {(a,s,t): a ∈ (0,1], |s| ≤ 1+a^1/2, t ∈ R^2}.

        // NN: R ^ J x n x n -> R ^ J x n x n
    };
} // namespace elsa
