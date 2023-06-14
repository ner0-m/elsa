#pragma once

#include "DataContainer.h"
#include "elsaDefines.h"

namespace elsa
{
    /**
     * @brief This is the proximal operator for the mixed L21 norm, or sometimes
     * also referred to as the group L1 Norm. The functional is specifically
     * important for isotropic TV
     *
     * A signal needs to be represented as different blocks. When viewing each
     * block as a 1D signal, you can create a matrix where each block is a row
     * in the matrix. Let \f$X \in \mathbb{R}^{m \times n}\f$, where \f$m\f$
     * is the number of blocks, and \f$n\f$ the size of each block linearized to
     * an 1D signal. The L21 Norm is then defined as:
     * \f[
     * ||X||_{2,1} = \sum_{j=0}^m || x_j ||_2
     * \f]
     * This is the sum (L1 norm) of the column-wise L2 norm.
     *
     * The proximal operator is then given by:
     * (1 - (tau * self.sigma) / np.maximum(aux, tau * self.sigma)) * x.ravel()
     * \f[
     * prox_{\sigma ||\cdot||_{2,1}}(x_j) = (1 - \frac{sigma}{\max\{ ||x_j||, 0 \}}) x_j \quad
     * \forall j
     * \f]
     * The factor \f$(1 - \frac{sigma}{\max\{ ||x_j||, 0\}})\f$ can be computed
     * for each column, which results in an \f$n\f$ sized vector, which, with
     * correct broadcasting, can be multiplied directly to the input signal.
     */
    template <typename data_t = real_t>
    class ProximalMixedL21Norm
    {
    public:
        ProximalMixedL21Norm() = default;

        ProximalMixedL21Norm(data_t sigma);

        ~ProximalMixedL21Norm() = default;

        /**
         * @brief apply the proximal operator of the l1 norm to an element in the operator's domain
         *
         * @param[in] v input DataContainer
         * @param[in] t input Threshold
         * @param[out] prox output DataContainer
         */
        void apply(const DataContainer<data_t>& v, SelfType_t<data_t> t,
                   DataContainer<data_t>& prox) const;

        DataContainer<data_t> apply(const DataContainer<data_t>& v, SelfType_t<data_t> t) const;

        data_t sigma() const { return sigma_; }

    private:
        data_t sigma_{1};
    };

    template <typename T>
    bool operator==(const ProximalMixedL21Norm<T>& lhs, const ProximalMixedL21Norm<T>& rhs)
    {
        return lhs.sigma() == rhs.sigma();
    }

    template <typename T>
    bool operator!=(const ProximalMixedL21Norm<T>& lhs, const ProximalMixedL21Norm<T>& rhs)
    {
        return !(lhs == rhs);
    }
} // namespace elsa
