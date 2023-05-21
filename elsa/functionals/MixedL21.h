#pragma once

#include "Functional.h"

namespace elsa
{
    /**
     * @brief Class representing the smooth mixed L12 functional.
     *
     * General p,q norm definition:
     * \|A\|_{p, q}=\left(\sum_{j=1}^n\left(\sum_{i=1}^m\left|a_{i
     * j}\right|^p\right)^{\frac{q}{p}}\right)^{\frac{1}{q}}
     *
     * The mixed L12 functional evaluates to
     * \|A\|_{1,
     * 2}=\left(\sum_{j=1}^n\left(\sum_{i=1}^m\left|a_{ij}\right|^1\right)^2\right)^{\frac{1}{2}}
     *
     */
    template <typename data_t = real_t>
    class MixedL21 : public Functional<data_t>
    {
    public:
        explicit MixedL21(const DataDescriptor& domainDescriptor);

        MixedL21(const MixedL21<data_t>&) = delete;

        ~MixedL21() override = default;

    protected:
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        MixedL21<data_t>* cloneImpl() const override;

        bool isEqual(const Functional<data_t>& other) const override;
    };

} // namespace elsa
