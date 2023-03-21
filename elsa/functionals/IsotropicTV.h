#pragma once

#include "Functional.h"

namespace elsa
{
    template <typename data_t = real_t>
    class IsotropicTV : public Functional<data_t>
    {
    public:
        explicit IsotropicTV(const DataDescriptor& domainDescriptor);

        IsotropicTV(const IsotropicTV<data_t>&) = delete;

        ~IsotropicTV() override = default;

    protected:
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        IsotropicTV<data_t>* cloneImpl() const override;

        bool isEqual(const Functional<data_t>& other) const override;
    };

} // namespace elsa
