#pragma once

#include "Functional.h"

namespace elsa
{
    template <typename data_t = real_t>
    class AnisotropicTV : public Functional<data_t>
    {
    public:
        explicit AnisotropicTV(const DataDescriptor& domainDescriptor);

        AnisotropicTV(const AnisotropicTV<data_t>&) = delete;

        ~AnisotropicTV() override = default;

    protected:
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        AnisotropicTV<data_t>* cloneImpl() const override;

        bool isEqual(const Functional<data_t>& other) const override;
    };

} // namespace elsa
