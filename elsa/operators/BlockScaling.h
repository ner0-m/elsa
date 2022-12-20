#pragma once

#include "LinearOperator.h"
#include "DataContainer.h"

namespace elsa
{
    /**
     * \brief Block scaling operator.
     * \author Shen Hu (shen.hu@tum.de), port to elsa
     *
     * \tparam real_t real type
     */
    template <typename data_t = real_t>
    class BlockScaling : public LinearOperator<data_t>
    {
    private:
        typedef LinearOperator<data_t> B;
        std::unique_ptr<DataContainer<data_t>> _scales;

    protected:
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        BlockScaling<data_t>* cloneImpl() const override;
        bool isEqual(const LinearOperator<data_t>& other) const override;

    public:
        BlockScaling(const DataDescriptor& dataDescriptor, const DataContainer<data_t>& scales);
        ~BlockScaling() override = default;
    };

} // namespace elsa
