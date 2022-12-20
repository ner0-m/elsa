#include "BlockScaling.h"

namespace elsa
{
    template <typename data_t>
    BlockScaling<data_t>::BlockScaling(const DataDescriptor& dataDescriptor,
                                       const DataContainer<data_t>& scales)
        : B(dataDescriptor, dataDescriptor, true),
          _scales(std::make_unique<DataContainer<data_t>>(scales))
    {
    }

    template <typename data_t>
    void BlockScaling<data_t>::applyImpl(const DataContainer<data_t>& x,
                                         DataContainer<data_t>& Ax) const
    {
        Ax = x;

        IndexVector_t coeffPerDim = x.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        index_t numBlks = coeffPerDim[coeffPerDim.size() - 1];

        for (index_t i = 0; i < numBlks; ++i) {
            Ax.getBlock(i) *= (*_scales)[i];
        }
    }

    template <typename data_t>
    void BlockScaling<data_t>::applyAdjointImpl(const DataContainer<data_t>& x,
                                                DataContainer<data_t>& Atx) const
    {
        // copy values
        Atx = x;

        IndexVector_t coeffPerDim = x.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        index_t numBlks = coeffPerDim[coeffPerDim.size() - 1];

        // apply scaling
        for (index_t i = 0; i < numBlks; ++i) {
            Atx.getBlock(i) *= (*_scales)[i];
        }
    }

    template <typename data_t>
    BlockScaling<data_t>* BlockScaling<data_t>::cloneImpl() const
    {
        return new BlockScaling(this->getDomainDescriptor(), _scales);
    }

    template <typename data_t>
    bool BlockScaling<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other)) {
            return false;
        }

        auto otherBlockScaling = downcast_safe<BlockScaling>(&other);
        if (!otherBlockScaling)
            return false;

        if (_scales != otherBlockScaling->_scales)
            return false;

        return true;
    }
} // namespace elsa