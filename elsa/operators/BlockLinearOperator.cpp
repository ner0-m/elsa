#include "BlockLinearOperator.h"
#include "PartitionDescriptor.h"
#include "RandomBlocksDescriptor.h"
#include "DescriptorUtils.h"
#include "TypeCasts.hpp"

#include <algorithm>

namespace elsa
{
    template <typename data_t>
    BlockLinearOperator<data_t>::BlockLinearOperator(const OperatorList& ops, BlockType blockType)
        : LinearOperator<data_t>{*determineDomainDescriptor(ops, blockType),
                                 *determineRangeDescriptor(ops, blockType)},
          _operatorList(0),
          _blockType{blockType}
    {
        for (const auto& op : ops)
            _operatorList.push_back(op->clone());
    }

    template <typename data_t>
    BlockLinearOperator<data_t>::BlockLinearOperator(const DataDescriptor& domainDescriptor,
                                                     const DataDescriptor& rangeDescriptor,
                                                     const OperatorList& ops, BlockType blockType)
        : LinearOperator<data_t>{domainDescriptor, rangeDescriptor},
          _operatorList(0),
          _blockType{blockType}
    {
        if (_blockType == COL) {
            const auto* trueDomainDesc = downcast_safe<BlockDescriptor>(&domainDescriptor);

            if (trueDomainDesc == nullptr)
                throw InvalidArgumentError(
                    "BlockLinearOperator: domain descriptor is not a BlockDescriptor");

            if (trueDomainDesc->getNumberOfBlocks() != static_cast<index_t>(ops.size()))
                throw InvalidArgumentError("BlockLinearOperator: domain descriptor number of "
                                           "blocks does not match operator list size");

            for (index_t i = 0; i < static_cast<index_t>(ops.size()); i++) {
                const auto& op = ops[static_cast<std::size_t>(i)];
                if (op->getRangeDescriptor().getNumberOfCoefficients()
                    != _rangeDescriptor->getNumberOfCoefficients())
                    throw InvalidArgumentError(
                        "BlockLinearOperator: the range descriptor of a COL BlockLinearOperator "
                        "must have the same size as the range of every operator in the list");

                if (op->getDomainDescriptor().getNumberOfCoefficients()
                    != trueDomainDesc->getDescriptorOfBlock(i).getNumberOfCoefficients())
                    throw InvalidArgumentError(
                        "BlockLinearOperator: block of incorrect size in domain descriptor");
            }
        }

        if (_blockType == ROW) {
            const auto* trueRangeDesc = downcast_safe<BlockDescriptor>(&rangeDescriptor);

            if (trueRangeDesc == nullptr)
                throw InvalidArgumentError(
                    "BlockLinearOperator: range descriptor is not a BlockDescriptor");

            if (trueRangeDesc->getNumberOfBlocks() != static_cast<index_t>(ops.size()))
                throw InvalidArgumentError("BlockLinearOperator: range descriptor number of "
                                           "blocks does not match operator list size");

            for (index_t i = 0; i < static_cast<index_t>(ops.size()); i++) {
                const auto& op = ops[static_cast<std::size_t>(i)];
                if (op->getDomainDescriptor().getNumberOfCoefficients()
                    != _domainDescriptor->getNumberOfCoefficients())
                    throw InvalidArgumentError(
                        "BlockLinearOperator: the domain descriptor of a ROW BlockLinearOperator "
                        "must have the same size as the domain of every operator in the list");

                if (op->getRangeDescriptor().getNumberOfCoefficients()
                    != trueRangeDesc->getDescriptorOfBlock(i).getNumberOfCoefficients())
                    throw InvalidArgumentError(
                        "BlockLinearOperator: block of incorrect size in range descriptor");
            }
        }

        for (const auto& op : ops)
            _operatorList.push_back(op->clone());
    }

    template <typename data_t>
    const LinearOperator<data_t>& BlockLinearOperator<data_t>::getIthOperator(index_t index) const
    {
        return *_operatorList.at(static_cast<std::size_t>(index));
    }

    template <typename data_t>
    index_t BlockLinearOperator<data_t>::numberOfOps() const
    {
        return static_cast<index_t>(_operatorList.size());
    }

    template <typename data_t>
    BlockLinearOperator<data_t>::BlockLinearOperator(const BlockLinearOperator<data_t>& other)
        : LinearOperator<data_t>{*other._domainDescriptor, *other._rangeDescriptor},
          _operatorList(0),
          _blockType{other._blockType}
    {
        for (const auto& op : other._operatorList)
            _operatorList.push_back(op->clone());
    }

    template <typename data_t>
    void BlockLinearOperator<data_t>::applyImpl(const DataContainer<data_t>& x,
                                                DataContainer<data_t>& Ax) const
    {
        switch (_blockType) {
            case BlockType::COL: {
                Ax = 0;
                auto tmpAx = DataContainer<data_t>(Ax.getDataDescriptor());

                auto xView = x.viewAs(*_domainDescriptor);
                index_t i = 0;
                for (const auto& op : _operatorList) {
                    op->apply(xView.getBlock(i), tmpAx);
                    Ax += tmpAx;
                    ++i;
                }

                break;
            }
            case BlockType::ROW: {
                index_t i = 0;

                auto AxView = Ax.viewAs(*_rangeDescriptor);
                for (const auto& op : _operatorList) {
                    auto blk = AxView.getBlock(i);
                    op->apply(x, blk);
                    ++i;
                }

                break;
            }
        }
    }

    template <typename data_t>
    void BlockLinearOperator<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                       DataContainer<data_t>& Aty) const
    {
        switch (_blockType) {
            case BlockType::COL: {
                index_t i = 0;

                auto AtyView = Aty.viewAs(*_domainDescriptor);
                for (const auto& op : _operatorList) {
                    auto&& blk = AtyView.getBlock(i);
                    op->applyAdjoint(y, blk);
                    ++i;
                }

                break;
            }
            case BlockType::ROW: {
                Aty = 0;
                auto tmpAty = DataContainer<data_t>(Aty.getDataDescriptor());

                auto yView = y.viewAs(*_rangeDescriptor);
                index_t i = 0;
                for (const auto& op : _operatorList) {
                    op->applyAdjoint(yView.getBlock(i), tmpAty);
                    Aty += tmpAty;
                    ++i;
                }

                break;
            }
        }
    }

    template <typename data_t>
    BlockLinearOperator<data_t>* BlockLinearOperator<data_t>::cloneImpl() const
    {
        return new BlockLinearOperator<data_t>(*this);
    }

    template <typename data_t>
    bool BlockLinearOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        // static_cast as type checked in base comparison
        auto otherBlockOp = static_cast<const BlockLinearOperator<data_t>*>(&other);

        for (std::size_t i = 0; i < _operatorList.size(); i++)
            if (*_operatorList[i] != *otherBlockOp->_operatorList[i])
                return false;

        return true;
    }

    template <typename data_t>
    std::unique_ptr<DataDescriptor>
        BlockLinearOperator<data_t>::determineDomainDescriptor(const OperatorList& operatorList,
                                                               BlockType blockType)
    {
        std::vector<const DataDescriptor*> vec(operatorList.size());
        for (std::size_t i = 0; i < vec.size(); i++)
            vec[i] = &operatorList[i]->getDomainDescriptor();

        switch (blockType) {
            case BlockType::ROW:
                return bestCommon(vec);

            case BlockType::COL:
                return bestBlockDescriptor(vec);

            default:
                throw InvalidArgumentError("BlockLinearOpearator: unsupported block type");
        }
    }

    template <typename data_t>
    std::unique_ptr<DataDescriptor>
        BlockLinearOperator<data_t>::determineRangeDescriptor(const OperatorList& operatorList,
                                                              BlockType blockType)
    {
        std::vector<const DataDescriptor*> vec(operatorList.size());
        for (std::size_t i = 0; i < vec.size(); i++)
            vec[i] = &operatorList[i]->getRangeDescriptor();

        switch (blockType) {
            case BlockType::ROW:
                return bestBlockDescriptor(vec);

            case BlockType::COL:
                return bestCommon(vec);

            default:
                throw InvalidArgumentError("BlockLinearOpearator: unsupported block type");
        }
    }

    template <typename data_t>
    std::unique_ptr<BlockDescriptor> BlockLinearOperator<data_t>::bestBlockDescriptor(
        const std::vector<const DataDescriptor*>& descList)
    {
        auto numBlocks = descList.size();
        if (numBlocks == 0)
            throw InvalidArgumentError("BlockLinearOperator: operator list cannot be empty");

        const auto& firstDesc = *descList[0];
        auto numDims = firstDesc.getNumberOfDimensions();
        auto coeffs = firstDesc.getNumberOfCoefficientsPerDimension();

        bool allNumDimSame = true;
        bool allButLastDimSame = true;
        IndexVector_t lastDimSplit(numBlocks);
        for (std::size_t i = 1; i < numBlocks && allButLastDimSame; i++) {
            lastDimSplit[static_cast<index_t>(i - 1)] =
                descList[i - 1]->getNumberOfCoefficientsPerDimension()[numDims - 1];

            allNumDimSame = allNumDimSame && descList[i]->getNumberOfDimensions() == numDims;
            allButLastDimSame =
                allNumDimSame && allButLastDimSame
                && descList[i]->getNumberOfCoefficientsPerDimension().head(numDims - 1)
                       == coeffs.head(numDims - 1);
        }

        if (allButLastDimSame) {
            lastDimSplit[static_cast<index_t>(numBlocks) - 1] =
                descList[numBlocks - 1]->getNumberOfCoefficientsPerDimension()[numDims - 1];

            auto spacing = firstDesc.getSpacingPerDimension();
            bool allSameSpacing =
                all_of(descList.begin(), descList.end(), [&spacing](const DataDescriptor* d) {
                    return d->getSpacingPerDimension() == spacing;
                });

            coeffs[numDims - 1] = lastDimSplit.sum();
            if (allSameSpacing) {
                VolumeDescriptor tmp(coeffs, spacing);
                return std::make_unique<PartitionDescriptor>(tmp, lastDimSplit);
            } else {
                VolumeDescriptor tmp(coeffs);
                return std::make_unique<PartitionDescriptor>(tmp, lastDimSplit);
            }
        }

        std::vector<std::unique_ptr<DataDescriptor>> tmp(numBlocks);
        auto it = descList.begin();
        std::generate(tmp.begin(), tmp.end(), [&it]() { return (*it++)->clone(); });

        return std::make_unique<RandomBlocksDescriptor>(std::move(tmp));
    }

    // ----------------------------------------------
    // explicit template instantiation
    template class BlockLinearOperator<float>;
    template class BlockLinearOperator<double>;
    template class BlockLinearOperator<std::complex<float>>;
    template class BlockLinearOperator<std::complex<double>>;
} // namespace elsa
