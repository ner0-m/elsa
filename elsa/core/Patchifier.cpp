#include "Patchifier.h"

namespace elsa
{
    template <typename data_t>
    Patchifier<data_t>::Patchifier(const VolumeDescriptor& imageShape, index_t blockSize,
                                   index_t stride)
        : _imageShape(imageShape), _blockSize(blockSize), _stride(stride)
    {
        IndexVector_t coeffsPerDim = imageShape.getNumberOfCoefficientsPerDimension();
        _nPatchesRow = (1 + (coeffsPerDim[0] - blockSize) / stride);
        _nPatchesCol = (1 + (coeffsPerDim[1] - blockSize) / stride);
    }

    template <typename data_t>
    DataContainer<data_t> Patchifier<data_t>::im2patches(const DataContainer<data_t>& image)
    {
        if (image.getDataDescriptor() != _imageShape)
            throw InvalidArgumentError(
                "Patchifier::im2patches: image has non-matching data descriptor");

        index_t nPatches = _nPatchesRow * _nPatchesCol;

        VolumeDescriptor patchDescriptor({_blockSize, _blockSize});
        IdenticalBlocksDescriptor patchesDescriptor(nPatches, patchDescriptor);
        DataContainer<data_t> patches(patchesDescriptor);

        for (index_t patchIdx = 0; patchIdx < nPatches; ++patchIdx) {
            IndexVector_t startIdx = getImageIdx(patchIdx);
            IndexVector_t endIdx = startIdx + IndexVector_t({_blockSize, _blockSize});

            DataContainer<data_t> patch(patchDescriptor);

            index_t k = 0;
            for (index_t i = startIdx[0]; i < endIdx[0]; ++i) {
                for (index_t j = startIdx[1]; j < endIdx[1]; ++j) {
                    patch[k] = image(IndexVector_t({i, j}));
                    ++k;
                }
            }
            patches.getBlock(patchIdx) = patch;
        }

        return patches;
    }

    template <typename data_t>
    DataContainer<data_t> Patchifier<data_t>::patches2im(const DataContainer<data_t>& patches)
    {
        DataContainer<data_t> image(_imageShape);

        const auto& patchesDescriptor =
            dynamic_cast<const IdenticalBlocksDescriptor&>(patches.getDataDescriptor());

        for (index_t patchIdx = 0; patchesDescriptor.getNumberOfBlocks(); ++patchIdx) {
            const auto& patch = patches.getBlock(patchIdx);
            IndexVector_t startIdx = getImageIdx(patchIdx);
            IndexVector_t endIdx = startIdx + IndexVector_t({_blockSize, _blockSize});

            index_t k = 0;
            for (index_t i = startIdx[0]; i < endIdx[0]; ++i) {
                for (index_t j = startIdx[1]; j < endIdx[1]; ++j) {
                    image(IndexVector_t({i, j})) = patch[k];
                    ++k;
                }
            }
        }

        return image;
    }

    template <typename data_t>
    IndexVector_t Patchifier<data_t>::getImageIdx(index_t patchNr)
    {
        IndexVector_t idx(2);
        idx[0] = patchNr / _nPatchesRow; // row
        idx[1] = patchNr % _nPatchesRow; // col
        return idx;
    }
    // ------------------------------------------
    // explicit template instantiation
    template class Patchifier<float>;
    template class Patchifier<double>;

} // namespace elsa
