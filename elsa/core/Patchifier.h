#pragma once

#include "DataContainer.h"
#include "IdenticalBlocksDescriptor.h"
#include "VolumeDescriptor.h"

namespace elsa
{

    /**
     * @brief class for turning an image into a set of (overlapping) patches
     *
     * @author Jonas Buerger - initial code
     *
     * @tparam data_t - data type that is used for the image and the patches, defaulting to real_t.
     *
     * TODO add description
     */
    template <typename data_t>
    class Patchifier
    {
    public:
        Patchifier(const VolumeDescriptor& imageShape, index_t blockSize, index_t stride);

        DataContainer<data_t> im2patches(const DataContainer<data_t>& image);

        DataContainer<data_t> patches2im(const DataContainer<data_t>& patches);

    private:
        IndexVector_t getImageIdx(index_t patchNr);
        /*
            nPatchesRow = (1 + (nCols - blocksize) / stride)

            y = patchNr / nPatchesRow
            x = patchNr % nPatchesRow
        */

        VolumeDescriptor _imageShape;
        index_t _blockSize;
        index_t _stride;
        index_t _nPatchesRow;
        index_t _nPatchesCol;
    };

} // namespace elsa
