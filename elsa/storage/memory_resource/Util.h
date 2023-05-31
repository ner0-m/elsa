#pragma once
#include <stddef.h>
#include <tuple>

namespace elsa::mr::util
{

    size_t computeRealSize(size_t requestedSize, size_t granularity);

    /// @return [realSize, blockSize] where
    ///     realSize:  the size of the block to return
    ///     blockSize: the required size for a block,
    ///                so that the allocation can be carved out of it.
    ///                blockSize >= realSize holds, to satisfy alignment guarantees.
    std::pair<size_t, size_t> computeSizeWithAlignment(size_t requestedSize,
                                                       size_t requestedAlignment,
                                                       size_t granularity);
} // namespace elsa::mr::util