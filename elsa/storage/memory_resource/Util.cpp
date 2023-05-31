#include "Util.h"

#include "Assertions.h"
#include "BitUtil.h"
#include "elsaDefines.h"

size_t elsa::mr::util::computeRealSize(size_t requestedSize, size_t granularity)
{
    // never return null! allways allocate some space
    if (requestedSize == 0) {
        ++requestedSize;
    }

    size_t realSize = (requestedSize + granularity - 1) & ~(granularity - 1);
    if (unlikely(realSize < requestedSize)) {
        throw std::bad_alloc();
    }
    return realSize;
}

std::pair<size_t, size_t> elsa::mr::util::computeSizeWithAlignment(size_t requestedSize,
                                                                   size_t requestedAlignment,
                                                                   size_t granularity)
{
    if (!isPowerOfTwo(requestedAlignment)) {
        throw std::bad_alloc();
    }

    // find best-fitting non-empty bin
    size_t realSize = computeRealSize(requestedSize, granularity);

    // this overflow check is probably unnecessary, since the log is already compared
    // against the max block size

    // minimal size of the free block to carve the allocation out of. must be enough for to
    // contain an aligned allocation
    size_t blockSize;
    if (requestedAlignment <= granularity) {
        blockSize = realSize;
    } else {
        blockSize = realSize + requestedAlignment;
        if (unlikely(blockSize < realSize)) {
            throw std::bad_alloc();
        }
    }
    return {realSize, blockSize};
}