#include "ContiguousMemory.h"

namespace elsa::mr
{
    MemoryResource* defaultInstance()
    {
        return nullptr;
    }

    MemoryResource::MemoryResource()
    {
        _refCount = 1;
    }

    void MemoryResource::addRef()
    {
        _refCount += 1;
    }

    void MemoryResource::releaseRef()
    {
        if (--_refCount == 0) {
            delete this;
        }
    }
} // namespace elsa::mr