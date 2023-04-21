#pragma once
#include "ContiguousMemory.h"

namespace elsa::mr
{
    class UniversalResource : public MemoryResource
    {
        void* allocate(size_t size, size_t alignment) override;
        void deallocate(void* ptr, size_t size, size_t alignment) override;
    };
} // namespace elsa::mr