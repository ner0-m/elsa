#include "MemoryResource.h"

#include "UniversalResource.h"
#include "AllocationHint.h"

#include <mutex>

namespace elsa::mr
{
    /*
     *   Memory-Resource Singleton
     */
    static std::mutex mrSingletonLock;
    static MemoryResource mrSingleton;

    void setGlobalResource(const MemoryResource& r)
    {
        if (!r)
            return;
        std::lock_guard<std::mutex> _lock(mrSingletonLock);
        mrSingleton = r;
    }
    MemoryResource globalResource()
    {
        std::lock_guard<std::mutex> _lock(mrSingletonLock);
        if (!mrSingleton)
            mrSingleton = UniversalResource::make();
        return mrSingleton;
    }
    MemoryResource defaultResource()
    {
        std::optional<MemoryResource> resource = hint::selectMemoryResource();
        if (resource) {
            return std::move(*resource);
        } else {
            return globalResource();
        }
    }
    bool isBaselineMRSet()
    {
        return bool(mrSingleton);
    }

    StorageType storageType() {
#ifdef ELSA_CUDA_ENABLED
        return StorageType::CUDAManaged;
#else
        return StorageType::Host;
#endif
    }
} // namespace elsa::mr
