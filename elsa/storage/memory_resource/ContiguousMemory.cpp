#include "ContiguousMemory.h"

#include "PrimitiveResource.h"

#include <mutex>

namespace elsa::mr
{
    MemResInterface::MemResInterface()
    {
        _refCount = 1;
    }

    MemoryResource::MemoryResource(const MemoryResource& r)
    {
        if ((_resource = r._resource) != 0)
            ++_resource->_refCount;
    }
    MemoryResource::MemoryResource(MemoryResource&& r) noexcept
    {
        std::swap(_resource, r._resource);
    }
    MemoryResource::~MemoryResource()
    {
        _release();
    }

    MemoryResource MemoryResource::MakeRef(MemResInterface* own)
    {
        MemoryResource _out;
        _out._resource = own;
        return _out;
    }

    void MemoryResource::_release()
    {
        if (_resource != 0 && --_resource->_refCount == 0)
            delete _resource;
    }

    MemoryResource& MemoryResource::operator=(const MemoryResource& r)
    {
        _release();
        if ((_resource = r._resource) != 0)
            ++_resource->_refCount;
        return *this;
    }
    MemoryResource& MemoryResource::operator=(MemoryResource&& r) noexcept
    {
        std::swap(_resource, r._resource);
        return *this;
    }
    MemResInterface* MemoryResource::operator->()
    {
        return _resource;
    }
    const MemResInterface* MemoryResource::operator->() const
    {
        return _resource;
    }
    MemResInterface& MemoryResource::operator*()
    {
        return *_resource;
    }
    const MemResInterface& MemoryResource::operator*() const
    {
        return *_resource;
    }
    bool MemoryResource::operator==(const MemoryResource& r) const
    {
        return _resource == r._resource;
    }
    bool MemoryResource::operator!=(const MemoryResource& r) const
    {
        return _resource == r._resource;
    }
    bool MemoryResource::valid() const
    {
        return _resource != 0;
    }
    void MemoryResource::release()
    {
        _release();
    }
    MemResInterface* MemoryResource::get()
    {
        return _resource;
    }
    const MemResInterface* MemoryResource::get() const
    {
        return _resource;
    }

    /*
     *   Memory-Resource Singleton
     */
    static std::mutex mrSingletonLock;
    static MemoryResource mrSingleton;

    void setDefaultInstance(const MemoryResource& r)
    {
        if (!r.valid())
            return;
        std::lock_guard<std::mutex> _lock(mrSingletonLock);
        mrSingleton = r;
    }
    MemoryResource defaultInstance()
    {
        std::lock_guard<std::mutex> _lock(mrSingletonLock);
        if (!mrSingleton.valid())
            mrSingleton = MemoryResource::MakeRef(new PrimitiveResource());
        return mrSingleton;
    }
} // namespace elsa::mr
