#include "ContiguousMemory.h"

#include <mutex>

namespace elsa::mr
{
    MemoryResource::MemoryResource()
    {
        _refCount = 1;
    }

    MRRef::MRRef()
    {
        _resource = 0;
    }
    MRRef::MRRef(const MRRef& r)
    {
        if ((_resource = r._resource) != 0)
            ++_resource->_refCount;
    }
    MRRef::MRRef(MRRef&& r) noexcept
    {
        std::swap(_resource, r._resource);
    }
    MRRef::~MRRef()
    {
        _release();
    }

    MRRef MRRef::MakeRef(MemoryResource* own)
    {
        MRRef _out;
        _out._resource = own;
        return _out;
    }

    void MRRef::_release()
    {
        if (_resource != 0 && --_resource->_refCount == 0)
            delete _resource;
    }

    MRRef& MRRef::operator=(const MRRef& r)
    {
        _release();
        if ((_resource = r._resource) != 0)
            ++_resource->_refCount;
        return *this;
    }
    MRRef& MRRef::operator=(MRRef&& r) noexcept
    {
        std::swap(_resource, r._resource);
        return *this;
    }
    MemoryResource* MRRef::operator->()
    {
        return _resource;
    }
    const MemoryResource* MRRef::operator->() const
    {
        return _resource;
    }
    MemoryResource& MRRef::operator*()
    {
        return *_resource;
    }
    const MemoryResource& MRRef::operator*() const
    {
        return *_resource;
    }
    bool MRRef::operator==(const MRRef& r) const
    {
        return _resource == r._resource;
    }
    bool MRRef::operator!=(const MRRef& r) const
    {
        return _resource == r._resource;
    }
    bool MRRef::valid() const
    {
        return _resource != 0;
    }
    void MRRef::release()
    {
        _release();
    }

    /*
     *   Memory-Resource Singleton
     */
    static std::mutex mrSingletonLock;
    static MRRef mrSingleton;

    void setDefaultInstance(const MRRef& r)
    {
        if (!r.valid())
            return;
        std::lock_guard<std::mutex> _lock(mrSingletonLock);
        mrSingleton = r;
    }
    MRRef defaultInstance()
    {
        std::lock_guard<std::mutex> _lock(mrSingletonLock);
        if (!mrSingleton.valid())
            throw std::runtime_error("Not yet implemented");
        return mrSingleton;
    }
} // namespace elsa::mr