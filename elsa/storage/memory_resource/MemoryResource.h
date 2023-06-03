#pragma once

#include <cstring>
#include <atomic>
#include <memory>

namespace elsa::mr
{
    /*
     *  Describes one memory resource interface.
     *  Allows for polymorphic allocators.
     *  Should be bound once to a MemoryResource wrapper at construction and from
     *  there on only be passed through the MemoryResource wrapper.
     *
     *  Deallocate and tryResize must not throw exceptions.
     */
    class MemResInterface
    {
    protected:
        MemResInterface() = default;
        virtual ~MemResInterface() = default;

    public:
        MemResInterface(MemResInterface&&) = delete;
        MemResInterface(const MemResInterface&) = delete;

    public:
        virtual void* allocate(size_t size, size_t alignment) = 0;
        virtual bool tryResize(void* ptr, size_t size, size_t alignment,
                               size_t newSize) noexcept = 0;
        virtual void deallocate(void* ptr, size_t size, size_t alignment) noexcept = 0;
    };
    using MemoryResource = std::shared_ptr<MemResInterface>;

    /// BaselineInstance will at all times return a reference to the
    /// memory-resource from the last call to setGlobalResource.
    ///
    /// If setGlobalResource has never been called, an instance
    ///     of UniversalResource will be instantiated.
    /// @param r Must be a synchronized memory resource!
    void setGlobalResource(const MemoryResource& r);
    /// Return the memory-resource currently set as global resource.
    /// The global-resource can be reassigned, see setGlobalResource.
    ///
    /// IMPORTANT! This means that there is no guarantee, that two calls
    /// to globalResource return the same memory-resource. Always make
    /// sure, pointers are deallocated in the memory-resource that allocated
    /// them.
    MemoryResource globalResource();
    bool isBaselineMRSet();

    /// Return the most recent thread-local set resource (when using hints/scoped-mr).
    /// If no hints/scopes have been applied, defaults to globalResource.
    MemoryResource defaultResource();
} // namespace elsa::mr
