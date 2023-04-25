#pragma once

#include <cstring>
#include <atomic>

/*
 *  TODO:
 *  What of thrust uses this container. Are variables passed to thrust? Could issues arise?
 *  In MemoryResouce mem-copy, decide if it makes sense to move in userspace or do call
 *
 *  Insert usage of ContiguousStorage
 *
 *  implement test cases
 */

namespace elsa::mr
{
    class MemoryResource;

    /*
     *  Describes one memory resource interface.
     *  Allows for polymorphic allocators.
     *  Should be bound once to a MemoryResource wrapper at construction and from
     *  there on only be passed through the MemoryResource wrapper.
     *
     *  allocate, copyMemory, moveMemory, setMemory may throw exceptions.
     *  deallocate and tryResize must not throw exceptions.
     */
    class MemResInterface
    {
        friend class MemoryResource;

    private:
        std::atomic<size_t> _refCount;

    protected:
        MemResInterface();
        virtual ~MemResInterface() = default;

    public:
        MemResInterface(MemResInterface&&) = delete;
        MemResInterface(const MemResInterface&) = delete;

    public:
        virtual void* allocate(size_t size, size_t alignment) = 0;
        virtual bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) = 0;
        virtual void deallocate(void* ptr, size_t size, size_t alignment) = 0;
        virtual void copyMemory(void* ptr, const void* src, size_t size) = 0;
        virtual void moveMemory(void* ptr, const void* src, size_t size) = 0;
        virtual void setMemory(void* ptr, const void* src, size_t stride, size_t count) = 0;
    };

    /*
     *  Manages a reference to a Memory-Resource interface.
     *  The refernce count of a Memory-Resource interface is thread-safe.
     *  The MemoryResource object itself is not thread-safe.
     */
    class MemoryResource
    {
    private:
        MemResInterface* _resource = nullptr;

    public:
        MemoryResource() = default;
        MemoryResource(const MemoryResource& r);
        MemoryResource(MemoryResource&& r) noexcept;
        ~MemoryResource();

    public:
        static MemoryResource MakeRef(MemResInterface* own);

    private:
        void _release();

    public:
        MemoryResource& operator=(const MemoryResource& r);
        MemoryResource& operator=(MemoryResource&& r) noexcept;
        MemResInterface* operator->();
        const MemResInterface* operator->() const;
        MemResInterface& operator*();
        const MemResInterface& operator*() const;
        bool operator==(const MemoryResource& r) const;
        bool operator!=(const MemoryResource& r) const;
        bool valid() const;
        void release();
        MemResInterface* get();
        const MemResInterface* get() const;
    };

    /*
     *  DefaultInstance will at all times return a reference to the
     *  memory-resource from the last call to setDefaultInstance.
     *
     *  If setDefaultInstance has never been called, an instance
     *      of PrimitiveResource will be instantiated.
     */
    void setDefaultInstance(const MemoryResource& r);
    MemoryResource defaultInstance();
} // namespace elsa::mr
