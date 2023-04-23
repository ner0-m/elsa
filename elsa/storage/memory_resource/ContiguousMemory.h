#pragma once
#include <cstring>
#include <atomic>

/*
 *  TODO:
 *  What of thrust uses this container. Are variables passed to thrust? Could issues arise?
 *  In MemoryResouce mem-copy, decide if it makes sense to move in userspace or do call
 *
 *  Implement default instance
 *  Insert usage of ContiguousStorage
 *  Rename MemoryResource to MemResInterface and MRRef to MemoryResource
 */

namespace elsa::mr
{
    class MRRef;

    /*
     *  Describes one memory resource interface.
     *  Should be bound once to an MRRef at construction and from
     *  there on only be passed through the MRRef wrapper.
     */
    class MemoryResource
    {
        friend class MRRef;

    private:
        std::atomic<size_t> _refCount;

    protected:
        MemoryResource();
        virtual ~MemoryResource() = default;

    public:
        virtual void* allocate(size_t size, size_t alignment) = 0;
        virtual bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize,
                               size_t newAlignment) = 0;
        virtual void deallocate(void* ptr, size_t size, size_t alignment) = 0;

        virtual void copyMemory(void* ptr, const void* src, size_t size) noexcept = 0;
        virtual void setMemory(void* ptr, const void* src, size_t stride,
                               size_t count) noexcept = 0;
        virtual void moveMemory(void* ptr, const void* src, size_t size) noexcept = 0;
    };

    /*
     *  Manages a reference to a Memory-Resource.
     *  The refernce count of a Memory-Resource is thread-safe.
     *  The MRRef object itself is not thread-safe.
     */
    class MRRef
    {
    private:
        MemoryResource* _resource = 0;

    public:
        MRRef();
        MRRef(const MRRef& r);
        MRRef(MRRef&& r) noexcept;
        ~MRRef();

    public:
        static MRRef MakeRef(MemoryResource* own);

    private:
        void _release();

    public:
        MRRef& operator=(const MRRef& r);
        MRRef& operator=(MRRef&& r) noexcept;
        MemoryResource* operator->();
        const MemoryResource* operator->() const;
        MemoryResource& operator*();
        const MemoryResource& operator*() const;
        bool operator==(const MRRef& r) const;
        bool operator!=(const MRRef& r) const;
        bool valid() const;
        void release();
    };

    /*
     *   DefaultInstance will at all times return a reference to the
     *   memory-resource from the last call to setDefaultInstance.
     *
     *   If setDefaultInstance has never been called, an instance
     *       of (TODO: insert here) will be instantiated.
     */
    void setDefaultInstance(const MRRef& r);
    MRRef defaultInstance();
} // namespace elsa::mr