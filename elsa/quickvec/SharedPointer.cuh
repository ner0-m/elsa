#pragma once

#include "Helpers.cuh"
#include <iostream>
#include <cassert>
#include <cstddef>
#include <atomic>

namespace quickvec
{

    /**
     * @brief Reference counter for custom shared pointer implementation
     *
     * It inherits from the Managed class so it uses the CUDA unified memory per default. This
     * allows manipulation of the counter in device code.
     *
     * @author Jens Petit
     */
    class Counter
    {
    public:
        /// Constructor
        Counter() = default;

        /// Removed copy constructor
        Counter(Counter const&) = delete;

        /// Removed copy assignment constructor
        Counter& operator=(Counter const&) = delete;

        /// Move constructor
        Counter(Counter&&) = delete;

        /// Move assignment constructor
        Counter& operator=(Counter&&) = delete;

        /// Destructor
        ~Counter() = default;

        /// Reset counter value
        void reset() { _counter = 0; }

        /// Return counter value
        unsigned int get() { return _counter; }

        /// Overload pre increment
        void operator++() { _counter++; }

        /// Overload post increment
        void operator++(int) { _counter++; }

        /// Overload pre decrement
        void operator--() { _counter--; }

        /// Overload post decrement
        void operator--(int) { _counter--; }

    private:
        /// Current reference count, atomic to be thread-safe
        std::atomic<unsigned int> _counter{0};
    };

    /**
     * @brief Shared pointer using CUDA unified memory
     *
     * It uses a pointer to a counter which does the reference counting.
     *
     * @author Jens Petit
     *
     * @tparam data_t - pointed to type
     */
    template <class T>
    class SharedPointer
    {
    public:
        SharedPointer() = delete;

        /**
         * @brief allocates in unified memory size bytes
         *
         * @param size the size in bytes of the heap memory allocated
         */
        explicit SharedPointer(size_t size);

        /**
         * @brief uses existing pointer which was already allocated memory
         *
         * If a non-owning SharedPoinerCUDA is constructed, it will start with a reference counter
         * of two, to prevent it from freeing its memory.
         *
         * @param pointer location of memory, note that this has to be unified memory
         * @param owning indicates if this instance owns the memory
         */
        explicit SharedPointer(T* pointer = nullptr, bool owning = true);

        /// copy constructor, increases reference count by one
        SharedPointer(SharedPointer const& other);

        /// move constructor, keeps reference count the same
        SharedPointer(SharedPointer&& other);

        /// copy assignment, increases reference count by one
        SharedPointer& operator=(const SharedPointer& other);

        /// TODO: move assignment, currently still increases reference count by one, should keep it
        // the same
        SharedPointer& operator=(SharedPointer&& other);

        /// frees the CUDA memory if reference count is only 1
        ~SharedPointer();

        /// dereferencing operator
        T& operator*() const;

        /// access pointed to object
        T* operator->() const;

        /// the pointed to address should be available on both device and host code
        __host__ __device__ T* get() const { return _ptr; }

        /// cast to bool
        operator bool() const { return _ptr != nullptr; }

        /// returns the reference count
        unsigned int useCount() { return _counter->get(); }

    private:
        /// swaps member with other
        void swap(SharedPointer& other) noexcept;

        /// points to the raw data
        T* _ptr;

        /// reference counter
        Counter* _counter;
    };

} // namespace quickvec
