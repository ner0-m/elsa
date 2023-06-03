#pragma once
#include "ContiguousMemory.h"
#include "Logger.h"

#define ELSA_MR_LOGGING_RESOURCE_STRINGIFY(x) #x

namespace elsa::mr
{
    /// @brief Logs all allocations and deallocations to the wrapped resource.
    template <typename T>
    class LoggingResource : public T
    {
    private:
        std::shared_ptr<spdlog::logger> _logger;
        // does not include failed allocations
        long int _timeSpentInResource{0};

    protected:
        template <typename... Ts>
        LoggingResource(Ts... args);

        ~LoggingResource();

    public:
        /// @brief Creates a SynchedResource wrapping a back-end resource, which performs the actual
        /// allocations.
        /// @param ...args Parameters passed to the constructor of the wrapped resource.
        /// @return A MemoryResource encapsulationg the SynchedResource
        template <typename... Ts>
        static MemoryResource make(Ts&&... args);

        // A brief note on exception safety: during logging, spdlog should not throw
        // (https://github.com/gabime/spdlog/wiki/Error-handling)

        /// @brief Allocates from the wrapped resource, logging the requested size and alignment,
        /// and the resulting address. allocation.
        void* allocate(size_t size, size_t alignment) override;
        /// @brief Passes the pointer along to the wrapped resource for deallocation, logging
        /// address, size and alignment.
        void deallocate(void* ptr, size_t size, size_t alignment) noexcept override;
        /// @brief Tries to have the wrapped resource resize the allocation, logging the parameters
        /// and result.
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) override;
    };

    template <typename T>
    inline void* LoggingResource<T>::allocate(size_t size, size_t alignment)
    {
        try {
            auto start = std::chrono::system_clock::now();
            void* ptr = T::allocate(size, alignment);
            auto stop = std::chrono::system_clock::now();
            auto nanoSeconds =
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            _timeSpentInResource += nanoSeconds;
            _logger->info("({}) {}::allocate(size: 0x{:x}, alignment: 0x{:x}) = {} in {}ns",
                          reinterpret_cast<void*>(this), typeid(T).name(), size, alignment, ptr,
                          nanoSeconds);
            return ptr;
        } catch (const std::bad_alloc& e) {
            _logger->warn(
                "({}) {}::allocate(size: 0x{:x}, alignment: 0x{:x}) -> std::bad_alloc (out "
                "of heap or resource managed memory)",
                reinterpret_cast<void*>(this), typeid(T).name(), size, alignment);
            throw;
        } catch (...) {
            _logger->error(
                "({}) {}::allocate(size: 0x{:x}, alignment: 0x{:x}) -> UNEXPECTED EXCEPTION "
                "THROWN! This is a bug in resource {} or its upstream resource.",
                reinterpret_cast<void*>(this), typeid(T).name(), size, alignment, typeid(T).name());
            throw;
        }
    }

    template <typename T>
    inline void LoggingResource<T>::deallocate(void* ptr, size_t size, size_t alignment) noexcept
    {
        auto start = std::chrono::system_clock::now();
        T::deallocate(ptr, size, alignment);
        auto stop = std::chrono::system_clock::now();
        auto nanoSeconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        _timeSpentInResource += nanoSeconds;
        _logger->info("({}) {}::deallocate(ptr: {}, size: 0x{:x}, alignment: 0x{:x}) in {}ns",
                      reinterpret_cast<void*>(this), typeid(T).name(), ptr, size, alignment,
                      nanoSeconds);
    }

    template <typename T>
    inline bool LoggingResource<T>::tryResize(void* ptr, size_t size, size_t alignment,
                                              size_t newSize)
    {
        auto start = std::chrono::system_clock::now();
        bool resized = T::tryResize(ptr, size, alignment, newSize);
        auto stop = std::chrono::system_clock::now();
        auto nanoSeconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        _timeSpentInResource += nanoSeconds;
        _logger->info("({}) {}::tryResize(ptr: {}, size: 0x{:x}, alignment: 0x{:x}, newSize: "
                      "0x{:x}) = {} in {}ns",
                      reinterpret_cast<void*>(this), typeid(T).name(), ptr, size, alignment,
                      newSize, resized, nanoSeconds);
        return resized;
    }

    template <typename T>
    template <typename... Ts>
    inline LoggingResource<T>::LoggingResource(Ts... args)
        : T{args...}, _logger{Logger::get("elsa::mr::LoggingResource")}
    {
    }

    template <typename T>
    inline LoggingResource<T>::~LoggingResource()
    {
        _logger->info("({}) {}: Total time spent: {}ns", reinterpret_cast<void*>(this),
                      typeid(T).name(), _timeSpentInResource);
    }

    template <typename T>
    template <typename... Ts>
    inline MemoryResource LoggingResource<T>::make(Ts&&... args)
    {
        return MemoryResource::MakeRef(new LoggingResource<T>(std::forward<Ts>(args)...));
    }
} // namespace elsa::mr
