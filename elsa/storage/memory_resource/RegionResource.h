#pragma once

#include "MemoryResource.h"
#include <unordered_map>
#include <list>
#include <memory>

namespace elsa::mr
{
    namespace region_resource
    {
        static constexpr size_t BLOCK_GRANULARITY = 256;
    }

    class RegionResource;

    class RegionResourceConfig
    {
    private:
        friend class RegionResource;

        size_t regionSize;
        bool isAdaptive;

        constexpr RegionResourceConfig(size_t regionSize, bool adaptive)
            : regionSize{regionSize}, isAdaptive{adaptive}
        {
        }

    public:
        /// @brief Default configuration for a region resource with (hopefully) sensible defaults.
        /// @return Default configuration for a region resource.
        static constexpr RegionResourceConfig defaultConfig()
        {
            return RegionResourceConfig(static_cast<size_t>(1) << 31, true);
        }

        constexpr RegionResourceConfig& setRegionSize(size_t size)
        {
            regionSize = size;
            return *this;
        }

        constexpr RegionResourceConfig& setAdaptive(bool adaptive)
        {
            isAdaptive = adaptive;
            return *this;
        }
    };

    /// @brief Memory resource for the ContiguousStorage class.
    /// This is the most specialized and fastest memory resource, if used appropriately.
    /// As a region resource, memory freed to this resource can only be reused once ALL memory
    /// allocated from it is freed. Hence, this memory resource is well suited for a sawtooth
    /// allocation pattern, first doing all allocations, then freeing everything, then potentially
    /// repeating.
    /// Advantages: Extremely fast, allocation/deallocations perform only a few simple additions.
    /// Disadvantages: If used inappropriately, huge memory overhead. Also, move assignment
    /// between containers with different memory resources is more costly. Use it only if you are
    /// sure that memory allocations are your bottle-neck! If your algorithm works on huge data,
    /// allocations are probably not worth optimizing.
    class RegionResource : public MemResInterface
    {
    private:
        MemoryResource _upstream;

        RegionResourceConfig _config;

        void* _basePtr;

        void* _endPtr;

        void* _bumpPtr;

        size_t _allocatedSize;

    public:
        RegionResource(const RegionResource& other) = delete;

        RegionResource& operator=(const RegionResource& other) = delete;

        RegionResource(RegionResource&& other) noexcept = delete;

        RegionResource& operator=(RegionResource&& other) noexcept = delete;

    protected:
        RegionResource(const MemoryResource& upstream,
                       const RegionResourceConfig& config = RegionResourceConfig::defaultConfig());

        ~RegionResource();

    public:
        static MemoryResource
            make(const MemoryResource& upstream = globalResource(),
                 const RegionResourceConfig& config = RegionResourceConfig::defaultConfig());

        void* allocate(size_t size, size_t alignment) override;

        void deallocate(void* ptr, size_t size, size_t alignment) noexcept override;

        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) noexcept override;
    };
} // namespace elsa::mr
