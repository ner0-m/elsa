#pragma once
#include "ContiguousMemory.h"

namespace elsa::mr
{
    template <typename T>
    class ThrustElsaMRAdaptor : public MemResInterface
    {
    private:
        T _thrustMR;

    protected:
        template <typename... Ts>
        ThrustElsaMRAdaptor(Ts... args) : _thrustMR{args...} {};

    public:
        void* allocate(size_t size, size_t alignment) override
        {
            return _thrustMR.do_allocate(size, alignment).get();
        };
        void deallocate(void* ptr, size_t size, size_t alignment) noexcept override
        {
            return _thrustMR.do_deallocate((typename T::pointer)(ptr), size, alignment);
        }
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) override
        {
            return false;
        }

        template <typename... Ts>
        static MemoryResource make(Ts... args)
        {
            return MemoryResource::MakeRef(new ThrustElsaMRAdaptor<T>(args...));
        };
    };
} // namespace elsa::mr
