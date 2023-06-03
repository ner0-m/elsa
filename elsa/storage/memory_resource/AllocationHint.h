#pragma once

#include <algorithm>
#include <memory>
#include <variant>
#include <optional>
#include "MemoryResource.h"

namespace elsa::mr::hint
{

    /// @brief Provides hints that help determine an appropriate memory resource. The hints are
    /// limited to the scope in which they are created and anything called from there. Hints are
    /// meant to be stack allocated. DO NOT HEAP ALLOCATE THEM OR STORE THEM IN MEMBERS!
    /// Note: the provided resource is shared by subsequent calls to elsa::mr::defaultResource().
    class ScopedMR
    {
    private:
        std::optional<MemoryResource> _previous;

    public:
        ScopedMR(const MemoryResource& hint);

        ~ScopedMR();

        void* operator new(size_t) = delete;
        void* operator new(size_t, size_t) = delete;
        void operator delete(void*) = delete;
        void operator delete(void*, size_t) = delete;

        ScopedMR(const ScopedMR&) = delete;
        ScopedMR(ScopedMR&&) = delete;
        ScopedMR& operator=(const ScopedMR&) = delete;
        ScopedMR& operator=(ScopedMR&&) = delete;
    };

    /// @brief Select a memory resource based on the current hint. The preferred way to obtain a
    /// MemoryResource is to call elsa::mr::defaultResource().
    /// @return std::nullopt if there is no hint in the current scope. An appropriate resource
    /// otherwise.
    std::optional<MemoryResource> selectMemoryResource();
} // namespace elsa::mr::hint