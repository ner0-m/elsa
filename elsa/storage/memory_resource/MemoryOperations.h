#pragma once

#include <cinttypes>

namespace elsa::mr::detail
{
    /* expects both pointers to be universally accessible */
    void memOpMove(void* ptr, const void* src, std::size_t size);

    /* expects ptr to be universally accessible and src to be optional */
    void memOpCopy(void* ptr, const void* src, std::size_t size, bool src_universal);

    /* expects ptr to be universally accessible */
    void memOpSet(void* ptr, const void* src, std::size_t count, std::size_t stride);

} // namespace elsa::mr::detail