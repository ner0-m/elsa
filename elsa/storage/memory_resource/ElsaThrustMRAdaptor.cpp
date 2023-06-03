#include "ElsaThrustMRAdaptor.h"

elsa::mr::ElsaThrustMRAdaptor::pointer
    elsa::mr::ElsaThrustMRAdaptor::do_allocate(std::size_t bytes, std::size_t alignment)
{
    return pointer(_mr->allocate(bytes, alignment));
}

void elsa::mr::ElsaThrustMRAdaptor::do_deallocate(pointer p, std::size_t bytes,
                                                  std::size_t alignment)
{
    _mr->deallocate(p.get(), bytes, alignment);
}

bool elsa::mr::ElsaThrustMRAdaptor::do_is_equal(const memory_resource& other) const noexcept
{
    return this == &other;
}

elsa::mr::ElsaThrustMRAdaptor::ElsaThrustMRAdaptor() : _mr{elsa::mr::defaultResource()} {}
