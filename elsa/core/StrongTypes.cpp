#include "StrongTypes.h"

namespace elsa::geometry::detail
{
    // Explicit template instantiation
    template class StaticVectorTemplate<1, RealVector_t>;
    template class StaticVectorTemplate<2, RealVector_t>;
    template class StaticVectorTemplate<3, RealVector_t>;
    template class StaticVectorTemplate<4, RealVector_t>;

    template class StaticVectorTemplate<1, IndexVector_t>;
    template class StaticVectorTemplate<2, IndexVector_t>;
    template class StaticVectorTemplate<3, IndexVector_t>;
    template class StaticVectorTemplate<4, IndexVector_t>;

    template class GeometryData<1>;
    template class GeometryData<2>;
    template class GeometryData<3>;
    template class GeometryData<4>;
} // namespace elsa::geometry::detail