#include "SphericalHarmonicsDescriptor.h"

namespace elsa
{
    SphericalHarmonicsDescriptor::SphericalHarmonicsDescriptor(size_t maxDegree, SYMMETRY symmetry)
        : DataDescriptor(
            ((symmetry == regular)
                 ? (maxDegree + 1) * (maxDegree + 1)
                 : (symmetry == odd ? (maxDegree + 2) * (maxDegree / 2 + 1) : (maxDegree + 1))
                       * (maxDegree / 2 + 1))
            * IndexVector_t::Ones(1)),
          _symmetry(symmetry),
          _maxDegree(maxDegree)
    {
        if (symmetry == odd)
            assert(maxDegree % 2);
        if (symmetry == even)
            assert(!(maxDegree % 2));
    }

    SphericalHarmonicsDescriptor* SphericalHarmonicsDescriptor::cloneImpl() const
    {
        return new SphericalHarmonicsDescriptor(_maxDegree, _symmetry);
    }

    index_t SphericalHarmonicsDescriptor::getMaxDegree() const
    {
        return _maxDegree;
    }

    SphericalHarmonicsDescriptor::SYMMETRY SphericalHarmonicsDescriptor::getSymmetry() const
    {
        return _symmetry;
    }

} // namespace elsa