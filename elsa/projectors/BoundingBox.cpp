#include "BoundingBox.h"

#include <stdexcept>

#include "Error.h"

namespace elsa
{
    BoundingBox::BoundingBox(const IndexVector_t& volumeDimensions) : _dim(volumeDimensions.size())
    {
        // sanity check
        if (volumeDimensions.size() < 2 || volumeDimensions.size() > 3)
            throw InvalidArgumentError("BoundingBox: can only deal with the 2d/3d cases");

        _min.setZero();
        _max = volumeDimensions.template cast<real_t>();
    }

    BoundingBox::BoundingBox(const RealVector_t& boxMin, const RealVector_t& boxMax)
        : _dim{boxMin.size()}, _min{boxMin}, _max{boxMax}
    {
        // sanity check
        if (boxMin.size() != boxMax.size() || boxMin.size() < 2 || boxMin.size() > 3)
            throw std::invalid_argument("BoundingBox: can only deal with the 2d/3d cases");

        if ((boxMax.array() <= boxMin.array()).any())
            throw std::invalid_argument("BoundingBox: boxMin must be smaller than boxMan");
    }
} // namespace elsa
