#include "BoundingBox.h"

#include <stdexcept>

namespace elsa
{
    BoundingBox::BoundingBox(const IndexVector_t& volumeDimensions)
    : _dim(volumeDimensions.size())
    {
        // sanity check
        if (volumeDimensions.size() < 2 || volumeDimensions.size() > 3)
            throw std::invalid_argument("BoundingBox: can only deal with the 2d/3d cases");

        _min.setZero();
        _max = volumeDimensions.template cast<real_t>();

        _voxelCoordToIndexVector[0] = 1;
        _voxelCoordToIndexVector[1] = volumeDimensions[1];
        if (_dim == 3)
            _voxelCoordToIndexVector[2] = volumeDimensions[2] * volumeDimensions[2];
    }

} // namespace elsa
