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

        _voxelCoordToIndexVector[0] = 1;
        _voxelCoordToIndexVector[1] = volumeDimensions[1];
        if (_dim == 3)
            _voxelCoordToIndexVector[2] = volumeDimensions[1] * volumeDimensions[2];
    }

    BoundingBox::BoundingBox(const RealVector_t& boxMin, const RealVector_t& boxMax)
        : _dim{boxMin.size()}, _min{boxMin}, _max{boxMax}
    {
        // sanity check
        if (boxMin.size() != boxMax.size() || boxMin.size() < 2 || boxMin.size() > 3)
            throw std::invalid_argument("BoundingBox: can only deal with the 2d/3d cases");

        // TODO: this only makes sense for the case of a box with integer dimensionss
        _voxelCoordToIndexVector[0] = 1;
        _voxelCoordToIndexVector[1] = std::floor(boxMax[1] - boxMin[1]) + 1;
        _voxelCoordToIndexVector[2] =
            (std::floor(boxMax[2] - boxMax[2]) + 1) * _voxelCoordToIndexVector[1];
    }
} // namespace elsa
