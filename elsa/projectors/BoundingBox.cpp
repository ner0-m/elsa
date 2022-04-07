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
            _voxelCoordToIndexVector[2] = volumeDimensions[2] * volumeDimensions[2];
    }

    RealVector_t BoundingBox::center() const { return (_max - _min).array() / 2; }

    void BoundingBox::recomputeBounds()
    {
        RealVector_t min = _min.cwiseMin(_max);
        RealVector_t max = _min.cwiseMax(_max);

        _min = min;
        _max = max;
    }

    bool operator==(const BoundingBox& box1, const BoundingBox& box2)
    {
        return box1._min == box2._min && box1._max == box2._max;
    }

    bool operator!=(const BoundingBox& box1, const BoundingBox& box2) { return !(box1 == box2); }

    std::ostream& operator<<(std::ostream& stream, const BoundingBox& aabb)
    {
        Eigen::IOFormat fmt(4, 0, ", ", ", ", "", "", "[", "]");
        stream << "AABB { min = " << aabb._min.format(fmt) << ", max = " << aabb._max.format(fmt)
               << " }";

        return stream;
    }

} // namespace elsa
