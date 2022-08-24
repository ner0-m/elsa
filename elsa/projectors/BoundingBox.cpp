#include "BoundingBox.h"

#include <stdexcept>

#include "Error.h"

namespace elsa
{
    BoundingBox::BoundingBox(const IndexVector_t& volDims)
        : _dim(volDims.size()),
          _min(RealVector_t::Zero(_dim)),
          _max(volDims.template cast<real_t>())
    {
    }

    BoundingBox::BoundingBox(const RealVector_t& min, const RealVector_t& max)
        : _dim(min.size()), _min(min), _max(max)
    {
    }

    index_t BoundingBox::dim() const
    {
        return _dim;
    }

    RealVector_t BoundingBox::center() const
    {
        return (_max - _min).array() / 2;
    }

    RealVector_t& BoundingBox::min()
    {
        return _min;
    }

    const RealVector_t& BoundingBox::min() const
    {
        return _min;
    }

    RealVector_t& BoundingBox::max()
    {
        return _max;
    }

    const RealVector_t& BoundingBox::max() const
    {
        return _max;
    }

    void BoundingBox::translateMin(const real_t& t)
    {
        this->min().array() += t;
    }

    void BoundingBox::translateMin(const RealVector_t& t)
    {
        this->min() += t;
    }

    void BoundingBox::translateMax(const real_t& t)
    {
        this->max().array() += t;
    }

    void BoundingBox::translateMax(const RealVector_t& t)
    {
        this->max() += t;
    }

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

    bool operator!=(const BoundingBox& box1, const BoundingBox& box2)
    {
        return !(box1 == box2);
    }

    std::ostream& operator<<(std::ostream& stream, const BoundingBox& aabb)
    {
        Eigen::IOFormat fmt(4, 0, ", ", ", ", "", "", "[", "]");
        stream << "AABB { min = " << aabb._min.format(fmt) << ", max = " << aabb._max.format(fmt)
               << " }";

        return stream;
    }

} // namespace elsa
