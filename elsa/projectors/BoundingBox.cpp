#include "BoundingBox.h"

#include <stdexcept>

#include "Error.h"

namespace elsa
{
    BoundingBox::BoundingBox(const IndexVector_t& volumeDimensions)
        : BoundingBox(IndexVector_t(IndexVector_t::Zero(volumeDimensions.size())), volumeDimensions)
    {
    }

    template <typename Scalar>
    BoundingBox::BoundingBox(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& boxMin,
                             const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& boxMax)
        : _dim{boxMin.size()},
          _min{boxMin.template cast<real_t>()},
          _max{boxMax.template cast<real_t>()}
    {
        // sanity check
        if (boxMin.size() != boxMax.size())
            throw std::invalid_argument(
                "BoundingBox: start and end coordinate must have the same dimensionality");

        if (boxMin.size() < 2 || boxMin.size() > 3)
            throw std::invalid_argument("BoundingBox: can only deal with the 2d/3d cases");

        if ((boxMax.array() < boxMin.array()).any())
            throw std::invalid_argument("BoundingBox: boxMin cannot be larger than boxMan");
    }

    template BoundingBox::BoundingBox(const IndexVector_t&, const IndexVector_t&);
    template BoundingBox::BoundingBox(const RealVector_t&, const RealVector_t&);
} // namespace elsa
