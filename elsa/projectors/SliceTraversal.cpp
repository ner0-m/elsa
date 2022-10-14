#include "SliceTraversal.h"
#include "Error.h"
#include "elsaDefines.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/fmt/ostr.h"

namespace elsa
{
    RealMatrix_t TransformToTraversal::create2DRotation(const real_t leadingCoeff,
                                                        const index_t leadingAxisIndex)
    {
        auto createRotationMatrix = [](real_t radian) {
            real_t c = std::cos(radian);
            real_t s = std::sin(radian);
            return RealMatrix_t({{c, -s}, {s, c}});
        };

        // TODO: Does a closed form solution exist?
        // Use conversion to polar coordinates
        if (leadingAxisIndex == 0 && leadingCoeff >= 0) {
            // Already identity, so do nothing
            return createRotationMatrix(0);
        } else if (leadingAxisIndex == 0 && leadingCoeff < 0) {
            // create a 2D 180° rotation matrix
            return createRotationMatrix(pi_t);
        } else if (leadingAxisIndex == 1 && leadingCoeff >= 0) {
            // create a 2D 270° rotation matrix counter clockwise
            return createRotationMatrix(3 * pi_t / 2);
        } else if (leadingAxisIndex == 1 && leadingCoeff <= 0) {
            // create a 2D 90° rotation matrix counter clockwise
            return createRotationMatrix(pi_t / 2);
        }
        return createRotationMatrix(0);
    }

    RealMatrix_t TransformToTraversal::create3DRotation(const real_t leadingCoeff,
                                                        const index_t leadingAxisIndex)
    {
        auto identity = []() { return RealMatrix_t({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}); };

        auto rotationX = [](real_t radian) {
            real_t c = std::cos(radian);
            real_t s = std::sin(radian);
            return RealMatrix_t({{1, 0, 0}, {0, c, -s}, {0, s, c}});
        };

        auto rotationY = [](real_t radian) {
            real_t c = std::cos(radian);
            real_t s = std::sin(radian);
            return RealMatrix_t({{c, 0, s}, {0, 1, 0}, {-s, 0, c}});
        };

        auto rotationZ = [](real_t radian) {
            real_t c = std::cos(radian);
            real_t s = std::sin(radian);
            return RealMatrix_t({{c, -s, 0}, {s, c, 0}, {0, 0, 1}});
        };

        // TODO: Does a closed form solution exist?
        // Use conversion to polar coordinates
        if (leadingAxisIndex == 0 && leadingCoeff >= 0) {
            // Already identity, so do nothing
            return identity();
        } else if (leadingAxisIndex == 0 && leadingCoeff < 0) {
            return rotationY(pi_t);
        } else if (leadingAxisIndex == 1 && leadingCoeff >= 0) {
            return rotationZ(-0.5 * pi_t);
        } else if (leadingAxisIndex == 1 && leadingCoeff <= 0) {
            return rotationZ(0.5 * pi_t);
        } else if (leadingAxisIndex == 2 && leadingCoeff >= 0) {
            return rotationY(0.5 * pi_t);
        } else if (leadingAxisIndex == 2 && leadingCoeff <= 0) {
            return rotationY(-0.5 * pi_t);
        }

        return identity();
    }

    RealMatrix_t TransformToTraversal::createRotation(RealRay_t ray)
    {
        // Get the leading axis, by absolute value
        index_t leadingAxisIndex = 0;
        ray.direction().array().abs().maxCoeff(&leadingAxisIndex);

        // Get the signed leading coefficient
        const auto leadingCoeff = ray.direction()(leadingAxisIndex);

        if (ray.dim() == 2) {
            return create2DRotation(leadingCoeff, leadingAxisIndex);
        } else if (ray.dim() == 3) {
            return create3DRotation(leadingCoeff, leadingAxisIndex);
        }
        throw Error("Can not create a {}-dimensional transformation for the slice traversal",
                    ray.dim());
    }

    RealMatrix_t TransformToTraversal::createTransformation(const RealRay_t& ray,
                                                            const RealVector_t& centerOfRotation)
    {
        const auto dim = ray.dim();

        // Setup translation
        RealMatrix_t translation(dim + 1, dim + 1);
        translation.setIdentity();
        translation.block(0, dim, dim, 1) = -centerOfRotation;

        // Setup rotation
        RealMatrix_t rotation(dim + 1, dim + 1);
        rotation.setIdentity();
        rotation.block(0, 0, dim, dim) = createRotation(ray);

        return rotation * translation;
    }

    TransformToTraversal::TransformToTraversal(const RealRay_t& ray,
                                               const RealVector_t& centerOfRotation)
        : transformation_(createTransformation(ray, centerOfRotation)),
          translation_(-centerOfRotation)
    {
    }

    RealRay_t TransformToTraversal::toTraversalCoordinates(RealRay_t ray) const
    {
        ray.origin() = *this * Point(ray.origin());
        ray.direction() = *this * Vec(ray.direction());
        return ray;
    }

    BoundingBox TransformToTraversal::toTraversalCoordinates(BoundingBox aabb) const
    {
        // Only translate, as the rotations are always 90, 180, 270 degrees,
        // TODO: this might be wrong, idk, what about non square things
        aabb.min() = *this * Point(aabb.min());
        aabb.max() = *this * Point(aabb.max());
        aabb.recomputeBounds();
        return aabb;
    }

    RealMatrix_t TransformToTraversal::transformation() const
    {
        return transformation_;
    }

    RealMatrix_t TransformToTraversal::invTransformation() const
    {
        return transformation().inverse();
    }

    RealMatrix_t TransformToTraversal::rotation() const
    {
        const auto dim = transformation_.rows();
        return transformation_.block(0, 0, dim - 1, dim - 1);
    }

    RealMatrix_t TransformToTraversal::invRotation() const
    {
        return rotation().transpose();
    }

    RealVector_t TransformToTraversal::translation() const
    {
        return translation_;
    }

    RealMatrix_t TransformToTraversal::linear() const
    {
        return rotation();
    }

    RealVector_t TransformToTraversal::operator*(const Point<real_t>& point) const
    {
        return (transformation() * point.point_.homogeneous()).hnormalized();
    }

    RealVector_t TransformToTraversal::operator*(const Vec<real_t>& vec) const
    {
        return linear() * vec.vec_;
    }

    SliceTraversal::SliceTraversal(BoundingBox aabb, RealRay_t ray)
        : transformation_(ray, aabb.max().array() / 2), ray_(ray)
    {
        // Transform ray and aabb to traversal coordinate space
        ray = transformation_.toTraversalCoordinates(ray);
        aabb = transformation_.toTraversalCoordinates(aabb);

        // TODO: We only want to shift in x direction by 0.5, not in y
        auto hit = intersectXPlanes(aabb, ray);

        // Default init is sufficient if we don't hit,  TODO: test that!
        if (hit) {
            const auto firstVoxel = ray.pointAt(hit->_tmin);
            const auto lastVoxel = ray.pointAt(hit->_tmax);

            endIndex_ = std::max<index_t>(
                1, static_cast<index_t>(std::ceil(lastVoxel[0] - firstVoxel[0] + 0.5f)));

            tDelta_ = 1 / ray.direction()[0];

            t_ = hit->_tmin;
        }
    }

    index_t SliceTraversal::leadingDirection() const
    {
        index_t idx = 0;
        ray_.direction().array().cwiseAbs().maxCoeff(&idx);
        return idx;
    }

    real_t SliceTraversal::t() const
    {
        return t_;
    }

    real_t SliceTraversal::tDelta() const
    {
        return tDelta_;
    }

    SliceTraversal::Iter SliceTraversal::begin() const
    {
        return {startIndex_, ray_.pointAt(t_), ray_.direction() * tDelta_};
    }

    SliceTraversal::Iter SliceTraversal::end() const
    {
        return {endIndex_};
    }

    index_t SliceTraversal::startIndex() const
    {
        return startIndex_;
    }

    index_t SliceTraversal::endIndex() const
    {
        return endIndex_;
    }

    SliceTraversal::Iter::value_type SliceTraversal::Iter::operator*() const
    {
        return cur_.template cast<index_t>();
    }

    SliceTraversal::Iter& SliceTraversal::Iter::operator++()
    {
        ++pos_;
        cur_ += dir_;
        return *this;
    }

    SliceTraversal::Iter SliceTraversal::Iter::operator++(int)
    {
        auto copy = *this;
        ++(*this);
        return copy;
    }

    bool operator==(const SliceTraversal::Iter& lhs, const SliceTraversal::Iter& rhs)
    {
        return lhs.pos_ == rhs.pos_;
    }

    bool operator!=(const SliceTraversal::Iter& lhs, const SliceTraversal::Iter& rhs)
    {
        return !(lhs == rhs);
    }

} // namespace elsa
