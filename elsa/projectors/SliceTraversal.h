#pragma once

#include "Intersection.h"
#include "elsaDefines.h"
#include "BoundingBox.h"
#include "Logger.h"
#include "Assertions.h"

#include <Eigen/Geometry>
#include <cassert>
#include <optional>
#include <cmath>
#include <utility>
#include <variant>

// TODO: remove
#include <iostream>

namespace elsa
{
    /// Strong type to distinguish transformation of points vs transformation of vectors
    template <class data_t>
    struct Point {
        explicit Point(Vector_t<data_t> point) : point_(std::move(point)) {}

        Vector_t<data_t> point_;
    };

    /// Strong type to distinguish transformation of points vs transformation of vectors
    template <class data_t>
    struct Vec {
        explicit Vec(Vector_t<data_t> vec) : vec_(std::move(vec)) {}

        Vector_t<data_t> vec_;
    };

    /**
     * @brief Represent a transformation, which will transform any point into the
     * traversal coordinate system.
     *
     * In the traversal coordinate system, the reference direction - form which it is constructed -
     * is transformed, such that the leading coefficient (i.e. the coefficient of largest absolute
     * value), is in the positive x-direction (i.e. the first component)
     *
     * The algorithm to determine the rotation in a 2D case is quite simple: Given the vector
     * \f$\begin{bmatrix} x & y \end{bmatrix}\f$,
     * first determine the leading axis, by finding the index of the component with maximum
     * absolute value, i.e. \f$\text{axis} = \text{maxAxis} (\lvert x \rvert, \lvert y \rvert) \f$
     * (in this case, the max returns "x", or "y" for our use cases instead of the actual value).
     * Assume `axis` is either `"x"` or `"y"` and `leadingCoeff` stores corresponding value
     * to the leading axis (but it's signed!). Then in pseudocode the algorithm determines the
     * rotation the following way:
     *
     * ```python
     * if axis == "x" and maxCoeff >= 0:
     *     return rotate_by(0)
     * elif axis == "x" and maxCoeff <= 0:
     *     return rotate_by(180)
     * elif axis == "y" and maxCoeff >= 0:
     *     return rotate_by(90)
     * elif axis == "y" and maxCoeff <= 0:
     *     return rotate_by(270)
     * ```
     *
     * To extent it to 3D one further decision has to be made, do we only rotate, such that the
     * leading direction is in the x direction, or do we rotate, such that y is the second largest
     * direction and z is the smallest.
     *
     * TODO: It might be nice to move this to a separate file and module, to wrap some of Eigens
     * transformation stuff, such that it's usable for dynamic cases, as we have here. This
     * might be nice to have for the Geometry class as well, but not for now.
     */
    class TransformToTraversal
    {
    private:
        static RealMatrix_t create2DRotation(const real_t leadingCoeff,
                                             const index_t leadingAxisIndex);

        static RealMatrix_t createRotation(RealRay_t ray);

        static RealMatrix_t createTransformation(const RealRay_t& ray,
                                                 const RealVector_t& centerOfRotation);

        RealMatrix_t transformation_;
        RealVector_t translation_;

    public:
        TransformToTraversal(const RealRay_t& ray, const RealVector_t& centerOfRotation);

        RealRay_t toTraversalCoordinates(RealRay_t ray) const;

        BoundingBox toTraversalCoordinates(BoundingBox aabb) const;

        RealMatrix_t transformation() const;

        RealMatrix_t invTransformation() const;

        RealMatrix_t rotation() const;

        RealMatrix_t invRotation() const;

        RealVector_t translation() const;

        RealMatrix_t linear() const;

        RealVector_t operator*(const Point<real_t>& point) const;

        RealVector_t operator*(const Vec<real_t>& vec) const;
    };

    /**
     * @brief Traverse a volume along the direction of a given ray. Each step it will advance one
     * slice further in the direction of the ray. The traversal visits voxels at the center planes,
     * i.e. it always evaluates at the center of voxel in the plane of leading direction.
     *
     * This is a slightly modified version of the algorithm of Amanatides & Woo's "A Fast Voxel
     * Traversal Algorithm". The reference visits each and every single voxel along the ray.
     * However, for this case we don't need that.
     *
     * Given a bounding box of the volume and a ray to traverse the volume, one can simply call:
     *
     * ```cpp
     * for(auto [pos, voxel, t] = SliceTraversal(aabb, ray)) {
     *     // pos: exact position on center plane of voxel
     *     // voxel: current voxel
     *     // t: position on the ray (i.e. pos = ray.origin() + t * ray.direction())
     * }
     * ```
     *
     * The number of visited voxels is known at construction, therefore moving along the ray is
     * basically decrementing a count. However, to return useful information, more bookkeeping is
     * needed.
     *
     * Dereferencing the iterator will return a small struct, which has information about the
     * current position in the volume, the current voxel and the t parameter of the ray.
     *
     * Note: When using voxel returned from iterating over the volume, it can happen that voxel
     * on the boundary of the volume are returned. This can trigger wrong behaviour for certain
     * applications and therefore should be handled with care.
     *
     * A note about the implementation: To ease the construction and handling, the incoming ray and
     * bounding box are transformed to a traversal internal coordinate space, in which the leading
     * direction lies in the positive x-axis. The 'world' space is partitioned in 4 quadrants based
     * on the direction. The first ranges for all direction from  with a polar angle in the range of
     * [-45°, 45°], the second (45°, 135°), the third [135°, 225°], and the fourth (225°, 315°).
     * Note that -45° = 315°.
     *
     * TODO:
     * - Make the iterator random access
     *
     * @author
     * - David Frank - initial code
     */
    class SliceTraversal
    {
    private:
        TransformToTraversal transformation_;

        /// TODO: Do we want to consider points that spawn inside the volume?
        index_t startIndex_{0};

        index_t endIndex_{0};

        /// t value for the first intersection of the ray and the bounding box
        real_t t_{0.0};

        /// t value to increment each iteration
        real_t tDelta_{0.0};

        RealRay_t ray_;

    public:
        /// The Dereference type of the iterator
        /// TODO: With C++20 support for proxy iterators is given, maybe this could change then
        struct IterValue {
            RealVector_t curPosition_;
            IndexVector_t curVoxel_;
            real_t t_;
        };

        /// Traversal iterator, models forward iterator, maybe this should actually be an input
        /// iterator due to the non reference type of the dereference type. IDK, as we use it, this
        /// works, but in the future this might should be different.
        struct Iter {
        public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = IterValue;
            using pointer = value_type*;
            using reference = value_type&;

        private:
            index_t pos_;
            RealRay_t ray_;
            real_t tDelta_;
            real_t t_;

        public:
            /// Construct iterator
            ///
            /// @param pos index position of the traversal, used mainly for comparing two iterators
            /// @param ray traversed ray used to compute exact position on dereference
            /// @param deltat increment of t each increment
            /// @param t position along the ray
            Iter(index_t pos, RealRay_t ray, real_t deltat, real_t t)
                : pos_(pos), ray_(ray), tDelta_(deltat), t_(t)
            {
            }

            /// Dereference iterator
            value_type operator*() const;

            /// Pre increment
            Iter& operator++();

            /// Post increment
            Iter operator++(int);

            /// Equal to operator with iterator
            friend bool operator==(const Iter& lhs, const Iter& rhs);

            /// Not equal to operator with iterator
            friend bool operator!=(const Iter& lhs, const Iter& rhs);
        };

        // Delete default construction
        SliceTraversal() = delete;

        // Construct traversal from bounding box and ray
        SliceTraversal(BoundingBox aabb, RealRay_t ray);

        // Get the first visited voxel
        Iter begin() const;

        // Get on past the end
        Iter end() const;

        // Get the index of the first visited voxel
        index_t startIndex() const;

        // Get the index of the one past the last visited voxel
        index_t endIndex() const;
    };
} // namespace elsa
