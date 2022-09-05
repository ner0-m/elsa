#pragma once

#include "elsaDefines.h"

namespace elsa
{
    /**
     * @brief Helper struct defining a 2d or 3d axis-aligned bounding box (or in short: AABB).
     *
     * @author David Frank - initial code
     * @author Tobias Lasser - minor changes
     */
    class BoundingBox
    {
    public:
        /**
         * @brief Construct AABB of particular size
         *
         * @param[in] volumeDimensions the number of coefficients per volume dimension
         */
        BoundingBox(const IndexVector_t& volShape);

        BoundingBox(const IndexVector_t& volShape, const IndexVector_t& volStrides);

        BoundingBox(const RealVector_t& min, const RealVector_t& max, const IndexVector_t& strides);

        /// Return the dimension of the bounding box
        index_t dim() const;

        /// Return the center point of the bounding box
        RealVector_t center() const;

        /// Return a reference to the minimum point of the bounding box
        RealVector_t& min();

        /// Return a reference to the minimum point of the bounding box
        const RealVector_t& min() const;

        /// Return a reference to the maximum point of the bounding box
        RealVector_t& max();

        /// Return a reference to the maximum point of the bounding box
        const RealVector_t& max() const;

        /// Return a reference to the strides of the bounding box
        const IndexVector_t& strides() const;

        /// Return a reference to the strides of the bounding box
        IndexVector_t& strides();

        /// Adjust the minimum of the bounding box
        void translateMin(const real_t& v);

        /// Adjust the minimum of the bounding box
        void translateMin(const RealVector_t& v);

        /// Adjust the maximum of the bounding box
        void translateMax(const real_t& v);

        /// Adjust the maximum of the bounding box
        void translateMax(const RealVector_t& v);

        /// Expand bounding bounding box by vectors `min` and `max`
        void expand(const RealVector_t& min, const RealVector_t& max);

        /// Recompute the min and max bounds of the bounding box
        void recomputeBounds();

        friend bool operator==(const BoundingBox& box1, const BoundingBox& box2);

        friend bool operator!=(const BoundingBox& box1, const BoundingBox& box2);

        friend std::ostream& operator<<(std::ostream& stream, const BoundingBox& aabb);

    private:
        /// the number of dimensions (2 or 3)
        index_t _dim;
        /// the front corner of the box
        RealVector_t _min{_dim};
        /// the back corner of the box
        RealVector_t _max{_dim};
        /// strides of bounding box
        IndexVector_t _strides{_dim};
    };

} // namespace elsa
