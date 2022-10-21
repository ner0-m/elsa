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
        BoundingBox(const IndexVector_t& volumeDimensions);

        BoundingBox(const RealVector_t& min, const RealVector_t& max);

        index_t dim() const;

        RealVector_t center() const;

        /// Return a reference to the minimum point of the bounding box
        RealVector_t& min();

        /// Return a reference to the minimum point of the bounding box
        const RealVector_t& min() const;

        /// Return a reference to the maximum point of the bounding box
        RealVector_t& max();

        /// Return a reference to the maximum point of the bounding box
        const RealVector_t& max() const;

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
    };

} // namespace elsa
