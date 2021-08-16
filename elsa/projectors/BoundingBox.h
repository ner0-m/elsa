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
    struct BoundingBox {
    public:
        /**
         * @brief Construct AABB of particular size
         *
         * @param[in] volumeDimensions the number of coefficients per volume dimension
         */
        BoundingBox(const IndexVector_t& volumeDimensions);

        /// the number of dimensions (2 or 3)
        index_t _dim;
        /// the front corner of the box
        RealVector_t _min{_dim};
        /// the back corner of the box
        RealVector_t _max{_dim};
        /// helper to convert coordinates to indices
        IndexVector_t _voxelCoordToIndexVector{_dim};
    };
} // namespace elsa
