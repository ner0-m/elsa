#pragma once

#include "elsaDefines.h"
#include "BoundingBox.h"

#include <Eigen/Geometry>
#include <limits>

namespace elsa
{
    /**
     * @brief Class implementing a voxel traversal of a volume (AABB) along a ray.
     *
     * @author Tobias Lasser - initial code
     * @author David Frank - major rewrite
     * @author Maximilian Hornung - modularization
     * @author Nikola Dinev - fixes
     *
     * This traversal always proceeds along "long" voxel edges, it will never "jump diagonally".
     * The method is based on J. Amantides, A. Woo: A Fast Voxel Traversal Algorithm for Ray
     * Tracing.
     */
    class TraverseAABB
    {
    public:
        /**
         * @brief Constructor for traversal, accepting bounding box and ray
         *
         * @param[in] aabb axis-aligned boundary box describing the volume
         * @param[in] r the ray to be traversed
         */
        TraverseAABB(const BoundingBox& aabb, const RealRay_t& r);

        /**
         * @brief Update the traverser status by taking the next traversal step
         */
        void updateTraverse();

        /**
         * @brief Update the traverser status by taking the next traversal step, return the distance
         * of step
         *
         * @returns the distance of the step taken
         */
        real_t updateTraverseAndGetDistance();

        /**
         * @brief Return whether the traversal is still in the bounding box
         *
         * @returns true if still in bounding box, false otherwise
         */
        bool isInBoundingBox() const;

        /**
         * @brief Return the current voxel in the bounding box
         *
         * @returns coordinate vector
         */
        IndexVector_t getCurrentVoxel() const;

    private:
        /// the volume / axis-aligned bounding box
        BoundingBox _aabb;
        /// the entry point parameters of the ray in the aabb
        RealVector_t _entryPoint{_aabb._dim};
        /// the step direction of the traverser
        RealVector_t _stepDirection{_aabb._dim};
        /// the current position of the traverser in the aabb
        RealVector_t _currentPos{_aabb._dim};
        /// the current maximum step parameter along the ray
        RealVector_t _tMax{_aabb._dim};
        /// the step sizes for the step parameter along the ray
        RealVector_t _tDelta{_aabb._dim};
        /// flag if traverser still in bounding box
        bool _isInAABB{false};
        /// the current step parameter exiting the current voxel
        real_t _tExit{0.0};

        /// constant vector containing epsilon
        const RealVector_t _EPS{
            RealVector_t(_aabb._dim).setConstant(std::numeric_limits<real_t>::epsilon())};
        /// constant vector containing the maximum number
        const RealVector_t _MAX{
            RealVector_t(_aabb._dim).setConstant(std::numeric_limits<real_t>::max())};

        /// compute the entry and exit points of ray r with the volume (aabb)
        void calculateAABBIntersections(const RealRay_t& r);
        /// setup the step directions (which is basically the sign of the ray direction rd)
        void initStepDirection(const RealVector_t& rd);
        /// select the closest voxel to the entry point
        void selectClosestVoxel();
        /// setup the step sizes considering the ray direction rd
        void initDelta(const RealVector_t& rd);
        /// setup the maximum step parameters considering the ray direction rd
        void initMax(const RealVector_t& rd);
        /// check if the current index is still in the bounding box
        bool isCurrentPositionInAABB(index_t index) const;
    };
} // namespace elsa
