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
    template <int dim>
    class TraverseAABBBranchless
    {
        using IndexArray_t = Eigen::Array<index_t, dim, 1>;
        using RealArray_t = Eigen::Array<real_t, dim, 1>;
        using BooleanArray_t = Eigen::Array<bool, dim, 1>;

    public:
        /**
         * @brief Constructor for traversal, accepting bounding box and ray
         *
         * @param[in] aabb axis-aligned boundary box describing the volume
         * @param[in] r the ray to be traversed
         */
        TraverseAABBBranchless(const BoundingBox& aabb, const RealRay_t& r);

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
        IndexArray_t getCurrentVoxel() const;

    private:
        /// the step direction of the traverser
        RealArray_t _stepDirection;
        /// the current position of the traverser in the aabb
        RealArray_t _currentPos;
        /// the current maximum step parameter along the ray
        RealArray_t _T;
        /// the step sizes for the step parameter along the ray
        RealArray_t _tDelta;
        /// flag if traverser still in bounding box
        bool _isInAABB{false};
        /// the current step parameter exiting the current voxel
        real_t _tExit{0.0};
        /// the current mask, with true for the directions in which we are stepping, and else fals
        BooleanArray_t _mask;
        /// result of aabb.min(), the lower corner of the aabb
        RealArray_t _aabbMin;
        /// result of aabb.max(), the upper corner of the aabb
        RealArray_t _aabbMax;

        /// compute the entry and exit points of ray r with the volume (aabb)
        RealArray_t calculateAABBIntersections(const RealRay_t& r, const BoundingBox& aabb);
        /// setup the step directions (which is basically the sign of the ray direction rd)
        void initStepDirection(const RealArray_t& rd);
        /// select the closest voxel to the entry point
        void selectClosestVoxel(const RealArray_t& entryPoint);
        /// setup the step sizes considering the ray direction rd
        void initDelta(const RealArray_t& rd, const RealArray_t& EPS, const RealArray_t& MAX);
        /// setup the maximum step parameters considering the ray direction rd
        void initT(const RealArray_t& rd, const RealArray_t& EPS, const RealArray_t& MAX,
                   const RealArray_t& entryPoint);
        /// check if the current index is still in the bounding box
        bool isCurrentPositionInAABB() const;
        /// calculate the mask which masks out all but the minimal coefficients in _T.
        void calcMask();

    };
} // namespace elsa
