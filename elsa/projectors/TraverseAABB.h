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
     * This traversal proceeds along "long" voxel edges, it will "jump diagonally" iff that "long"
     * voxel edge has the same value along more than one dimension.
     * The method is based on Xiao et al.:  Efficient implementation of the 3D-DDA ray traversal
     * algorithm on GPU and its application in radiation dose calculation.
     */
    template <int dim>
    class TraverseAABB
    {

    public:
        /**
         * @brief Constructor for traversal, accepting bounding box and ray
         *
         * @param[in] aabb axis-aligned boundary box describing the volume
         * @param[in] r the ray to be traversed
         */
        TraverseAABB(const BoundingBox& aabb, const RealRay_t& r,
                     IndexArray_t<dim> productOfCoefficientsPerDimension);

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
        IndexArray_t<dim> getCurrentVoxel() const;

        /**
         * @brief Return the index that corresponds to the current position
         */
        index_t getCurrentIndex() const;

    private:
        /// the step direction of the traverser
        IndexArray_t<dim> _stepDirection;
        /// the current position of the traverser in the aabb
        RealArray_t<dim> _currentPos;
        /// the current maximum step parameter along the ray
        RealArray_t<dim> _T;
        /// the step sizes for the step parameter along the ray
        RealArray_t<dim> _tDelta;
        /// flag if traverser still in bounding box
        bool _isInAABB{false};
        /// the current step parameter exiting the current voxel
        real_t _tExit{0.0};
        /// the current mask, with true for the directions in which we are stepping, and else fals
        BooleanArray_t<dim> _mask;
        /// result of aabb.min(), the lower corner of the aabb
        RealArray_t<dim> _aabbMin;
        /// result of aabb.max(), the upper corner of the aabb
        RealArray_t<dim> _aabbMax;
        /// the product of coefficients per dimension
        IndexArray_t<dim> _productOfCoefficientsPerDimension;
        /// the current index which corresponds to the current position
        index_t _currentIndex;

        /// compute the entry and exit points of ray r with the volume (aabb)
        RealArray_t<dim> calculateAABBIntersections(const RealRay_t& r, const BoundingBox& aabb);
        /// setup the step directions (which is basically the sign of the ray direction rd)
        void initStepDirection(const RealArray_t<dim>& rd);
        /// select the closest voxel to the entry point
        void selectClosestVoxel(const RealArray_t<dim>& entryPoint);
        /// setup the step sizes considering the ray direction rd
        void initDelta(const RealArray_t<dim>& rd, const RealArray_t<dim>& EPS,
                       const RealArray_t<dim>& MAX);
        /// setup the maximum step parameters considering the ray direction rd
        void initT(const RealArray_t<dim>& rd, const RealArray_t<dim>& EPS,
                   const RealArray_t<dim>& MAX, const RealArray_t<dim>& entryPoint);
        /// check if the current index is still in the bounding box
        bool isCurrentPositionInAABB() const;
        /// calculate the mask which masks out all but the minimal coefficients in _T.
        void calcMask();
        /// compute the index that corresponds to the initial position
        void initCurrentIndex();
        /// compute the index that corresponds to the current position
        void updateCurrentIndex();
    };
} // namespace elsa
