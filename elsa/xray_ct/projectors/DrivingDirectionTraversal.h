#pragma once

// core
#include <utility>
#include "elsaDefines.h"

// geometry
#include "Intersection.h"

namespace elsa
{
    /**
     *  @brief Special traversal wrapper for Joseph's method.
     */
    template <int dim>
    class DrivingDirectionTraversal
    {
    public:
        /**
         * @brief Constructor for traversal, accepting bounding box and ray
         *
         * @param[in] aabb axis-aligned boundary box describing the volume
         * @param[in] r the ray to be traversed
         */
        explicit DrivingDirectionTraversal(const BoundingBox& aabb, const RealRay_t& r);

        /**
         * @brief Update the traverser status by taking the next traversal step
         */
        void updateTraverse();

        /**
         * @brief Return whether the traversal is still in the bounding box
         *
         * @returns true if still in bounding box, false otherwise
         */
        bool isInBoundingBox() const;

        /**
         * @brief Get the driving direction
         *
         * The sampling points are chosen so that the direction which grows the quickest can be
         * ignored during interpolation, effectively reducing the number of operations needed for
         * the calculation of the interpolated value by half.
         *
         * @return index_t index of the driving direction
         */
        index_t getDrivingDirection() const;

        /**
         * @brief Get the intersection length for the current step
         *
         * @return real_t intersection length
         */
        real_t getIntersectionLength() const;

        /**
         * @brief Get the coordinates of the voxel which contains the current sampling point
         *
         * @return IndexVector_t coordinates of the voxel which contains the current sampling point
         */
        IndexArray_t<dim> getCurrentVoxel() const;

        /**
         * @brief get the current position in 'continuous' space
         * The position along the driving direction is always at .5, the other directions are set so
         * that currentPos sits on the ray.
         *
         * @return RealArray_t current position along the ray
         */
        RealArray_t<dim> getCurrentPos() const;

        /**
         * @brief get the floor of the current position as a voxel
         * Along the driving direction, the index is exactly the index corresponding to the current
         * position
         * Along the other directions, the indices are rounded down
         * @return IndexArray_t of the floor of the current position
         */
        IndexArray_t<dim> getCurrentVoxelFloor() const;

        /**
         * @brief get the ceil of the current position as a voxel
         * Along the driving direction, the index is exactly the index corresponding to the current
         * position
         * Along the other directions, the indices are rounded up
         * @return IndexArray_t of the ceil of the current position
         */
        IndexArray_t<dim> getCurrentVoxelCeil() const;

    private:
        /// the current position of the traverser in the aabb
        RealArray_t<dim> _currentPos;
        /// the step sizes for the next step along the ray
        RealArray_t<dim> _nextStep;
        /// length of ray segment currently being handled
        real_t _intersectionLength{0};
        /// index of direction for which no interpolation needs to be performed
        index_t _drivingDirection;
        /// counter for the number of steps already taken
        index_t _stepCount{0};
        /// the total number of steps to be taken
        index_t _numSteps;
    };
} // namespace elsa
