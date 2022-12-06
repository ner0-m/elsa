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
    class DrivingDirectionTraversalBranchless
    {
    private:
        /**
         * @brief Possible stages of the traversal
         *
         * FIRST: currently handling the first sampling
         * INTERIOR: currently handling an interior point
         * LAST: currently handling the last sampling point
         *
         * If the ray only intersects a single pixel, the sole sampling point
         * is considered last.
         *
         */
        enum STAGE { FIRST, INTERIOR, LAST };

    public:
        /**
         * @brief Constructor for traversal, accepting bounding box and ray
         *
         * @param[in] aabb axis-aligned boundary box describing the volume
         * @param[in] r the ray to be traversed
         */
        explicit DrivingDirectionTraversalBranchless(const BoundingBox& aabb, const RealRay_t& r);

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
         * @brief Returns the fractional part of the current position wrt. the center of the
         * pixel/voxel
         *
         * For example: if the current position is (15.4, 21.7, 23.0), then the current voxel is
         * (15,21,23) and the fractional parts are (15.4, 21.7, 23.0) - (15.5, 21.5, 23.5) = (-0.1,
         * 0.2, -0.5)
         *
         * @return RealVector_t fractionals
         */
        const RealArray_t<dim>& getFractionals() const;

        /**
         * @brief Get direction to be ignored during interpolation
         *
         * The sampling points are chosen so that one of the directions can be ignored during
         * interpolation, effectively reducing the number of operations needed for the calculation
         * of the interpolated value by half. For interior points, the direction to be ignored is
         * always the main direction. For the entry and exit points it's the entry resp. exit
         * direction.
         *
         * @return int index of the direction to be ignored
         */
        int getIgnoreDirection() const;

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

    private:
        /// the volume / axis-aligned bounding box
        BoundingBox _aabb;
        /// the entry point parameters of the ray in the aabb
        RealArray_t<dim> _entryPoint;
        /// the exit point parameters of the ray in the aabb
        RealArray_t<dim> _exitPoint;
        /// the current position of the traverser in the aabb
        RealArray_t<dim> _currentPos;
        /// the step direction of the traverser
        RealArray_t<dim> _stepDirection;
        /// the step sizes for the next step along the ray
        RealArray_t<dim> _nextStep;
        /// the fractional parts of the current position coordinates (actually frac(_currentPos) -
        /// 0.5)
        RealArray_t<dim> _fractionals;
        /// flag if traverser still in bounding box
        bool _isInAABB{false};
        /// length of ray segment currently being handled
        real_t _intersectionLength{0};
        /// index of direction for which no interpolation needs to be performed
        int _ignoreDirection{-1};
        /// index of direction in which the ray exits the volume
        int _exitDirection{-1};
        /// stage at which the traversal currently is
        STAGE _stage{FIRST};

        /// initialize fractionals from a coordinate vector
        void initFractionals(const RealArray_t<dim>& currentPosition);
        /// setup the step directions (which is basically the sign of the ray direction rd)
        void initStepDirection(const RealVector_t& rd);
        /// compute the entry and exit points of ray r with the volume (aabb), returns the length of
        /// the intersection
        real_t calculateAABBIntersections(const RealRay_t& ray);
        /// select the closest voxel to the entry point (considering the current position)
        void selectClosestVoxel(const RealArray_t<dim>& currentPosition);
        /// check if the current index is still in the bounding box
        bool isCurrentPositionInAABB(index_t index) const;
        /// advances the traversal algorithm to the first sampling point
        void moveToFirstSamplingPoint(const RealVector_t& rd, real_t intersectionLength);
    };
} // namespace elsa
