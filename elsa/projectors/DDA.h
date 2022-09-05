#pragma once

#include "elsaDefines.h"
#include "BoundingBox.h"

#include <Eigen/Geometry>
#include <limits>

namespace elsa
{
    /**
     * @brief Class implementing a voxel traversal of a volume (AABB) along a ray. The volume is
     * assumed to be a regular grid, this reduces the algorithm to a form of 'Digital differential
     * analyzer' (DDA). DDA algorithms are commonly known from computer graphics to rasterization
     * lines and other primitives.
     *
     * This traversal always proceeds along "long" voxel edges, it will never "jump diagonally".
     * The method is based on J. Amantides, A. Woo: A Fast Voxel Traversal Algorithm for Ray
     * Tracing.
     *
     * @author
     * - Tobias Lasser - initial code
     * - David Frank - major rewrite
     * - Maximilian Hornung - modularization
     * - Nikola Dinev - fixes
     */
    class DDA
    {
    public:
        /**
         * @brief Constructor for traversal, accepting bounding box and ray
         *
         * @param[in] aabb axis-aligned boundary box describing the volume
         * @param[in] r the ray to be traversed
         */
        DDA(const BoundingBox& aabb, const RealRay_t& r);

        /**
         * @brief Update the traverser status by taking the next traversal step in case the
         * indexToChange is unknown
         */
        void updateTraverse();

        /**
         * @brief Update the traverser status by taking the next traversal step in case the
         * indexToChange is known
         */
        void updateTraverse(const index_t& indexToChange);

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
        RealVector_t _entryPoint{_aabb.dim()};
        /// the step direction of the traverser
        RealVector_t _stepDirection{_aabb.dim()};
        /// the current position of the traverser in the aabb
        RealVector_t _currentPos{_aabb.dim()};
        /// the current maximum step parameter along the ray
        RealVector_t _tMax{_aabb.dim()};
        /// the step sizes for the step parameter along the ray
        RealVector_t _tDelta{_aabb.dim()};
        /// flag if traverser still in bounding box
        bool _isInAABB{false};
        /// the current step parameter exiting the current voxel
        real_t _tExit{0.0};

        /// constant vector containing epsilon
        const RealVector_t _EPS{
            RealVector_t(_aabb.dim()).setConstant(std::numeric_limits<real_t>::epsilon())};
        /// constant vector containing the maximum number
        const RealVector_t _MAX{
            RealVector_t(_aabb.dim()).setConstant(std::numeric_limits<real_t>::max())};

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

    /// @brief Simple iterator interface over the DDA traversal algorithm. This class provides an
    /// begin/end iterator pair/view interface to the `DDA` class. The iterator returned
    /// models an input iterator using an iterator/sentinel pair.
    class DDAView
    {
    public:
        /// Construct a view from a bounding box and ray
        DDAView(const BoundingBox& aabb, const RealRay_t& ray);

        struct DDASentinel;

        /// Iterator returned to the call side, responsible for properly calling the underlying
        /// `DDA` class.
        struct DDAIterator {
            using iterator_category = std::input_iterator_tag;
            using value_type = std::pair<real_t, IndexVector_t>;
            using difference_type = std::ptrdiff_t;
            using pointer = value_type*;
            using reference = value_type&;

            /// Construct iterator from reference to `DDA`
            DDAIterator(DDA& traverse);

            /// Dereference the iterator
            value_type operator*() const;

            /// Increment the iterator
            DDAIterator& operator++();

            /// Increment the iterator
            DDAIterator operator++(int);

            /// Compare iterators
            friend bool operator==(const DDAIterator& lhs, const DDAIterator& rhs);

            /// Compare iterators
            friend bool operator!=(const DDAIterator& lhs, const DDAIterator& rhs);

            /// Comparison to sentinel/end
            friend bool operator==(const DDAIterator& iter, DDASentinel);

        private:
            void advance();

            // TODO: Optimize this a bit, the objects are quite large and expensive to copy
            DDA& traverse_;
            real_t weight_;
            IndexVector_t current_;
            bool isInAABB_;
        };

        /// Sentinel indicating the end of the traversal.
        struct DDASentinel {
            friend bool operator!=(const DDAIterator& lhs, DDASentinel rhs);

            friend bool operator==(const DDASentinel& lhs, const DDAIterator& rhs);

            friend bool operator!=(const DDASentinel& lhs, const DDAIterator& rhs);
        };

        /// Return the begin iterator
        DDAIterator begin() { return {traverse}; }

        /// Return the end sentinel
        DDASentinel end() { return {}; }

    private:
        DDA traverse;
    };

    /// Create a view/range (i.e. an object `begin()` and `end()`) over the bounding box traversing
    /// along the given ray using DDA style traversal. A usage example can be found in the Siddons
    /// Method
    DDAView dda(const BoundingBox& aabb, const RealRay_t& ray);
} // namespace elsa
