#include "DrivingDirectionTraversal.h"

namespace elsa
{

    template <int dim>
    DrivingDirectionTraversal<dim>::DrivingDirectionTraversal(const BoundingBox& aabb,
                                                              const RealRay_t& r)
    {
        static_assert(dim == 2 || dim == 3);

        // --> calculate intersection parameter and if the volume is hit
        auto opt = intersectRay(aabb, r);

        r.direction().cwiseAbs().maxCoeff(&_drivingDirection);

        if (opt) {

            // --> get points at which they intersect
            _currentPos = r.pointAt(opt->_tmin);

            RealArray_t<dim> aabbMin = aabb.min();
            RealArray_t<dim> aabbMax = aabb.max();

            // --> because of floating point error it can happen, that values are out of
            // the bounding box, this can lead to errors
            _currentPos = (_currentPos < aabbMin).select(aabbMin, _currentPos);
            _currentPos = (_currentPos > aabbMax).select(aabbMax, _currentPos);

            RealArray_t<dim> exitPoint = r.pointAt(opt->_tmax);

            exitPoint = (exitPoint < aabbMin).select(aabbMin, exitPoint);
            exitPoint = (exitPoint > aabbMax).select(aabbMax, exitPoint);

            // the step is 1 in driving direction so that we step in increments of 1 along the grid
            // the other directions are set so that we step along the course of the ray
            _nextStep = r.direction() / abs(r.direction()[_drivingDirection]);

            // if the entryPoint is before .5 in driving direction, move _currentPos forward along
            // the ray to .5, otherwise move backward along the ray to .5
            real_t dist = _currentPos(_drivingDirection) - floor(_currentPos(_drivingDirection));
            _currentPos += _nextStep * (0.5f - dist) * _nextStep.sign()(_drivingDirection);

            // this is to make sure that along _drivingDirection, _currentPos is exactly .5 after
            // the decimal point
            _currentPos(_drivingDirection) = floor(_currentPos(_drivingDirection)) + 0.5f;

            // number of steps is the distance between exit and entry along the driving direction
            _numSteps = static_cast<index_t>(
                ceil(abs(exitPoint(_drivingDirection) - _currentPos(_drivingDirection))));

            _intersectionLength = _nextStep.matrix().norm();
        } else {
            _stepCount = _numSteps;
            return;
        }
    }

    template <int dim>
    void DrivingDirectionTraversal<dim>::updateTraverse()
    {
        _currentPos += _nextStep;
        _stepCount++;
    }

    template <int dim>
    IndexArray_t<dim> DrivingDirectionTraversal<dim>::getCurrentVoxel() const
    {
        return (_currentPos).floor().template cast<index_t>();
    }

    template <int dim>
    IndexArray_t<dim> DrivingDirectionTraversal<dim>::getCurrentVoxelFloor() const
    {
        return (_currentPos - 0.5f).floor().template cast<index_t>();
    }

    template <int dim>
    IndexArray_t<dim> DrivingDirectionTraversal<dim>::getCurrentVoxelCeil() const
    {
        return (_currentPos - 0.5f).ceil().template cast<index_t>();
    }

    template <int dim>
    index_t DrivingDirectionTraversal<dim>::getDrivingDirection() const
    {
        return _drivingDirection;
    }

    template <int dim>
    bool DrivingDirectionTraversal<dim>::isInBoundingBox() const
    {
        return (_stepCount < _numSteps);
    }

    template <int dim>
    real_t DrivingDirectionTraversal<dim>::getIntersectionLength() const
    {
        return _intersectionLength;
    }

    template <int dim>
    RealArray_t<dim> DrivingDirectionTraversal<dim>::getCurrentPos() const
    {
        return _currentPos;
    }

    template class DrivingDirectionTraversal<2>;
    template class DrivingDirectionTraversal<3>;
} // namespace elsa
