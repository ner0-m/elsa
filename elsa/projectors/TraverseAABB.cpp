#include "TraverseAABB.h"
#include "Intersection.h"

namespace elsa
{
    TraverseAABB::TraverseAABB(const BoundingBox& aabb, const RealRay_t& r) : _aabb{aabb}
    {
        // compute the first intersection
        calculateAABBIntersections(r);
        if (!isInBoundingBox()) // early abort if necessary
            return;

        // determine whether we go up/down or left/right
        initStepDirection(r.direction());

        // select the initial voxel (closest to the entry point)
        selectClosestVoxel();

        // initialize the step sizes for the step parameter
        initDelta(r.direction());

        // initialize the maximum step parameters
        initMax(r.direction());
    }

    void TraverseAABB::updateTraverse()
    {
        // --> select the index that has lowest t value
        index_t indexToChange;
        _tMax.minCoeff(&indexToChange);

        // --> step into the current direction
        _currentPos(indexToChange) += _stepDirection(indexToChange);

        // --> update the tMax for next iteration
        _tMax(indexToChange) += _tDelta(indexToChange);

        // --> check if we are still in bounding box
        _isInAABB = isCurrentPositionInAABB(indexToChange);
    }

    real_t TraverseAABB::updateTraverseAndGetDistance()
    {
        // --> select the index that has lowest t value
        index_t indexToChange;
        _tMax.minCoeff(&indexToChange);

        // --> compute the distance
        real_t tEntry = _tExit;
        _tExit = _tMax(indexToChange);

        // --> do the update
        updateTraverse();

        return (_tExit - tEntry);
    }

    bool TraverseAABB::isInBoundingBox() const { return _isInAABB; }

    IndexVector_t TraverseAABB::getCurrentVoxel() const
    {
        return _currentPos.template cast<IndexVector_t::Scalar>();
    }

    void TraverseAABB::calculateAABBIntersections(const RealRay_t& r)
    {
        // entry and exit point parameters
        real_t tmin;

        // --> calculate intersection parameter and if the volume is hit
        auto opt = Intersection::withRay(_aabb, r);

        if (opt) { // hit!
            _isInAABB = true;
            tmin = opt->_tmin;

            // --> get points at which they intersect
            _entryPoint = r.pointAt(tmin);

            // --> because of floating point error it can happen, that values are out of
            // the bounding box, this can lead to errors
            _entryPoint =
                (_entryPoint.array() < _aabb._min.array()).select(_aabb._min, _entryPoint);
            _entryPoint =
                (_entryPoint.array() > _aabb._max.array()).select(_aabb._max, _entryPoint);
        }
    }

    void TraverseAABB::initStepDirection(const RealVector_t& rd)
    {
        _stepDirection = rd.array().sign();
    }

    void TraverseAABB::selectClosestVoxel()
    {
        RealVector_t lowerCorner = _entryPoint.array().floor();
        lowerCorner =
            ((lowerCorner.array() == _entryPoint.array()) && (_stepDirection.array() < 0.0))
                .select(lowerCorner.array() - 1, lowerCorner);

        _currentPos = lowerCorner;

        // check if we are still inside the aabb
        if ((_currentPos.array() >= _aabb._max.array()).any()
            || (_currentPos.array() < _aabb._min.array()).any())
            _isInAABB = false;
    }

    void TraverseAABB::initDelta(const RealVector_t& rd)
    {
        RealVector_t tdelta = _stepDirection.template cast<real_t>().cwiseQuotient(rd);

        _tDelta = (Eigen::abs(rd.array()) > _EPS.array()).select(tdelta, _MAX);
    }

    void TraverseAABB::initMax(const RealVector_t& rd)
    {
        RealVector_t tmax =
            (((rd.array() > 0.0f).select(_currentPos.array() + 1., _currentPos)).matrix()
             - _entryPoint)
                .cwiseQuotient(rd)
                .matrix();

        _tMax = (Eigen::abs(rd.array()) > _EPS.array()).select(tmax, _MAX);
    }

    bool TraverseAABB::isCurrentPositionInAABB(index_t index) const
    {
        return _currentPos(index) < _aabb._max(index) && _currentPos(index) >= _aabb._min(index);
    }

} // namespace elsa
