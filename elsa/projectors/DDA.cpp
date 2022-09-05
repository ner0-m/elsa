#include "DDA.h"
#include "Intersection.h"

namespace elsa
{
    // ------------------------------------------
    // Implementation of DDA
    DDA::DDA(const BoundingBox& aabb, const RealRay_t& r) : _aabb{aabb}
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

    void DDA::updateTraverse()
    {
        // --> select the index that has lowest t value
        index_t indexToChange;
        _tMax.minCoeff(&indexToChange);

        updateTraverse(indexToChange);
    }

    void DDA::updateTraverse(const index_t& indexToChange)
    {
        // --> step into the current direction
        _currentPos(indexToChange) += _stepDirection(indexToChange);

        // --> update the tMax for next iteration
        _tMax(indexToChange) += _tDelta(indexToChange);

        // --> check if we are still in bounding box
        _isInAABB = isCurrentPositionInAABB(indexToChange);
    }

    real_t DDA::updateTraverseAndGetDistance()
    {
        // --> select the index that has lowest t value
        index_t indexToChange;
        _tMax.minCoeff(&indexToChange);

        // --> compute the distance
        real_t tEntry = _tExit;
        _tExit = _tMax(indexToChange);

        // --> do the update
        updateTraverse(indexToChange);

        return (_tExit - tEntry);
    }

    bool DDA::isInBoundingBox() const
    {
        return _isInAABB;
    }

    IndexVector_t DDA::getCurrentVoxel() const
    {
        return _currentPos.template cast<IndexVector_t::Scalar>();
    }

    void DDA::calculateAABBIntersections(const RealRay_t& r)
    {
        // entry and exit point parameters
        real_t tmin;

        // --> calculate intersection parameter and if the volume is hit
        auto opt = intersectRay(_aabb, r);

        if (opt) { // hit!
            _isInAABB = true;
            tmin = opt->_tmin;

            // --> get points at which they intersect
            _entryPoint = r.pointAt(tmin);

            // --> because of floating point error it can happen, that values are out of
            // the bounding box, this can lead to errors
            _entryPoint =
                (_entryPoint.array() < _aabb.min().array()).select(_aabb.min(), _entryPoint);
            _entryPoint =
                (_entryPoint.array() > _aabb.max().array()).select(_aabb.max(), _entryPoint);
        }
    }

    void DDA::initStepDirection(const RealVector_t& rd)
    {
        _stepDirection = rd.array().sign();
    }

    void DDA::selectClosestVoxel()
    {
        RealVector_t lowerCorner = _entryPoint.array().floor();
        lowerCorner =
            ((lowerCorner.array() == _entryPoint.array()) && (_stepDirection.array() < 0.0))
                .select(lowerCorner.array() - 1, lowerCorner);

        _currentPos = lowerCorner;

        // check if we are still inside the aabb
        if ((_currentPos.array() >= _aabb.max().array()).any()
            || (_currentPos.array() < _aabb.min().array()).any())
            _isInAABB = false;
    }

    void DDA::initDelta(const RealVector_t& rd)
    {
        RealVector_t tdelta = _stepDirection.template cast<real_t>().cwiseQuotient(rd);

        _tDelta = (Eigen::abs(rd.array()) > _EPS.array()).select(tdelta, _MAX);
    }

    void DDA::initMax(const RealVector_t& rd)
    {
        RealVector_t tmax =
            (((rd.array() > 0.0f).select(_currentPos.array() + 1., _currentPos)).matrix()
             - _entryPoint)
                .cwiseQuotient(rd)
                .matrix();

        _tMax = (Eigen::abs(rd.array()) > _EPS.array()).select(tmax, _MAX);
    }

    bool DDA::isCurrentPositionInAABB(index_t index) const
    {
        return _currentPos(index) < _aabb.max()(index) && _currentPos(index) >= _aabb.min()(index);
    }

    // ------------------------------------------
    // Implementation of DDAView
    DDAView::DDAView(const BoundingBox& aabb, const RealRay_t& ray) : traverse(aabb, ray) {}

    // ------------------------------------------
    // Implementation of DDAIterator
    DDAView::DDAIterator::DDAIterator(DDA& traverse) : traverse_(traverse)
    {
        // advance once, that is it initialized
        advance();
    }

    typename DDAView::DDAIterator::value_type DDAView::DDAIterator::operator*() const
    {
        return {weight_, current_};
    }

    DDAView::DDAIterator& DDAView::DDAIterator::operator++()
    {
        advance();
        return *this;
    }

    DDAView::DDAIterator DDAView::DDAIterator::operator++(int)
    {
        auto copy = *this;
        ++(*this);
        return copy;
    }

    bool operator==(const DDAView::DDAIterator& lhs, const DDAView::DDAIterator& rhs)
    {
        return lhs.current_ == rhs.current_;
    }

    bool operator!=(const DDAView::DDAIterator& lhs, const DDAView::DDAIterator& rhs)
    {
        return !(lhs == rhs);
    }

    void DDAView::DDAIterator::advance()
    {
        // Order is important! First get the voxel then move forward. Basically, the current
        // points to the previous position, this is needed to calculate the weight
        current_ = traverse_.getCurrentVoxel();
        isInAABB_ = traverse_.isInBoundingBox();
        weight_ = traverse_.updateTraverseAndGetDistance();
    }

    // ------------------------------------------
    // Implementation of DDASentinel
    bool operator==(const DDAView::DDAIterator& iter, DDAView::DDASentinel)
    {
        return !iter.isInAABB_;
    }

    bool operator!=(const DDAView::DDAIterator& lhs, DDAView::DDASentinel rhs)
    {
        return !(lhs == rhs);
    }

    bool operator==(const DDAView::DDASentinel& lhs, const DDAView::DDAIterator& rhs)
    {
        return rhs == lhs;
    }

    bool operator!=(const DDAView::DDASentinel& lhs, const DDAView::DDAIterator& rhs)
    {
        return !(lhs == rhs);
    }

    DDAView dda(const BoundingBox& aabb, const RealRay_t& ray)
    {
        return DDAView(aabb, ray);
    }
} // namespace elsa
