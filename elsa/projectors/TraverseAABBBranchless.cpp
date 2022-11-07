#include "TraverseAABBBranchless.h"
#include "Intersection.h"

namespace elsa
{
    template <int dim>
    TraverseAABBBranchless<dim>::TraverseAABBBranchless(const BoundingBox& aabb, const RealRay_t& r)
        : _aabb{aabb}
    {
        static_assert(dim == 2 || dim == 3);
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
        initT(r.direction());
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::updateTraverse()
    {
        // --> calculate the mask that masks out all but the lowest t values
        calcMask();

        // --> step into the current direction
        _currentPos += _mask.select(_stepDirection, 0);

        // --> update the T for next iteration
        _T += _mask.select(_tDelta, 0);

        // --> check if we are still in bounding box
        _isInAABB = isCurrentPositionInAABB();
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::calcMask()
    {
        if constexpr (dim == 2) {
            _mask[0] = (_T[0] <= _T[1]);
            _mask[1] = (_T[1] <= _T[0]);
        } else if constexpr (dim == 3) {
            _mask[0] = ((_T[0] <= _T[1]) && (_T[0] <= _T[2]));
            _mask[1] = ((_T[1] <= _T[0]) && (_T[1] <= _T[2]));
            _mask[2] = ((_T[2] <= _T[0]) && (_T[2] <= _T[1]));
        }
    }

    template <int dim>
    real_t TraverseAABBBranchless<dim>::updateTraverseAndGetDistance()
    {
        // --> compute the distance
        real_t tEntry = _tExit;
        _tExit = _T.minCoeff();

        // --> do the update
        updateTraverse();

        return (_tExit - tEntry);
    }

    template <int dim>
    bool TraverseAABBBranchless<dim>::isInBoundingBox() const
    {
        return _isInAABB;
    }

    template <int dim>
    Eigen::Array<index_t, dim, 1> TraverseAABBBranchless<dim>::getCurrentVoxel() const
    {
        return _currentPos.template cast<index_t>();
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::calculateAABBIntersections(const RealRay_t& r)
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
            _entryPoint = (_entryPoint < _aabb.min().array()).select(_aabb.min(), _entryPoint);
            _entryPoint = (_entryPoint > _aabb.max().array()).select(_aabb.max(), _entryPoint);
        }
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::initStepDirection(const RealArray_t& rd)
    {
        _stepDirection = rd.sign();
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::selectClosestVoxel()
    {
        RealArray_t lowerCorner = _entryPoint.floor();
        lowerCorner = ((lowerCorner == _entryPoint) && (_stepDirection < 0.0))
                          .select(lowerCorner - 1, lowerCorner);

        _currentPos = lowerCorner;

        // check if we are still inside the aabb
        if ((_currentPos >= _aabb.max().array()).any() || (_currentPos < _aabb.min().array()).any())
            _isInAABB = false;
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::initDelta(const RealArray_t& rd)
    {
        RealArray_t tdelta = _stepDirection / rd;

        _tDelta = (Eigen::abs(rd) > _EPS).select(tdelta, _MAX);
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::initT(const RealArray_t& rd)
    {
        RealArray_t T = (((rd > 0.0f).select(_currentPos + 1., _currentPos)) - _entryPoint) / rd;

        _T = (Eigen::abs(rd) > _EPS).select(T, _MAX);
    }

    template <int dim>
    bool TraverseAABBBranchless<dim>::isCurrentPositionInAABB() const
    {
        return (_currentPos < _aabb.max().array()).all()
               && (_currentPos >= _aabb.min().array()).all();
    }

    template class TraverseAABBBranchless<2>;
    template class TraverseAABBBranchless<3>;
} // namespace elsa
