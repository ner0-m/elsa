#include "TraverseAABBBranchless.h"
#include "Intersection.h"

namespace elsa
{
    template <int dim>
    TraverseAABBBranchless<dim>::TraverseAABBBranchless(
        const BoundingBox& aabb, const RealRay_t& r,
        IndexArray_t<dim> productOfCoefficientsPerDimension)
        : _productOfCoefficientsPerDimension{std::move(productOfCoefficientsPerDimension)}
    {
        static_assert(dim == 2 || dim == 3);
        _aabbMin = aabb.min();
        _aabbMax = aabb.max();

        // compute the first intersection
        const RealArray_t<dim> entryPoint = calculateAABBIntersections(r, aabb);
        if (!isInBoundingBox()) // early abort if necessary
            return;

        // constant array containing epsilon
        const RealArray_t<dim> EPS{
            RealArray_t<dim>().setConstant(std::numeric_limits<real_t>::epsilon())};

        // constant vector containing the maximum number
        const RealArray_t<dim> MAX{
            RealArray_t<dim>().setConstant(std::numeric_limits<real_t>::max())};

        // determine whether we go up/down or left/right
        initStepDirection(r.direction());

        // select the initial voxel (closest to the entry point)
        selectClosestVoxel(entryPoint);

        // initialize the step sizes for the step parameter
        initDelta(r.direction(), EPS, MAX);

        // initialize the maximum step parameters
        initT(r.direction(), EPS, MAX, entryPoint);
        initCurrentIndex();
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::updateTraverse()
    {
        // --> calculate the mask that masks out all but the lowest t values
        calcMask();

        // --> step into the current direction
        _currentPos += _mask.select(_stepDirection.template cast<real_t>(), 0);

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
        updateCurrentIndex();

        return (_tExit - tEntry);
    }

    template <int dim>
    bool TraverseAABBBranchless<dim>::isInBoundingBox() const
    {
        return _isInAABB;
    }

    template <int dim>
    IndexArray_t<dim> TraverseAABBBranchless<dim>::getCurrentVoxel() const
    {
        return _currentPos.template cast<index_t>();
    }

    template <int dim>
    index_t TraverseAABBBranchless<dim>::getCurrentIndex() const
    {
        return _currentIndex;
    }

    template <int dim>
    RealArray_t<dim>
        TraverseAABBBranchless<dim>::calculateAABBIntersections(const RealRay_t& r,
                                                                const BoundingBox& aabb)
    {
        RealArray_t<dim> entryPoint;
        // entry and exit point parameters
        real_t tmin;

        // --> calculate intersection parameter and if the volume is hit
        auto opt = intersectRay(aabb, r);

        if (opt) { // hit!
            _isInAABB = true;
            tmin = opt->_tmin;

            // --> get points at which they intersect
            entryPoint = r.pointAt(tmin);

            // --> because of floating point error it can happen, that values are out of
            // the bounding box, this can lead to errors
            entryPoint = (entryPoint < _aabbMin).select(_aabbMin, entryPoint);
            entryPoint = (entryPoint > _aabbMax).select(_aabbMax, entryPoint);
        }
        return entryPoint;
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::initStepDirection(const RealArray_t<dim>& rd)
    {
        _stepDirection = rd.sign().template cast<index_t>();
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::selectClosestVoxel(const RealArray_t<dim>& entryPoint)
    {
        RealArray_t<dim> lowerCorner = entryPoint.floor();
        lowerCorner = ((lowerCorner == entryPoint) && (_stepDirection < 0.0))
                          .select(lowerCorner - 1, lowerCorner);

        _currentPos = lowerCorner;

        // check if we are still inside the aabb
        if ((_currentPos >= _aabbMax).any() || (_currentPos < _aabbMin).any())
            _isInAABB = false;
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::initDelta(const RealArray_t<dim>& rd,
                                                const RealArray_t<dim>& EPS,
                                                const RealArray_t<dim>& MAX)
    {
        RealArray_t<dim> tdelta = _stepDirection.template cast<real_t>() / rd;

        _tDelta = (Eigen::abs(rd) > EPS).select(tdelta, MAX);
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::initT(const RealArray_t<dim>& rd, const RealArray_t<dim>& EPS,
                                            const RealArray_t<dim>& MAX,
                                            const RealArray_t<dim>& entryPoint)
    {
        RealArray_t<dim> T =
            (((rd > 0.0f).select(_currentPos + 1., _currentPos)) - entryPoint) / rd;

        _T = (Eigen::abs(rd) > EPS).select(T, MAX);
    }

    template <int dim>
    bool TraverseAABBBranchless<dim>::isCurrentPositionInAABB() const
    {
        return (_currentPos < _aabbMax).all() && (_currentPos >= _aabbMin).all();
    }

    template <int dim>
    void TraverseAABBBranchless<dim>::updateCurrentIndex()
    {
        _currentIndex +=
            (_stepDirection * _mask.select(_productOfCoefficientsPerDimension, 0)).sum();
    }
    template <int dim>
    void TraverseAABBBranchless<dim>::initCurrentIndex()
    {
        _currentIndex = (_productOfCoefficientsPerDimension * getCurrentVoxel()).sum();
    }

    template class TraverseAABBBranchless<2>;
    template class TraverseAABBBranchless<3>;
} // namespace elsa
