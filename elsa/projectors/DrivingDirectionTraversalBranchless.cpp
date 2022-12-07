#include "DrivingDirectionTraversalBranchless.h"

namespace elsa
{

    template <int dim>
    DrivingDirectionTraversalBranchless<dim>::DrivingDirectionTraversalBranchless(
        const BoundingBox& aabb, const RealRay_t& r)
    {
        static_assert(dim == 2 || dim == 3);
        _aabbMin = aabb.min();
        _aabbMax = aabb.max();

        initStepDirection(r.direction());

        RealRay_t rt(r.origin(), r.direction());

        // determinge length of entire intersection with AABB
        real_t intersectionLength = calculateAABBIntersections(rt, aabb);

        if (!isInBoundingBox())
            return;

        selectClosestVoxel(_entryPoint);

        // exit direction stored in _exitDirection
        (_exitPoint - (_exitPoint.floor() + static_cast<real_t>(0.5)))
            .cwiseAbs()
            .maxCoeff(&_exitDirection);

        moveToFirstSamplingPoint(r.direction(), intersectionLength);

        // last pixel/voxel handled separately, reduce box dimensionality for easy detection
        int mainDir;
        rt.direction().cwiseAbs().maxCoeff(&mainDir);
        real_t prevBorder = std::floor(_exitPoint(mainDir));
        if (prevBorder == _exitPoint(mainDir) && rt.direction()[mainDir] > 0) {
            prevBorder--;
        }
        if (rt.direction()[mainDir] < 0) {
            _aabbMin[mainDir] = prevBorder + 1;
        } else {
            _aabbMax[mainDir] = prevBorder;
        }
    }

    template <int dim>
    void DrivingDirectionTraversalBranchless<dim>::updateTraverse()
    {
        _fractionals += _nextStep;
        for (int i = 0; i < dim; i++) {
            //*0.5 because a change of 1 spacing corresponds to a change of
            // fractionals from -0.5 to 0.5  0.5 or -0.5 = voxel border is still
            // ok
            if (std::abs(_fractionals(i)) > 0.5) {
                _fractionals(i) -= _stepDirection(i);
                _currentPos(i) += _stepDirection(i);
                // --> is the traverse algorithm still in the bounding box?
                _isInAABB = _isInAABB && isCurrentPositionInAABB(i);
            }
        }
        switch (_stage) {
            case FIRST:
                _nextStep.cwiseAbs().maxCoeff(&_ignoreDirection);
                if (_isInAABB) {
                    // now ignore main direction
                    _nextStep /= std::abs(_nextStep(_ignoreDirection));
                    _fractionals(_ignoreDirection) = 0;
                    _intersectionLength = _nextStep.matrix().norm();
                    _stage = INTERIOR;
                }
                [[fallthrough]];
            case INTERIOR:
                if (!_isInAABB) {
                    // revert to exit position and adjust values
                    real_t prevBorder = _nextStep(_ignoreDirection) < 0
                                            ? _aabbMin[_ignoreDirection]
                                            : _aabbMax[_ignoreDirection];
                    _nextStep.matrix().normalize();
                    _intersectionLength =
                        (_exitPoint[_ignoreDirection] - prevBorder) / (_nextStep[_ignoreDirection]);
                    // determine exit direction (the exit coordinate furthest from the center of the
                    // volume)
                    _ignoreDirection = _exitDirection;
                    // move to last sampling point
                    RealArray_t<dim> currentPosition =
                        _exitPoint - _intersectionLength * _nextStep / 2;
                    selectClosestVoxel(currentPosition);
                    initFractionals(currentPosition);
                    _isInAABB = true;
                    _stage = LAST;
                }
                break;
            case LAST:
                _isInAABB = false;
                break;
            default:
                break;
        }
    }

    template <int dim>
    const RealArray_t<dim>& DrivingDirectionTraversalBranchless<dim>::getFractionals() const
    {
        return _fractionals;
    }

    template <int dim>
    int DrivingDirectionTraversalBranchless<dim>::getIgnoreDirection() const
    {
        return _ignoreDirection;
    }

    template <int dim>
    void DrivingDirectionTraversalBranchless<dim>::initFractionals(
        const RealArray_t<dim>& currentPosition)
    {
        for (int i = 0; i < dim; i++)
            _fractionals(i) =
                static_cast<real_t>(std::abs(currentPosition(i)) - _currentPos(i) - 0.5);
    }

    template <int dim>
    void DrivingDirectionTraversalBranchless<dim>::moveToFirstSamplingPoint(
        const RealVector_t& rd, real_t intersectionLength)
    {
        // determine main direction
        rd.cwiseAbs().maxCoeff(&_ignoreDirection);

        // initialize _nextStep as the step for interior pixels
        _nextStep(_ignoreDirection) = _stepDirection(_ignoreDirection);
        for (int i = 0; i < dim; ++i) {
            if (i != _ignoreDirection) {
                // tDelta(i) is given in relation to tDelta(_ignoreDirection)
                _nextStep(i) = _nextStep(_ignoreDirection) * rd(i) / rd(_ignoreDirection);
            }
        }

        RealArray_t<dim> currentPosition = _entryPoint;
        // move to first sampling point
        if (std::abs(_entryPoint[_ignoreDirection] - _exitPoint[_ignoreDirection])
            <= static_cast<real_t>(1.)) {
            _stage = LAST;
            // use midpoint of intersection as the interpolation value
            currentPosition += rd.array() * intersectionLength / 2;
            _intersectionLength = intersectionLength;
        } else {
            // find distance to next plane orthogonal to main direction
            real_t nextBoundary = std::trunc(_currentPos(_ignoreDirection));
            if (_stepDirection(_ignoreDirection) > 0)
                nextBoundary += static_cast<real_t>(1.);

            real_t distToBoundary =
                (nextBoundary - _entryPoint[_ignoreDirection]) / rd[_ignoreDirection];

            currentPosition += rd.array() * distToBoundary / 2;
            _nextStep = _nextStep / 2 + _nextStep * (distToBoundary / (2 * _nextStep.matrix().norm()));
            _intersectionLength = distToBoundary;

            // determine entry direction (the entry coordinate furthest from the center of the
            // volume)
            (_entryPoint - (_entryPoint.floor() + static_cast<real_t>(0.5)))
                .cwiseAbs()
                .maxCoeff(&_ignoreDirection);
        }
        selectClosestVoxel(currentPosition);

        // init fractionals
        initFractionals(currentPosition);
    }

    template <int dim>
    inline bool
        DrivingDirectionTraversalBranchless<dim>::isCurrentPositionInAABB(index_t index) const
    {
        return _currentPos(index) < _aabbMax(index) && _currentPos(index) >= _aabbMin(index);
    }

    template <int dim>
    void DrivingDirectionTraversalBranchless<dim>::selectClosestVoxel(
        const RealArray_t<dim>& currentPosition)
    {
        RealArray_t<dim> lowerCorner = currentPosition.floor();
        lowerCorner = (((lowerCorner == currentPosition) && (_stepDirection < 0.0))
                       || currentPosition == _aabbMax)
                          .select(lowerCorner - 1, lowerCorner);

        // --> If ray is parallel and we are close, choose the next previous/next voxel
        _currentPos = lowerCorner;

        // check if we are still inside the aabb
        if ((_currentPos >= _aabbMax).any() || (_currentPos < _aabbMin).any())
            _isInAABB = false;
    }

    template <int dim>
    void DrivingDirectionTraversalBranchless<dim>::initStepDirection(const RealVector_t& rd)
    {
        _stepDirection = rd.array().sign();
    }

    template <int dim>
    real_t DrivingDirectionTraversalBranchless<dim>::calculateAABBIntersections(
        const RealRay_t& ray, const BoundingBox& aabb)
    {
        real_t tmin;

        // --> calculate intersection parameter and if the volume is hit
        auto opt = intersectRay(aabb, ray);

        if (opt) {
            _isInAABB = true;
            tmin = opt->_tmin;

            // --> get points at which they intersect
            _entryPoint = ray.pointAt(tmin);

            // --> because of floating point error it can happen, that values are out of
            // the bounding box, this can lead to errors
            _entryPoint = (_entryPoint < _aabbMin).select(_aabbMin, _entryPoint);
            _entryPoint = (_entryPoint > _aabbMax).select(_aabbMax, _entryPoint);

            // exit point stored in _exitPoint
            _exitPoint = ray.pointAt(opt->_tmax);
            _exitPoint = (_exitPoint < _aabbMin).select(_aabbMin, _exitPoint);
            _exitPoint = (_exitPoint > _aabbMax).select(_aabbMax, _exitPoint);

            return opt->_tmax - opt->_tmin;
        } else {
            return 0;
        }
    }

    template <int dim>
    bool DrivingDirectionTraversalBranchless<dim>::isInBoundingBox() const
    {
        return _isInAABB;
    }

    template <int dim>
    real_t DrivingDirectionTraversalBranchless<dim>::getIntersectionLength() const
    {
        return _intersectionLength;
    }

    template <int dim>
    IndexArray_t<dim> DrivingDirectionTraversalBranchless<dim>::getCurrentVoxel() const
    {
        return _currentPos.template cast<index_t>();
    }

    template class DrivingDirectionTraversalBranchless<2>;
    template class DrivingDirectionTraversalBranchless<3>;
} // namespace elsa
