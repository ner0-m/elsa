#include "TraverseAABBJosephsMethod.h"

namespace elsa
{

    TraverseAABBJosephsMethod::TraverseAABBJosephsMethod(const BoundingBox& aabb,
                                                         const RealRay_t& r)
        : _aabb{aabb}
    {
        initStepDirection(r.direction());

        RealRay_t rt(r.origin(), r.direction());

        // determinge length of entire intersection with AABB
        real_t intersectionLength = calculateAABBIntersections(rt);

        if (!isInBoundingBox())
            return;

        selectClosestVoxel(_entryPoint);

        // exit direction stored in _exitDirection
        (_exitPoint - (_exitPoint.array().floor() + static_cast<real_t>(0.5)).matrix())
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
            _aabb._min[mainDir] = prevBorder + 1;
        } else {
            _aabb._max[mainDir] = prevBorder;
        }
    }

    void TraverseAABBJosephsMethod::updateTraverse()
    {
        _fractionals += _nextStep;
        for (int i = 0; i < _aabb._dim; i++) {
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
                    _intersectionLength = _nextStep.norm();
                    _stage = INTERIOR;
                }
                [[fallthrough]];
            case INTERIOR:
                if (!_isInAABB) {
                    // revert to exit position and adjust values
                    real_t prevBorder = _nextStep(_ignoreDirection) < 0
                                            ? _aabb._min[_ignoreDirection]
                                            : _aabb._max[_ignoreDirection];
                    _nextStep.normalize();
                    _intersectionLength =
                        (_exitPoint[_ignoreDirection] - prevBorder) / (_nextStep[_ignoreDirection]);
                    // determine exit direction (the exit coordinate furthest from the center of the
                    // volume)
                    _ignoreDirection = _exitDirection;
                    // move to last sampling point
                    RealVector_t currentPosition = _exitPoint - _intersectionLength * _nextStep / 2;
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

    const RealVector_t& TraverseAABBJosephsMethod::getFractionals() const { return _fractionals; }
    int TraverseAABBJosephsMethod::getIgnoreDirection() const { return _ignoreDirection; }

    void TraverseAABBJosephsMethod::initFractionals(const RealVector_t& currentPosition)
    {
        for (int i = 0; i < _aabb._dim; i++)
            _fractionals(i) =
                static_cast<real_t>(std::abs(currentPosition(i)) - _currentPos(i) - 0.5);
    }

    void TraverseAABBJosephsMethod::moveToFirstSamplingPoint(const RealVector_t& rd,
                                                             real_t intersectionLength)
    {
        // determine main direction
        rd.cwiseAbs().maxCoeff(&_ignoreDirection);

        // initialize _nextStep as the step for interior pixels
        _nextStep(_ignoreDirection) = _stepDirection(_ignoreDirection);
        for (int i = 0; i < _aabb._dim; ++i) {
            if (i != _ignoreDirection) {
                // tDelta(i) is given in relation to tDelta(_ignoreDirection)
                _nextStep(i) = _nextStep(_ignoreDirection) * rd(i) / rd(_ignoreDirection);
            }
        }

        RealVector_t currentPosition = _entryPoint;
        // move to first sampling point
        if (std::abs(_entryPoint[_ignoreDirection] - _exitPoint[_ignoreDirection])
            <= static_cast<real_t>(1.)) {
            _stage = LAST;
            // use midpoint of intersection as the interpolation value
            currentPosition += rd * intersectionLength / 2;
            _intersectionLength = intersectionLength;
        } else {
            // find distance to next plane orthogonal to main direction
            real_t nextBoundary = std::trunc(_currentPos(_ignoreDirection));
            if (_stepDirection(_ignoreDirection) > 0)
                nextBoundary += static_cast<real_t>(1.);

            real_t distToBoundary =
                (nextBoundary - _entryPoint[_ignoreDirection]) / rd[_ignoreDirection];

            currentPosition += rd * distToBoundary / 2;
            _nextStep = _nextStep / 2 + _nextStep * (distToBoundary / (2 * _nextStep.norm()));
            _intersectionLength = distToBoundary;

            // determine entry direction (the entry coordinate furthest from the center of the
            // volume)
            (_entryPoint - (_entryPoint.array().floor() + static_cast<real_t>(0.5)).matrix())
                .cwiseAbs()
                .maxCoeff(&_ignoreDirection);
        }
        selectClosestVoxel(currentPosition);

        // init fractionals
        initFractionals(currentPosition);
    }

    inline bool TraverseAABBJosephsMethod::isCurrentPositionInAABB(index_t index) const
    {
        return _currentPos(index) < _aabb._max(index) && _currentPos(index) >= _aabb._min(index);
    }

    void TraverseAABBJosephsMethod::selectClosestVoxel(const RealVector_t& currentPosition)
    {
        RealVector_t lowerCorner = currentPosition.array().floor();
        lowerCorner =
            (((lowerCorner.array() == currentPosition.array()) && (_stepDirection.array() < 0.0))
             || currentPosition.array() == _aabb._max.array())
                .select(lowerCorner.array() - 1, lowerCorner);

        // --> If ray is parallel and we are close, choose the next previous/next voxel
        _currentPos = lowerCorner;

        // check if we are still inside the aabb
        if ((_currentPos.array() >= _aabb._max.array()).any()
            || (_currentPos.array() < _aabb._min.array()).any())
            _isInAABB = false;
    }

    RealVector_t floorFallback(const RealVector_t& v)
    {
        RealVector_t ret(v.rows());
        for (int i = 0; i < v.rows(); i++)
            ret(i) = std::floor(v(i));
        return ret;
    }

    void TraverseAABBJosephsMethod::initStepDirection(const RealVector_t& rd)
    {
        _stepDirection = rd.array().sign();
    }

    real_t TraverseAABBJosephsMethod::calculateAABBIntersections(const RealRay_t& ray)
    {
        real_t tmin;

        // --> calculate intersection parameter and if the volume is hit
        auto opt = Intersection::withRay(_aabb, ray);

        if (opt) {
            _isInAABB = true;
            tmin = opt->_tmin;

            // --> get points at which they intersect
            _entryPoint = ray.pointAt(tmin);

            // --> because of floating point error it can happen, that values are out of
            // the bounding box, this can lead to errors
            _entryPoint =
                (_entryPoint.array() < _aabb._min.array()).select(_aabb._min, _entryPoint);
            _entryPoint =
                (_entryPoint.array() > _aabb._max.array()).select(_aabb._max, _entryPoint);

            // exit point stored in _exitPoint
            _exitPoint = ray.pointAt(opt->_tmax);
            _exitPoint = (_exitPoint.array() < _aabb._min.array()).select(_aabb._min, _exitPoint);
            _exitPoint = (_exitPoint.array() > _aabb._max.array()).select(_aabb._max, _exitPoint);

            return opt->_tmax - opt->_tmin;
        } else {
            return 0;
        }
    }

    bool TraverseAABBJosephsMethod::isInBoundingBox() const { return _isInAABB; }

    real_t TraverseAABBJosephsMethod::getIntersectionLength() const { return _intersectionLength; }

    IndexVector_t TraverseAABBJosephsMethod::getCurrentVoxel() const
    {
        return _currentPos.template cast<index_t>();
    }

} // namespace elsa
