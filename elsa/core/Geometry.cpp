#include "Geometry.h"

#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

#include <iostream>

namespace elsa
{
    using namespace geometry;

    Geometry::Geometry(SourceToCenterOfRotation sourceToCenterOfRotation,
                       CenterOfRotationToDetector centerOfRotationToDetector, Radian angle,
                       VolumeData2D&& volData, SinogramData2D&& sinoData,
                       std::optional<Radian> fanAngle, PrincipalPointOffset offset,
                       RotationOffset2D centerOfRotOffset)
        : _objectDimension{2},
          _P{RealMatrix_t::Identity(2, 2 + 1)},
          _Pinv{RealMatrix_t::Identity(2 + 1, 2)},
          _K{RealMatrix_t::Identity(2, 2)},
          _R{RealMatrix_t::Identity(2, 2)},
          _invR{RealMatrix_t::Identity(2, 2)},
          _t{RealVector_t::Zero(2)},
          _S{RealMatrix_t::Identity(2 + 1, 2 + 1)},
          _C{RealVector_t::Zero(2)},
          _principalPoint{RealVector_t::Zero(2)},
          _fanAngle{0}
    {
        auto [volSpacing, volOrigin] = std::move(volData);
        auto [sinoSpacing, sinoOrigin] = std::move(sinoData);

        // setup rotation matrix _R
        real_t c = std::cos(angle);
        real_t s = std::sin(angle);
        _R << c, -s, s, c;

        // Inverse of rotation matrix is it's transpose
        _invR = _R.transpose();

        // setup scaling matrix _S
        // clang-format off
        _S << volSpacing[0],             0, 0,
                          0, volSpacing[1], 0, 
                          0,             0, 1;
        // clang-format on

        _t = _R * (-centerOfRotOffset.get() - volOrigin);
        _t[_objectDimension - 1] += sourceToCenterOfRotation;

        // set the intrinsic parameters _K
        real_t alpha = sinoSpacing[0];
        _K << (sourceToCenterOfRotation + centerOfRotationToDetector) / alpha,
            (sinoOrigin[0] / alpha + offset), 0, 1;

        buildMatrices();

        buildPrincipalPoint(volOrigin, sourceToCenterOfRotation + centerOfRotationToDetector);

        if (fanAngle)
            _fanAngle = *fanAngle;
        else
            calculateFanAngle(sinoOrigin);
    }

    Geometry::Geometry(SourceToCenterOfRotation sourceToCenterOfRotation,
                       CenterOfRotationToDetector centerOfRotationToDetector,
                       VolumeData3D&& volData, SinogramData3D&& sinoData, RotationAngles3D angles,
                       std::optional<Radian> fanAngle, PrincipalPointOffset2D offset,
                       RotationOffset3D centerOfRotOffset)
        : _objectDimension{3},
          _P{RealMatrix_t::Identity(3, 3 + 1)},
          _Pinv{RealMatrix_t::Identity(3 + 1, 3)},
          _K{RealMatrix_t::Identity(3, 3)},
          _R{RealMatrix_t::Identity(3, 3)},
          _invR{RealMatrix_t::Identity(3, 3)},
          _t{RealVector_t::Zero(3)},
          _S{RealMatrix_t::Identity(3 + 1, 3 + 1)},
          _C{RealVector_t::Zero(3)},
          _principalPoint{RealVector_t::Zero(3)},
          _fanAngle{0}
    {
        auto [volSpacing, volOrigin] = std::move(volData);
        auto [sinoSpacing, sinoOrigin] = std::move(sinoData);

        const real_t alpha = angles.alpha();
        const real_t beta = angles.beta();
        const real_t gamma = angles.gamma();

        // setup rotation matrix
        const real_t ca = std::cos(alpha);
        const real_t sa = std::sin(alpha);
        const real_t cb = std::cos(beta);
        const real_t sb = std::sin(beta);
        const real_t cg = std::cos(gamma);
        const real_t sg = std::sin(gamma);

        // YZY convention
        // clang-format off
        _R = (RealMatrix_t(3, 3) <<  cg, 0, sg, 
                                      0, 1,  0, 
                                    -sg, 0, cg).finished() // rotate around y
             * (RealMatrix_t(3, 3) << cb,-sb, 0, 
                                      sb, cb, 0,
                                       0,  0, 1).finished() // rotate around z
             * (RealMatrix_t(3, 3) << ca, 0, sa,
                                       0, 1,  0, 
                                     -sa, 0, ca).finished(); // rotate around x
        // clang-format on

        // Inverse of rotation matrix is it's transpose
        _invR = _R.transpose();

        // setup scaling matrix _S
        _S << volSpacing[0], 0, 0, 0, 0, volSpacing[1], 0, 0, 0, 0, volSpacing[2], 0, 0, 0, 0, 1;

        // setup translation
        _t = _R * (-centerOfRotOffset.get() - volOrigin);
        _t(_objectDimension - 1) += sourceToCenterOfRotation;

        // setup the intrinsic parameters _K
        const real_t alpha1 = sinoSpacing[0];
        const real_t alpha2 = sinoSpacing[1];

        // clang-format off
        _K << (sourceToCenterOfRotation + centerOfRotationToDetector) / alpha1, 0, sinoOrigin[0] / alpha1 + offset[0], 
               0, (sourceToCenterOfRotation + centerOfRotationToDetector) / alpha2, sinoOrigin[1] / alpha2 + offset[1], 
               0, 0, 1;
        // clang-format on

        buildMatrices();

        buildPrincipalPoint(volOrigin, sourceToCenterOfRotation + centerOfRotationToDetector);

        if (fanAngle)
            _fanAngle = *fanAngle;
        else
            calculateFanAngle(sinoOrigin);
    }

    Geometry::Geometry(real_t sourceToCenterOfRotation, real_t centerOfRotationToDetector,
                       VolumeData3D&& volData, SinogramData3D&& sinoData, const RealMatrix_t& R,
                       real_t px, real_t py, real_t centerOfRotationOffsetX,
                       real_t centerOfRotationOffsetY, real_t centerOfRotationOffsetZ)
        // : _objectDimension{3}
        : _objectDimension{3},
          _P{RealMatrix_t::Identity(3, 3 + 1)},
          _Pinv{RealMatrix_t::Identity(3 + 1, 3)},
          _K{RealMatrix_t::Identity(3, 3)},
          _R{R},
          _invR{R.transpose()},
          _t{RealVector_t::Zero(3)},
          _S{RealMatrix_t::Identity(3 + 1, 3 + 1)},
          _C{RealVector_t::Zero(3)},
          _principalPoint{RealVector_t::Zero(3)},
          _fanAngle{0}
    {
        // sanity check
        if (R.rows() != _objectDimension || R.cols() != _objectDimension)
            throw std::invalid_argument(
                "Geometry: 3D geometry requested with non-3D rotation matrix");

        auto [volSpacing, volOrigin] = std::move(volData);
        auto [sinoSpacing, sinoOrigin] = std::move(sinoData);

        // // setup scaling matrix _S
        _S << volSpacing[0], 0, 0, 0, 0, volSpacing[1], 0, 0, 0, 0, volSpacing[2], 0, 0, 0, 0, 1;

        // // setup the translation _t
        RealVector_t centerOfRotationOffset(_objectDimension);
        centerOfRotationOffset << centerOfRotationOffsetX, centerOfRotationOffsetY,
            centerOfRotationOffsetZ;

        _t = _R * (-centerOfRotationOffset - volOrigin);
        _t(_objectDimension - 1) += sourceToCenterOfRotation;

        // // setup the intrinsic parameters _K
        real_t alpha1 = sinoSpacing[0];
        real_t alpha2 = sinoSpacing[1];

        _K << (sourceToCenterOfRotation + centerOfRotationToDetector) / alpha1, 0,
            sinoOrigin[0] / alpha1 + px, 0,
            (sourceToCenterOfRotation + centerOfRotationToDetector) / alpha2,
            sinoOrigin[1] / alpha2 + py, 0, 0, 1;

        buildMatrices();

        buildPrincipalPoint(volOrigin, sourceToCenterOfRotation + centerOfRotationToDetector);

        calculateFanAngle(sinoOrigin);
    }

    const RealMatrix_t& Geometry::getProjectionMatrix() const { return _P; }

    const RealMatrix_t& Geometry::getInverseProjectionMatrix() const { return _Pinv; }

    const RealVector_t& Geometry::getCameraCenter() const { return _C; }

    const RealVector_t& Geometry::getPrincipalPoint() const { return _principalPoint; }

    const RealMatrix_t& Geometry::getRotationMatrix() const { return _R; }

    const RealMatrix_t& Geometry::getInverseRotationMatrix() const { return _invR; }

    geometry::Radian Geometry::getFanAngle() const { return _fanAngle; }

    real_t Geometry::getSourceToPrincipalPointDistance() const
    {
        return (_principalPoint - _C).norm();
    }

    bool Geometry::operator==(const Geometry& other) const
    {
        return (_objectDimension == other._objectDimension && _P == other._P && _Pinv == other._Pinv
                && _K == other._K && _R == other._R && _t == other._t && _S == other._S
                && _C == other._C);
    }

    bool Geometry::operator!=(const Geometry& other) const { return !(*this == other); }

    void Geometry::buildPrincipalPoint(RealVector_t origin, real_t distance)
    {
        _principalPoint << _C + distance * (origin - _C).normalized();
    }

    void Geometry::buildMatrices()
    {
        RealMatrix_t tmpRt(_objectDimension + 1, _objectDimension + 1);
        tmpRt.block(0, 0, _objectDimension, _objectDimension) = _R;
        tmpRt.block(0, _objectDimension, _objectDimension, 1) = _t;
        tmpRt.block(_objectDimension, 0, 1, _objectDimension).setZero();
        tmpRt(_objectDimension, _objectDimension) = 1;

        RealMatrix_t tmpId(_objectDimension, _objectDimension + 1);
        tmpId.setIdentity();

        // setup projection matrix _P
        _P = _K * tmpId * tmpRt * _S;

        // compute the camera center
        _C = -(_P.block(0, 0, _objectDimension, _objectDimension)
                   .colPivHouseholderQr()
                   .solve(_P.block(0, _objectDimension, _objectDimension, 1)));

        // compute inverse _Pinv of _P via its components
        RealMatrix_t Sinv =
            (static_cast<real_t>(1.0) / _S.diagonal().array()).matrix().asDiagonal();

        RealMatrix_t Kinv = RealMatrix_t::Identity(_objectDimension, _objectDimension);
        Kinv(0, 0) = static_cast<real_t>(1.0) / _K(0, 0);
        Kinv(0, _objectDimension - 1) = -_K(0, _objectDimension - 1) / _K(0, 0);
        if (_objectDimension == 3) {
            Kinv(1, 1) = static_cast<real_t>(1.0) / _K(1, 1);
            Kinv(1, _objectDimension - 1) = -_K(1, _objectDimension - 1) / _K(1, 1);
        }

        RealMatrix_t Rtinv(_objectDimension + 1, _objectDimension + 1);
        Rtinv.block(0, 0, _objectDimension, _objectDimension) = _R.transpose();
        Rtinv.block(0, _objectDimension, _objectDimension, 1) = -_R.transpose() * _t;
        Rtinv.block(_objectDimension, 0, 1, _objectDimension).setZero();
        Rtinv(_objectDimension, _objectDimension) = 1;

        RealMatrix_t tmpIdinv(_objectDimension + 1, _objectDimension);
        tmpIdinv.setIdentity();

        _Pinv = Sinv * Rtinv * tmpIdinv * Kinv;
    }

    void Geometry::calculateFanAngle(const RealVector_t& sinoOrigin)
    {
        // Vector from source to edge of detector
        const RealVector_t v1 = [&]() {
            // Size of the vector should be `_objectDimension` - 1
            // Copy it from sinoOrigin so we can preserve the y-coordinate in 3D (sucht that we are
            // in the center of the detector in y direction)
            RealVector_t tmp = sinoOrigin.head(_objectDimension - 1);

            // And then set x to be all the way on the edge
            tmp[0] = 0;

            // compute ray from detector coord (here edge of detector), to center
            return (_Pinv * tmp.homogeneous()).head(_objectDimension).normalized();
        }();

        // Vector from source to principal point
        const RealVector_t v2 = [&]() {
            // Center of sinogram is just the origin of it
            const auto tmp = sinoOrigin.head(_objectDimension - 1);

            // compute ray from detector coord (here principal point), to center
            return (_Pinv * tmp.homogeneous()).head(_objectDimension).normalized();
        }();

        // angle between two vectors is given by it's dot product normalized by the product of the
        // vectors lengths
        _fanAngle = geometry::Radian{std::acos(v1.dot(v2) / (v1.norm() * v2.norm()))};
        std::cout << "Calculated fan angle: " << _fanAngle.to_degree() << "\n";
    }
} // namespace elsa
