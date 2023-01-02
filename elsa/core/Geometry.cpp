#include "Geometry.h"

#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

namespace elsa
{
    using namespace geometry;

    Geometry::Geometry(SourceToCenterOfRotation sourceToCenterOfRotation,
                       CenterOfRotationToDetector centerOfRotationToDetector, Radian angle,
                       VolumeData2D&& volData, SinogramData2D&& sinoData,
                       PrincipalPointOffset offset, RotationOffset2D centerOfRotOffset)
        : _objectDimension{2},
          _sdDistance{sourceToCenterOfRotation + centerOfRotationToDetector},
          _P{RealMatrix_t::Identity(2, 2 + 1)},
          _Pinv{RealMatrix_t::Identity(2 + 1, 2)},
          _ext{RealMatrix_t::Identity(2, 2 + 1)},
          _K{RealMatrix_t::Identity(2, 2)},
          _R{RealMatrix_t::Identity(2, 2)},
          _t{RealVector_t::Zero(2)},
          _S{RealMatrix_t::Identity(2 + 1, 2 + 1)},
          _C{RealVector_t::Zero(2)}
    {
        auto [volSpacing, volOrigin] = std::move(volData);
        auto [sinoSpacing, sinoOrigin] = std::move(sinoData);

        // setup rotation matrix _R
        real_t c = std::cos(angle);
        real_t s = std::sin(angle);
        _R << c, -s, s, c;

        // setup scaling matrix _S
        _S << volSpacing[0], 0, 0, 0, volSpacing[1], 0, 0, 0, 1;

        // set the translation _t
        // auto centerOfRotationOffset = static_cast<RealVector_t>(centerOfRotOffset);
        // centerOfRotationOffset << centerOfRotationOffsetX, centerOfRotationOffsetY;

        _t = _R * (-centerOfRotOffset.get() - volOrigin);
        _t[_objectDimension - 1] += sourceToCenterOfRotation;

        // set the intrinsic parameters _K
        real_t alpha = sinoSpacing[0];
        _K << (sourceToCenterOfRotation + centerOfRotationToDetector) / alpha,
            (sinoOrigin[0] / alpha + offset), 0, 1;

        buildMatrices();
    }

    Geometry::Geometry(SourceToCenterOfRotation sourceToCenterOfRotation,
                       CenterOfRotationToDetector centerOfRotationToDetector,
                       VolumeData3D&& volData, SinogramData3D&& sinoData, RotationAngles3D angles,
                       PrincipalPointOffset2D offset, RotationOffset3D centerOfRotOffset)
        : _objectDimension{3},
          _sdDistance{sourceToCenterOfRotation + centerOfRotationToDetector},
          _P{RealMatrix_t::Identity(3, 3 + 1)},
          _Pinv{RealMatrix_t::Identity(3 + 1, 3)},
          _ext{RealMatrix_t::Identity(3, 3 + 1)},
          _K{RealMatrix_t::Identity(3, 3)},
          _R{RealMatrix_t::Identity(3, 3)},
          _t{RealVector_t::Zero(3)},
          _S{RealMatrix_t::Identity(3 + 1, 3 + 1)},
          _C{RealVector_t::Zero(3)}
    {
        auto [volSpacing, volOrigin] = std::move(volData);
        auto [sinoSpacing, sinoOrigin] = std::move(sinoData);

        real_t alpha = angles.alpha();
        real_t beta = angles.beta();
        real_t gamma = angles.gamma();

        // setup rotation matrix
        real_t ca = std::cos(alpha);
        real_t sa = std::sin(alpha);
        real_t cb = std::cos(beta);
        real_t sb = std::sin(beta);
        real_t cg = std::cos(gamma);
        real_t sg = std::sin(gamma);

        // YZY convention
        _R = (RealMatrix_t(3, 3) << cg, 0, sg, 0, 1, 0, -sg, 0, cg).finished()
             * (RealMatrix_t(3, 3) << cb, -sb, 0, sb, cb, 0, 0, 0, 1).finished()
             * (RealMatrix_t(3, 3) << ca, 0, sa, 0, 1, 0, -sa, 0, ca).finished();

        // setup scaling matrix _S
        _S << volSpacing[0], 0, 0, 0, 0, volSpacing[1], 0, 0, 0, 0, volSpacing[2], 0, 0, 0, 0, 1;

        // setup the translation _t
        // RealVector_t centerOfRotationOffset(_objectDimension);
        // centerOfRotationOffset << centerOfRotationOffsetX, centerOfRotationOffsetY,
        //     centerOfRotationOffsetZ;

        _t = _R * (-centerOfRotOffset.get() - volOrigin);
        _t(_objectDimension - 1) += sourceToCenterOfRotation;

        // setup the intrinsic parameters _K
        real_t alpha1 = sinoSpacing[0];
        real_t alpha2 = sinoSpacing[1];

        _K << (sourceToCenterOfRotation + centerOfRotationToDetector) / alpha1, 0,
            sinoOrigin[0] / alpha1 + offset[0], 0,
            (sourceToCenterOfRotation + centerOfRotationToDetector) / alpha2,
            sinoOrigin[1] / alpha2 + offset[1], 0, 0, 1;

        buildMatrices();
    }

    Geometry::Geometry(real_t sourceToCenterOfRotation, real_t centerOfRotationToDetector,
                       const DataDescriptor& volumeDescriptor, const DataDescriptor& sinoDescriptor,
                       const RealMatrix_t& R, real_t px, real_t py, real_t centerOfRotationOffsetX,
                       real_t centerOfRotationOffsetY, real_t centerOfRotationOffsetZ)
        : _objectDimension{3},
          _sdDistance{sourceToCenterOfRotation + centerOfRotationToDetector},
          _P{RealMatrix_t::Identity(3, 3 + 1)},
          _Pinv{RealMatrix_t::Identity(3 + 1, 3)},
          _ext{RealMatrix_t::Identity(3, 3 + 1)},
          _K{RealMatrix_t::Identity(3, 3)},
          _R{R},
          _t{RealVector_t::Zero(3)},
          _S{RealMatrix_t::Identity(3 + 1, 3 + 1)},
          _C{RealVector_t::Zero(3)}
    {
        // sanity check
        if (R.rows() != _objectDimension || R.cols() != _objectDimension)
            throw InvalidArgumentError(
                "Geometry: 3D geometry requested with non-3D rotation matrix");

        // setup scaling matrix _S
        _S << volumeDescriptor.getSpacingPerDimension()[0], 0, 0, 0, 0,
            volumeDescriptor.getSpacingPerDimension()[1], 0, 0, 0, 0,
            volumeDescriptor.getSpacingPerDimension()[2], 0, 0, 0, 0, 1;

        // setup the translation _t
        RealVector_t centerOfRotationOffset(_objectDimension);
        centerOfRotationOffset << centerOfRotationOffsetX, centerOfRotationOffsetY,
            centerOfRotationOffsetZ;

        _t = _R * (-centerOfRotationOffset - volumeDescriptor.getLocationOfOrigin());
        _t(_objectDimension - 1) += sourceToCenterOfRotation;

        // setup the intrinsic parameters _K
        real_t alpha1 = sinoDescriptor.getSpacingPerDimension()[0];
        real_t alpha2 = sinoDescriptor.getSpacingPerDimension()[1];

        _K << (sourceToCenterOfRotation + centerOfRotationToDetector) / alpha1, 0,
            sinoDescriptor.getLocationOfOrigin()[0] / alpha1 + px, 0,
            (sourceToCenterOfRotation + centerOfRotationToDetector) / alpha2,
            sinoDescriptor.getLocationOfOrigin()[1] / alpha2 + py, 0, 0, 1;

        buildMatrices();
    }

    Geometry::Geometry(geometry::VolumeData3D&& volData, geometry::SinogramData3D&& sinoData,
                       const RealMatrix_t& R, const RealMatrix_t& t, const RealMatrix_t& K)
        : _objectDimension{3},
          _P{RealMatrix_t::Identity(3, 3 + 1)},
          _Pinv{RealMatrix_t::Identity(3 + 1, 3)},
          _ext{RealMatrix_t::Identity(3, 3 + 1)},
          _K{K},
          _R{R},
          _t{t},
          _S{RealMatrix_t::Identity(3 + 1, 3 + 1)},
          _C{RealVector_t::Zero(3)}
    {
        auto [volSpacing, volOrigin] = std::move(volData);

        // reconstruct source detector Distance from intrinsics
        _sdDistance = sinoData.getSpacing()[0] * K(0, 0);

        _S << volSpacing[0], 0, 0, 0, 0, volSpacing[1], 0, 0, 0, 0, volSpacing[2], 0, 0, 0, 0, 1;

        buildMatrices();
    }

    const RealMatrix_t& Geometry::getProjectionMatrix() const
    {
        return _P;
    }

    const RealMatrix_t& Geometry::getInverseProjectionMatrix() const
    {
        return _Pinv;
    }

    const RealVector_t& Geometry::getCameraCenter() const
    {
        return _C;
    }

    const RealMatrix_t& Geometry::getRotationMatrix() const
    {
        return _R;
    }

    real_t Geometry::getSourceDetectorDistance() const
    {
        return _sdDistance;
    }

    const RealMatrix_t& Geometry::getExtrinsicMatrix() const
    {
        return _ext;
    }

    const RealMatrix_t& Geometry::getIntrinsicMatrix() const
    {
        return _K;
    }

    const RealVector_t& Geometry::getTranslationVector() const
    {
        return _t;
    }

    bool Geometry::operator==(const Geometry& other) const
    {
        return (_objectDimension == other._objectDimension && _P == other._P && _Pinv == other._Pinv
                && _K == other._K && _R == other._R && _t == other._t && _S == other._S
                && _C == other._C && _sdDistance == other._sdDistance && _ext == other._ext);
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

        // setup extrinsic matrix
        auto _extFull = tmpRt * _S;
        _ext = _extFull.block(0, 0, _objectDimension, _objectDimension + 1);

        // setup projection matrix _P
        _P = _K * tmpId * _extFull;

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

} // namespace elsa
