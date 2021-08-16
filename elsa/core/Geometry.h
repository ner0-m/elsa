#pragma once

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "StrongTypes.h"

#include <utility>
#include <cassert>

namespace elsa
{
    /**
     * @brief Class representing 2d/3d projective camera geometry for use in CT projectors.
     *
     * @author Matthias Wieczorek - initial code
     * @author Maximilian Hornung - modularization, redesign
     * @author David Frank - bugfixes, strong typing
     * @author Nikola Dinev - refactoring
     * @author Tobias Lasser - refactoring, modernization
     *
     * The location of X-ray source, volume (typically containing the center of rotation), and X-ray
     * detector are encoded using projection matrices (see A. Zissermann, "Multiple View Geometry in
     * Computer Vision"). Detectors are assumed to be flat.
     */
    class Geometry
    {
    public:
        /**
         * @brief Constructor for 2D projective geometry
         *
         * @param[in] sourceToCenterOfRotation distance from source to the center of rotation (along
         * y axis)
         * @param[in] centerOfRotationToDetector distance from center of rotation to detector (along
         * y axis)
         * @param[in] angle rotation angle (in radians)
         * @param[in] volData descriptor for the 2d volume
         * @param[in] sinoData descriptor for the sinogram
         * @param[in] offset offset of the principal point [default 0]
         * @param[in] centerOfRotOffset offset of the center of rotation
         *
         *
         * VolumeData2D and SinogramData2D are taken as r-value references, as it's cheaper to move,
         * them in, and they are only intended as temporary objects. o construct a Geometry with
         * VolumeData2D{...}/SinogramData2D{...} as temporary or move the object in, but be aware of
         * reusing it
         */
        Geometry(geometry::SourceToCenterOfRotation sourceToCenterOfRotation,
                 geometry::CenterOfRotationToDetector centerOfRotationToDetector,
                 geometry::Radian angle, geometry::VolumeData2D&& volData,
                 geometry::SinogramData2D&& sinoData,
                 geometry::PrincipalPointOffset offset = geometry::PrincipalPointOffset{0},
                 geometry::RotationOffset2D centerOfRotOffset = geometry::RotationOffset2D{0, 0});

        /**
         * @brief Constructor for 3D projective geometry using Euler angles
         *
         * @param[in] sourceToCenterOfRotation distance from source to the center of rotation (along
         * z axis)
         * @param[in] centerOfRotationToDetector distance from center of rotation to detector (along
         * z axis)
         * @param[in] volData descriptor for the 3d volume
         * @param[in] sinoData descriptor for the sinogram
         * @param[in] angles (gamma -> around y''-axis, beta -> around z' axis, alpha -> around y
         * axis) in radians
         * @param[in] offset offset of the principal point
         * @param[in] centerOfRotOffset offset of the center of rotation
         *
         * Alpha, beta, gamma are Euler rotation angles using the YZY convention. They are specified
         * in radians. In standard circular trajectory CT settings, we would have alpha = beta = 0,
         * while gamma is the angle of rotation)
         *
         * VolumeData3D and SinogramData3D are taken as r-value references, as it's cheaper to move,
         * them in, and they are only intended as temporary objects. So construct a Geometry with
         * VolumeData3D{...}/SinogramData3D{...} as temporary or move the object in, but be aware of
         * reusing it
         */
        Geometry(geometry::SourceToCenterOfRotation sourceToCenterOfRotation,
                 geometry::CenterOfRotationToDetector centerOfRotationToDetector,
                 geometry::VolumeData3D&& volData, geometry::SinogramData3D&& sinoData,
                 geometry::RotationAngles3D angles,
                 geometry::PrincipalPointOffset2D offset = geometry::PrincipalPointOffset2D{0, 0},
                 geometry::RotationOffset3D centerOfRotOffset = geometry::RotationOffset3D{0, 0,
                                                                                           0});

        /**
         * @brief Constructor for 3D projective geometry using a 3x3 rotation matrix
         *
         * @param[in] sourceToCenterOfRotation distance from source to the center of rotation (along
         * z axis)
         * @param[in] centerOfRotationToDetector distance from center of rotation to
         * detector (along z axis)
         * @param[in] volumeDescriptor descriptor for the 3d volume
         * @param[in] sinoDescriptor descriptor for the sinogram
         * @param[in] R a 3x3 rotation matrix
         * @param[in] px offset of the principal point in x-direction [default 0]
         * @param[in] py offset of the principal point in y-direction [default 0]
         * @param[in] centerOfRotationOffsetX offset of the center of rotation in x direction
         * [default 0]
         * @param[in] centerOfRotationOffsetY offset of the center of rotation in y direction
         * [default 0]
         * @param[in] centerOfRotationOffsetZ offset of the center of rotation in z direction
         * [default 0]
         */
        Geometry(real_t sourceToCenterOfRotation, real_t centerOfRotationToDetector,
                 const DataDescriptor& volumeDescriptor, const DataDescriptor& sinoDescriptor,
                 const RealMatrix_t& R, real_t px = static_cast<real_t>(0.0),
                 real_t py = static_cast<real_t>(0.0),
                 real_t centerOfRotationOffsetX = static_cast<real_t>(0.0),
                 real_t centerOfRotationOffsetY = static_cast<real_t>(0.0),
                 real_t centerOfRotationOffsetZ = static_cast<real_t>(0.0));

        Geometry(geometry::VolumeData3D&& volData, geometry::SinogramData3D&& sinoData,
                 const RealMatrix_t& R, const RealMatrix_t& t, const RealMatrix_t& K);

        /**
         * @brief Return the projection matrix
         *
         * @returns projection matrix
         */
        const RealMatrix_t& getProjectionMatrix() const;

        /**
         * @brief Return the inverse of the projection matrix
         *
         * @returns the inverse of the projection matrix
         */
        const RealMatrix_t& getInverseProjectionMatrix() const;

        /**
         * @brief Return the camera center corresponding to the projection matrix
         *
         * @returns the camera center (as a coordinate vector)
         */
        const RealVector_t& getCameraCenter() const;

        /**
         * @brief Return the rotation matrix corresponding to the projection matrix
         *
         * @returns the rotation matrix
         */
        const RealMatrix_t& getRotationMatrix() const;

        /// comparison operator
        bool operator==(const Geometry& other) const;

    private:
        /// the dimension of the object space / volume (either 2 or 3)
        index_t _objectDimension;

        /// the projection matrix (= [_K|0] * [_R|_t] * _S)
        RealMatrix_t _P;
        /// the inverse of the projection matrix
        RealMatrix_t _Pinv;

        /// the intrinsic parameters _K
        RealMatrix_t _K;
        /// the rotation matrix
        RealMatrix_t _R;
        /// the translation in object space
        RealVector_t _t;
        /// the scaling in object space
        RealMatrix_t _S;

        /// the camera center _C
        RealVector_t _C;

        /// build the projection matrix, its inverse and the camera center
        void buildMatrices();
    };
} // namespace elsa
