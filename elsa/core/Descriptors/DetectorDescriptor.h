#pragma once

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "Geometry.h"
#include "TypeCasts.hpp"

#include <optional>
#include "Eigen/Geometry"

namespace elsa
{
    /**
     * @brief Class representing metadata for lineraized n-dimensional signal stored in memory. It
     * is a base class for different type signals caputred by some kind of detectors (i.e. a planar
     * detector, curved or some other shaped detector). Is additionally stored information about the
     * different poses of a trajectory.
     */
    class DetectorDescriptor : public DataDescriptor
    {
    public:
        /// There is not default signal
        DetectorDescriptor() = delete;

        /// Default destructor
        ~DetectorDescriptor() override = default;

        /**
         * @brief Construct a DetectorDescriptor with a number of coefficients for each dimension
         * and a list of geometry poses
         */
        DetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                           const std::vector<Geometry>& geometryList);
        /**
         * @brief Construct a DetectorDescriptor with a number of coefficients and spacing for each
         * dimension and a list of geometry poses
         */
        DetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                           const RealVector_t& spacingPerDim,
                           const std::vector<Geometry>& geometryList);

        /**
         * @brief Overload of computeRayToDetector with a single detectorIndex. Compute the pose and
         * coord index using getCoordinateFromIndex and call overload
         */
        RealRay_t computeRayFromDetectorCoord(const index_t detectorIndex) const;

        /**
         * @brief Overload of computeRayToDetector with a single coord vector. This vector encodes
         * the pose index and the detector coordinate. So for a 1D detector, this will be 2D and the
         * second dimension, will reference the pose index
         */
        RealRay_t computeRayFromDetectorCoord(const IndexVector_t coord) const;

        /**
         * @brief Compute a ray from the source from a pose to the given detector coordinate
         *
         * @param[in] detectorCoord Vector of size dim - 1, specifying the coordinate the ray should
         * hit
         * @param[in] poseIndex index into geometryList array, which pose to use for ray computation
         *
         */
        virtual RealRay_t computeRayFromDetectorCoord(const RealVector_t& detectorCoord,
                                                      const index_t poseIndex) const = 0;

        /**
         * @brief Computes the projection of the center of a voxel to the detector and its scaling
         *        !!be aware that this function is not optimized, as it uses dynamic sized arrays!!
         *
         * @param[in] voxelCoord coordinate of the voxel to be projected in volume coordinates
         * @param[in] poseIndex index into geometryList array, which pose to use for projection
         * @return std::pair<RealVector_t, real_t> detector coordinate and scaling on detector
         */
        virtual std::pair<RealVector_t, real_t>
            projectAndScaleVoxelOnDetector(const RealVector_t& voxelCoord,
                                           const index_t poseIndex) const;

        /// Get the number of poses used in the geometry
        index_t getNumberOfGeometryPoses() const;

        /// Get the list of geometry poses
        const std::vector<Geometry>& getGeometry() const;

        /// Get the i-th geometry in the trajectory.
        std::optional<Geometry> getGeometryAt(const index_t index) const;

    protected:
        /// implement the polymorphic comparison operation
        bool isEqual(const DataDescriptor& other) const override;

        /// List of geometry poses
        std::vector<Geometry> _geometry;
    };
} // namespace elsa
