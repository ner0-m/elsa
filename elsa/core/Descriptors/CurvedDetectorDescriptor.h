#pragma once

#include "DetectorDescriptor.h"

namespace elsa
{
    /**
     * @brief Class representing metadata for lineraized n-dimensional signal stored in memory. It
     * specifically describes signals, which were captured by a planar detector and stores
     * additional information such as different poses
     *
     * Derived from DetectorDescriptor.
     *
     * The class is parameterized by a single angle theta. This angle describes the curvature. It
     * is the angle between the lines from source to principal point and source to outer point of
     * the detector on the x-y plane (i.e. z = 0)
     *
     *
     * @author David Frank - initial code
     */
    class CurvedDetectorDescriptor : public DetectorDescriptor
    {
        using DetectorDescriptor::Ray;

    public:
        CurvedDetectorDescriptor() = delete;

        ~CurvedDetectorDescriptor() = default;

        /**
         * @brief Construct a PlanatDetectorDescriptor with given number of coefficients and spacing
         * per dimension and a list of geometry poses in the trajectory
         */
        CurvedDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                 const RealVector_t& spacingPerDim,
                                 const std::vector<Geometry>& geometryList, real_t scale);

        /**
         * @brief Construct a PlanatDetectorDescriptor with given number of coefficients
         * per dimension and a list of geometry poses in the trajectory
         */
        CurvedDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                 const std::vector<Geometry>& geometryList, real_t scale);

        using DetectorDescriptor::computeRayFromDetectorCoord;

        /// Override function to compute rays for a planar detector
        Ray computeRayFromDetectorCoord(const RealVector_t& detectorCoord,
                                        const index_t poseIndex) const override;

        RealVector_t computeDetectorCoordFromRay(const Ray& ray,
                                                 const index_t poseIndex) const override;

    private:
        CurvedDetectorDescriptor* cloneImpl() const override;

        bool isEqual(const DataDescriptor& other) const override;

        RealVector_t mapCurvedPixelToDir(const RealVector_t& p, const Geometry& geom) const;

        /// Angle between line from source to principal point and source to outer edge of
        /// detector
        geometry::Radian _theta;

        real_t _scale;
    };
} // namespace elsa
