#pragma once

#include "DetectorDescriptor.h"

namespace elsa
{
    /**
     * @brief Class representing metadata for lineraized n-dimensional signal stored in memory. It
     * specifically describes signals, which were captured by a planar detector and stores
     * additional information such as different poses
     *
     * @author David Frank - initial code
     */
    class PlanarDetectorDescriptor : public DetectorDescriptor
    {
    public:
        PlanarDetectorDescriptor() = delete;

        ~PlanarDetectorDescriptor() = default;

        /**
         * @brief Construct a PlanatDetectorDescriptor with given number of coefficients and spacing
         * per dimension and a list of geometry poses in the trajectory
         */
        PlanarDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                 const RealVector_t& spacingPerDim,
                                 const std::vector<Geometry>& geometryList);

        /**
         * @brief Construct a PlanatDetectorDescriptor with given number of coefficients
         * per dimension and a list of geometry poses in the trajectory
         */
        PlanarDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                 const std::vector<Geometry>& geometryList);

        using DetectorDescriptor::computeRayFromDetectorCoord;

        /// Override function to compute rays for a planar detector
        RealRay_t computeRayFromDetectorCoord(const RealVector_t& detectorCoord,
                                              const index_t poseIndex) const override;

    private:
        PlanarDetectorDescriptor* cloneImpl() const override;

        bool isEqual(const DataDescriptor& other) const override;
    };
} // namespace elsa
