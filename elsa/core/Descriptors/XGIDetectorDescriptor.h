#pragma once

#include "DetectorDescriptor.h"

namespace elsa
{
    /**
     * @brief Class representing metadata for a (planar) XGI detector
     *
     * @author Matthias Wieczorek (wieczore@cs.tum.edu) - initial code
     * @author Shen Hu (shen.hu@tum.de) - Port to elsa
     */
    class XGIDetectorDescriptor : public DetectorDescriptor
    {
    public:
        typedef Eigen::Matrix<real_t, 3, 1> DirVec;

        XGIDetectorDescriptor() = delete;
        ~XGIDetectorDescriptor() override = default;

        XGIDetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                              const RealVector_t& spacingPerDim,
                              const std::vector<Geometry>& geometryList,
                              const DirVec& sensDir = DirVec(1, 0, 0), bool isParallelBeam = true);

        bool isParallelBeam() const;
        const DirVec& getSensDir() const;

        using DetectorDescriptor::computeRayFromDetectorCoord;

        RealRay_t computeRayFromDetectorCoord(const RealVector_t& detectorCoord,
                                              const index_t poseIndex) const override;

    protected:
        XGIDetectorDescriptor* cloneImpl() const override;
        bool isEqual(const DataDescriptor& other) const override;

    private:
        const DirVec
            _sensDir; ///< sensitivity direction (in-plane orthogonal vector to the grating bars)
        const bool _isParallelBeam; ///< indicating if the geometry is assumed to represent an
                                    ///< approximation to a parallel beam geometry
    };
} // namespace elsa
