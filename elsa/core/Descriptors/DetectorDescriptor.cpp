#include "DetectorDescriptor.h"
#include "TypeCasts.hpp"

namespace elsa
{
    DetectorDescriptor::DetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                           const std::vector<Geometry>& geometryList)
        : DataDescriptor(numOfCoeffsPerDim), _geometry(geometryList)
    {
        // TODO Clarify: What about empty geometryList? Do we want to support it, or throw an
        // execption?
    }

    DetectorDescriptor::DetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                           const RealVector_t& spacingPerDim,
                                           const std::vector<Geometry>& geometryList)
        : DataDescriptor(numOfCoeffsPerDim, spacingPerDim), _geometry(geometryList)
    {
    }

    DetectorDescriptor::Ray
        DetectorDescriptor::computeRayFromDetectorCoord(const index_t detectorIndex) const
    {

        // Return empty, if access out of bounds
        assert(detectorIndex < getNumberOfCoefficients()
               && "PlanarDetectorDescriptor::computeRayToDetector(index_t): Assumption "
                  "detectorIndex smaller than number of coeffs, broken");

        auto coord = getCoordinateFromIndex(detectorIndex);
        return computeRayFromDetectorCoord(coord);
    }

    DetectorDescriptor::Ray
        DetectorDescriptor::computeRayFromDetectorCoord(const IndexVector_t coord) const
    {
        // Assume all of the coordinates are inside of the volume
        // auto tmp = (coord.array() < getNumberOfCoefficientsPerDimension().array());
        // assert(tmp.all()
        // && "DetectorDescriptor::computeRayToDetector(IndexVector_t): Assumption coord "
        // "in bound wrong");

        auto dim = getNumberOfDimensions();

        // Assume dimension of coord is equal to dimension of descriptor
        assert(dim == coord.size());

        // Cast to real_t and shift to center of pixel
        auto detectorCoord = coord.head(dim - 1).template cast<real_t>().array() + 0.5;

        // Last dimension is always the pose index
        auto poseIndex = coord[dim - 1];

        return computeRayFromDetectorCoord(detectorCoord, poseIndex);
    }

    std::vector<Geometry> DetectorDescriptor::getGeometry() const { return _geometry; }

    index_t DetectorDescriptor::getNumberOfGeometryPoses() const
    {
        return static_cast<index_t>(_geometry.size());
    }

    std::optional<Geometry> DetectorDescriptor::getGeometryAt(const index_t index) const
    {
        // Cast to size_t to silence warnings
        auto i = asUnsigned(index);

        if (_geometry.size() <= i)
            return {};

        return _geometry[i];
    }

    IndexVector_t DetectorDescriptor::getCountOfPrincipalRaysPerMainDirection() const
    {
        if (!_countPrincipalRaysPerMainDir)
            computeMainDirectionsOfPrincipalRays();

        return *_countPrincipalRaysPerMainDir;
    }

    IndexVector_t DetectorDescriptor::getMainDirectionOfPrincipalRayPerPose() const
    {
        if (!_mainDirectionOfPrincipalRayPerPose)
            computeMainDirectionsOfPrincipalRays();

        return *_mainDirectionOfPrincipalRayPerPose;
    }

    bool DetectorDescriptor::isEqual(const DataDescriptor& other) const
    {
        if (!DataDescriptor::isEqual(other))
            return false;

        // static cast as type checked in base comparison
        auto otherBlock = static_cast<const DetectorDescriptor*>(&other);

        if (getNumberOfGeometryPoses() != otherBlock->getNumberOfGeometryPoses())
            return false;

        return std::equal(std::cbegin(_geometry), std::cend(_geometry),
                          std::cbegin(otherBlock->_geometry));
    }

    void DetectorDescriptor::computeMainDirectionsOfPrincipalRays() const
    {
        const auto numDim = getNumberOfDimensions();
        const auto numPoses = getNumberOfGeometryPoses();
        RealVector_t detectorCenter = getNumberOfCoefficientsPerDimension().template cast<real_t>();
        detectorCenter.conservativeResize(numDim - 1);
        detectorCenter *= static_cast<real_t>(0.5);

        _mainDirectionOfPrincipalRayPerPose = IndexVector_t(numPoses);
        _countPrincipalRaysPerMainDir = IndexVector_t::Zero(numDim);
        for (index_t i = 0; i < numPoses; i++) {
            auto principalRay = computeRayFromDetectorCoord(detectorCenter, i);
            principalRay.direction().maxCoeff(&(*_mainDirectionOfPrincipalRayPerPose)[i]);
            (*_countPrincipalRaysPerMainDir)[(*_mainDirectionOfPrincipalRayPerPose)[i]]++;
        }
    }
} // namespace elsa
