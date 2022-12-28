#include "DetectorDescriptor.h"
#include "Logger.h"

namespace elsa
{
    DetectorDescriptor::DetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                           const std::vector<Geometry>& geometryList)
        : DataDescriptor(numOfCoeffsPerDim), _geometry(geometryList)
    {
        // TODO Clarify: What about empty geometryList? Do we want to support it, or throw an
        // exception?
    }

    DetectorDescriptor::DetectorDescriptor(const IndexVector_t& numOfCoeffsPerDim,
                                           const RealVector_t& spacingPerDim,
                                           const std::vector<Geometry>& geometryList)
        : DataDescriptor(numOfCoeffsPerDim, spacingPerDim), _geometry(geometryList)
    {
    }

    RealRay_t DetectorDescriptor::computeRayFromDetectorCoord(const index_t detectorIndex) const
    {

        // Return empty, if access out of bounds
        assert(detectorIndex < getNumberOfCoefficients()
               && "PlanarDetectorDescriptor::computeRayToDetector(index_t): Assumption "
                  "detectorIndex smaller than number of coeffs, broken");

        auto coord = getCoordinateFromIndex(detectorIndex);
        return computeRayFromDetectorCoord(coord);
    }

    RealRay_t DetectorDescriptor::computeRayFromDetectorCoord(const IndexVector_t coord) const
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

    const std::vector<Geometry>& DetectorDescriptor::getGeometry() const
    {
        return _geometry;
    }

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

    RealRay_t DetectorDescriptor::computeRayFromDetectorCoord(const RealVector_t& detectorCoord,
                                                              const index_t poseIndex) const
    {
        // Assert that for all dimension of detectorCoord is in bounds and poseIndex can
        // be index in the _geometry. If not the calculation will not be correct, but
        // as this is the hot path, I don't want exceptions and unpacking everything
        // We'll just have to ensure, that we don't mess up in our hot path! :-)
        assert((detectorCoord.block(0, 0, getNumberOfDimensions() - 1, 0).array()
                < getNumberOfCoefficientsPerDimension()
                      .block(0, 0, getNumberOfDimensions() - 1, 0)
                      .template cast<real_t>()
                      .array())
                   .all()
               && "PlanarDetectorDescriptor::computeRayToDetector: Assumption detectorCoord in "
                  "bounds, wrong");
        assert(asUnsigned(poseIndex) < _geometry.size()
               && "PlanarDetectorDescriptor::computeRayToDetector: Assumption poseIndex smaller "
                  "than number of poses, wrong");

        auto dim = getNumberOfDimensions();

        // get the pose of trajectory
        auto geometry = _geometry[asUnsigned(poseIndex)];

        auto projInvMatrix = geometry.getInverseProjectionMatrix();

        // homogeneous coordinates [p;1], with p in detector space
        RealVector_t homogeneousPixelCoord(dim);
        homogeneousPixelCoord << detectorCoord, 1;

        // Camera center is always the ray origin
        auto ro = geometry.getCameraCenter();

        auto rd = (projInvMatrix * homogeneousPixelCoord) // Matrix-Vector multiplication
                      .head(dim)                          // Transform to non-homogeneous
                      .normalized();                      // normalize vector

        return RealRay_t(ro, rd);
    }

    std::pair<RealVector_t, real_t>
        DetectorDescriptor::projectAndScaleVoxelOnDetector(const RealVector_t& voxelCoord,
                                                           const index_t poseIndex) const
    {
        assert(asUnsigned(poseIndex) < _geometry.size()
               && "PlanarDetectorDescriptor::computeRayToDetector: Assumption poseIndex smaller "
                  "than number of poses, wrong");

        auto dim = getNumberOfDimensions();

        // get the pose of trajectory
        const auto& geometry = _geometry[asUnsigned(poseIndex)];
        const auto& projMatrix = geometry.getProjectionMatrix();
        const auto& extMatrix = geometry.getExtrinsicMatrix();

        // homogeneous coordinates [p;1], with p in detector space
        RealVector_t homogeneousVoxelCoord(voxelCoord.size() + 1);
        homogeneousVoxelCoord << voxelCoord, 1;

        RealVector_t voxelCenterOnDetectorHomogenous = (projMatrix * homogeneousVoxelCoord);
        voxelCenterOnDetectorHomogenous.block(0, 0, dim - 1, 1) /=
            voxelCenterOnDetectorHomogenous[dim - 1];
        voxelCenterOnDetectorHomogenous[dim - 1] = asUnsigned(poseIndex);

        RealVector_t voxelInCameraSpace = (extMatrix * homogeneousVoxelCoord);
        auto distance = voxelInCameraSpace.head(dim).norm();

        // compute scaling assuming rays orthogonal to detector
        auto scaling = geometry.getSourceDetectorDistance() / distance;

        return {std::move(voxelCenterOnDetectorHomogenous), scaling};
    }

} // namespace elsa
