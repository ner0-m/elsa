#pragma once

#include "elsaDefines.h"
#include "Geometry.h"
#include "Blobs.h"

namespace elsa
{

    namespace voxel
    {

        using RealVector2D_t = Eigen::Matrix<real_t, 2, 1>;
        using RealVector3D_t = Eigen::Matrix<real_t, 3, 1>;
        using RealVector4D_t = Eigen::Matrix<real_t, 4, 1>;
        using IndexVector2D_t = Eigen::Matrix<index_t, 2, 1>;
        using IndexVector3D_t = Eigen::Matrix<index_t, 3, 1>;

        template <typename data_t, index_t N>
        class VoxelLut
        {
        public:
            constexpr VoxelLut() : blob_(2, 10.83, 2) {}

            constexpr data_t radius() const { return blob_.radius(); }

            constexpr data_t alpha() const { return blob_.alpha(); }

            constexpr data_t order() const { return blob_.order(); }

            template <index_t dim>
            constexpr data_t weight(Eigen::Matrix<data_t, dim, 1> distance) const
            {
                return blob_.get_lut()((distance.norm() / blob_.radius()) * N);
            }

            constexpr auto data() const { return blob_.get_lut().data(); }

        private:
            ProjectedBlob<data_t> blob_;
        };

        template <index_t dim, class Fn>
        void visitDetector(index_t domainIndex, index_t geomIndex, Geometry geometry, real_t radius,
                           Eigen::Matrix<index_t, dim - 1, 1> detectorDims,
                           Eigen::Matrix<index_t, dim, 1> volumeStrides, Fn apply)
        {
            using IndexVectorVolume_t = Eigen::Matrix<index_t, dim, 1>;
            using IndexVectorDetector_t = Eigen::Matrix<index_t, dim - 1, 1>;
            using RealVectorVolume_t = Eigen::Matrix<real_t, dim, 1>;
            using RealVectorDetector_t = Eigen::Matrix<real_t, dim - 1, 1>;
            using RealVectorHom_t = Eigen::Matrix<real_t, dim + 1, 1>;

            // compute coordinate from index
            IndexVectorVolume_t coordinate = detail::idx2Coord(domainIndex, volumeStrides);

            // Cast to real_t and shift to center of voxel according to origin
            RealVectorVolume_t coordinateShifted = coordinate.template cast<real_t>().array() + 0.5;
            RealVectorHom_t homogenousVoxelCoord = coordinateShifted.homogeneous();

            // Project voxel center onto detector
            RealVectorDetector_t center =
                (geometry.getProjectionMatrix() * homogenousVoxelCoord).hnormalized();
            center = center.array() - 0.5;

            auto sourceVoxelDistance =
                (geometry.getExtrinsicMatrix() * homogenousVoxelCoord).norm();

            auto scaling = geometry.getSourceDetectorDistance() / sourceVoxelDistance;

            auto radiusOnDetector = scaling * radius;
            IndexVectorDetector_t detector_max = detectorDims.array() - 1;
            index_t detectorZeroIndex = (detector_max[0] + 1) * geomIndex;

            IndexVectorDetector_t min_corner =
                (center.array() - radiusOnDetector).ceil().template cast<index_t>();
            min_corner = min_corner.cwiseMax(IndexVectorDetector_t::Zero());
            IndexVectorDetector_t max_corner =
                (center.array() + radiusOnDetector).floor().template cast<index_t>();
            max_corner = max_corner.cwiseMin(detector_max);

            RealVectorDetector_t current = min_corner.template cast<real_t>();
            index_t currentIndex{0}, iStride{0}, jStride{0};
            if constexpr (dim == 2) {
                currentIndex = detectorZeroIndex + min_corner[0];
            } else {
                currentIndex = detectorZeroIndex * (detector_max[1] + 1)
                               + min_corner[1] * (detector_max[0] + 1) + min_corner[0];

                iStride = max_corner[0] - min_corner[0] + 1;
                jStride = (detector_max[0] + 1) - iStride;
            }
            for (index_t i = min_corner[0]; i <= max_corner[0]; i++) {
                if constexpr (dim == 2) {
                    // traverse detector pixel in voxel footprint
                    const Eigen::Matrix<real_t, 1, 1> distance((center[0] - i) / scaling);
                    apply(detectorZeroIndex + i, distance);
                } else {
                    for (index_t j = min_corner[1]; j <= max_corner[1]; j++) {
                        const RealVector2D_t distanceVec = (center - current) / scaling;
                        apply(currentIndex, distanceVec);
                        currentIndex += 1;
                        current[0] += 1;
                    }
                    currentIndex += jStride;
                    current[0] -= iStride;
                    current[1] += 1;
                }
            }
        }

        template <int dim, typename data_t, typename basis_function>
        void forwardVoxel(const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                          basis_function& lut)
        {
            const DetectorDescriptor& detectorDesc =
                downcast<DetectorDescriptor>(Ax.getDataDescriptor());

            auto& volume = x.getDataDescriptor();
            const Eigen::Matrix<index_t, dim, 1>& volumeStrides =
                volume.getProductOfCoefficientsPerDimension();
            const Eigen::Matrix<index_t, dim - 1, 1>& detectorDims =
                detectorDesc.getNumberOfCoefficientsPerDimension().head(dim - 1);

            // loop over geometries/poses in parallel
#pragma omp parallel for
            for (index_t geomIndex = 0; geomIndex < detectorDesc.getNumberOfGeometryPoses();
                 geomIndex++) {
                auto& geometry = detectorDesc.getGeometry()[asUnsigned(geomIndex)];
                // loop over voxels
                for (index_t domainIndex = 0; domainIndex < volume.getNumberOfCoefficients();
                     ++domainIndex) {
                    auto voxelWeight = x[domainIndex];

                    visitDetector<dim>(
                        domainIndex, geomIndex, geometry, lut.radius(), detectorDims, volumeStrides,
                        [&](const index_t index, Eigen::Matrix<real_t, dim - 1, 1> distance) {
                            auto wght = lut.template weight<dim - 1>(distance);
                            Ax[index] += voxelWeight * wght;
                        });
                }
            }
        }

        template <int dim, typename data_t, typename basis_function>
        void backwardVoxel(const DataContainer<data_t>& y, DataContainer<data_t>& Aty,
                           basis_function& lut)
        {
            const DetectorDescriptor& detectorDesc =
                downcast<DetectorDescriptor>(y.getDataDescriptor());

            auto& volume = Aty.getDataDescriptor();
            const Eigen::Matrix<index_t, dim, 1>& volumeStrides =
                volume.getProductOfCoefficientsPerDimension();
            const Eigen::Matrix<index_t, dim - 1, 1>& detectorDims =
                detectorDesc.getNumberOfCoefficientsPerDimension().head(dim - 1);

#pragma omp parallel for
            // loop over voxels in parallel
            for (index_t domainIndex = 0; domainIndex < volume.getNumberOfCoefficients();
                 ++domainIndex) {
                // loop over geometries/poses in parallel
                for (index_t geomIndex = 0; geomIndex < detectorDesc.getNumberOfGeometryPoses();
                     geomIndex++) {

                    auto& geometry = detectorDesc.getGeometry()[asUnsigned(geomIndex)];

                    visitDetector<dim>(
                        domainIndex, geomIndex, geometry, lut.radius(), detectorDims, volumeStrides,
                        [&](const index_t index, Eigen::Matrix<real_t, dim - 1, 1> distance) {
                            auto wght = lut.template weight<dim - 1>(distance);
                            Aty[domainIndex] += wght * y[index];
                        });
                }
            }
        }
    }; // namespace voxel
};     // namespace elsa