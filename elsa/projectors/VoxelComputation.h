#pragma once

#include "elsaDefines.h"
#include "Math.hpp"
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

        template <index_t dim, size_t N, typename data_t>
        data_t classic_weight_function(Lut<data_t, N> lut, Eigen::Matrix<real_t, dim, 1> distance)
        {
            return lut(distance.norm());
        }

        template <size_t N, typename data_t>
        data_t differential_weight_function_2D(Lut<data_t, N> lut,
                                               Eigen::Matrix<real_t, 1, 1> distance)
        {
            return lut(distance.norm()) * math::sgn(distance[0]);
        }

        template <size_t N, typename data_t>
        data_t differential_weight_function_3D(Lut<data_t, N> lut,
                                               Eigen::Matrix<real_t, 2, 1> distance)
        {
            return lut(distance.norm()) * distance[0];
        }

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
            RealVectorHom_t homogeneousVoxelCoord = coordinateShifted.homogeneous();

            // Project voxel center onto detector
            RealVectorDetector_t center =
                (geometry.getProjectionMatrix() * homogeneousVoxelCoord).hnormalized();
            center = center.array() - 0.5;

            auto scaling = geometry.getSourceDetectorDistance()
                           / (geometry.getExtrinsicMatrix() * homogeneousVoxelCoord).norm();

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

        template <int dim, typename data_t, size_t N, typename basis_function_t>
        void forwardVoxel(const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                          const Lut<data_t, N>& lut, basis_function_t& basis_function)
        {

            // Just to be sure, zero out the result
            Ax = 0;

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
                        domainIndex, geomIndex, geometry, lut.support(), detectorDims,
                        volumeStrides,
                        [&](const index_t index, Eigen::Matrix<real_t, dim - 1, 1> distance) {
                            auto wght = basis_function(lut, distance);
                            Ax[index] += voxelWeight * wght;
                        });
                }
            }
        }

        template <int dim, typename data_t, size_t N, typename basis_function_t>
        void backwardVoxel(const DataContainer<data_t>& y, DataContainer<data_t>& Aty,
                           const Lut<data_t, N>& lut, basis_function_t basis_function)
        {

            // Just to be sure, zero out the result
            Aty = 0;

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
                        domainIndex, geomIndex, geometry, lut.support(), detectorDims,
                        volumeStrides,
                        [&](const index_t index, Eigen::Matrix<real_t, dim - 1, 1> distance) {
                            auto wght = basis_function(lut, distance);
                            Aty[domainIndex] += wght * y[index];
                        });
                }
            }
        }
    }; // namespace voxel
};     // namespace elsa