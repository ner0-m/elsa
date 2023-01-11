#include "AXDTOperator.h"
#include "IdenticalBlocksDescriptor.h"
#include "Scaling.h"
#include "Math.hpp"

#include <iostream>

using namespace elsa::axdt;

namespace elsa
{
    template <typename data_t>
    AXDTOperator<data_t>::AXDTOperator(const VolumeDescriptor& domainDescriptor,
                                       const XGIDetectorDescriptor& rangeDescriptor,
                                       const LinearOperator<data_t>& projector,
                                       const DirVecList& sphericalFuncDirs,
                                       const WeightVec& sphericalFuncWeights,
                                       const Symmetry& sphericalHarmonicsSymmetry,
                                       const index_t& sphericalHarmonicsMaxDegree)
        : B(IdenticalBlocksDescriptor{SphericalFunctionInformation<data_t>(
                                          sphericalFuncDirs, sphericalFuncWeights,
                                          sphericalHarmonicsSymmetry, sphericalHarmonicsMaxDegree)
                                          .basisCnt,
                                      domainDescriptor},
            rangeDescriptor,
            computeOperatorList(rangeDescriptor,
                                SphericalFunctionInformation<data_t>(
                                    sphericalFuncDirs, sphericalFuncWeights,
                                    sphericalHarmonicsSymmetry, sphericalHarmonicsMaxDegree),
                                projector),
            B::BlockType::COL)
    {
    }

    template <typename data_t>
    AXDTOperator<data_t>::AXDTOperator(const AXDTOperator& other) : B(other)
    {
    }

    template <typename data_t>
    AXDTOperator<data_t>* AXDTOperator<data_t>::cloneImpl() const
    {
        return new AXDTOperator<data_t>(*this);
    }

    template <typename data_t>
    bool AXDTOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        return BlockLinearOperator<data_t>::isEqual(other);
    }

    template <typename data_t>
    typename AXDTOperator<data_t>::OperatorList AXDTOperator<data_t>::computeOperatorList(
        const XGIDetectorDescriptor& rangeDescriptor,
        const SphericalFunctionInformation<data_t>& sf_info,
        const LinearOperator<data_t>& projector)
    {
        auto weights = computeSphericalHarmonicsWeights(rangeDescriptor, sf_info);

        OperatorList ops;

        const index_t numBlocks = sf_info.basisCnt;

        // create composite operators of projector and scalings
        for (index_t i = 0; i < numBlocks; ++i) {
            // create a matching scaling operator
            std::unique_ptr<Scaling<data_t>> s;
            if (rangeDescriptor.isParallelBeam()) {
                DataContainer<data_t> tmp(rangeDescriptor);
                index_t totalCnt = rangeDescriptor.getNumberOfCoefficients();
                index_t blkCnt = rangeDescriptor.getNumberOfCoefficientsPerDimension()
                                     [rangeDescriptor.getNumberOfDimensions() - 1];
                index_t perBlkCnt = totalCnt / blkCnt;

                index_t idx = 0;
                for (index_t j = 0; j < blkCnt; ++j) {
                    index_t cnt_down = perBlkCnt;
                    while ((cnt_down--) != 0) {
                        tmp[idx++] = weights->getBlock(i)[j];
                    }
                }

                s = std::make_unique<Scaling<data_t>>(rangeDescriptor, tmp);
            } else {
                s = std::make_unique<Scaling<data_t>>(rangeDescriptor,
                                                      materialize(weights->getBlock(i)));
            }

            ops.emplace_back(std::make_unique<LinearOperator<data_t>>(*s * projector));

            //            std::cout << ".. >> initialized composite operator " << i + 1 << "/" <<
            //            numBlocks
            //                      << std::endl;
        }

        // EDF::write<data_t>(*weights,"_weights.edf");

        return ops;
    }

    template <typename data_t>
    std::unique_ptr<DataContainer<data_t>> AXDTOperator<data_t>::computeSphericalHarmonicsWeights(
        const XGIDetectorDescriptor& rangeDescriptor,
        const SphericalFunctionInformation<data_t>& sf_info)
    {
        std::unique_ptr<DataContainer<data_t>> spfWeights;

        if (rangeDescriptor.isParallelBeam())
            spfWeights = computeParallelWeights(rangeDescriptor, sf_info);
        else
            spfWeights = computeConeBeamWeights(rangeDescriptor, sf_info);

        // extract volDescriptor
        const auto& trueFuncWeightsDesc =
            dynamic_cast<const IdenticalBlocksDescriptor&>(spfWeights->getDataDescriptor());
        const auto& weightVolDesc = trueFuncWeightsDesc.getDescriptorOfBlock(0);

        // create sph weight descriptor
        auto sphWeightsDesc =
            std::make_unique<IdenticalBlocksDescriptor>(sf_info.basisCnt, weightVolDesc);
        auto sphWeights = std::make_unique<DataContainer<data_t>>(*sphWeightsDesc);

        SphericalFieldsTransform<data_t> sft(sf_info);

        index_t voxelCount = weightVolDesc.getNumberOfCoefficients();
        auto samplingDirs = static_cast<index_t>(sf_info.dirs.size());
        index_t sphCoeffsCount = sf_info.basisCnt;

        // transpose of mode-4 unfolding of x
        Eigen::Map<const typename SphericalFieldsTransform<data_t>::MatrixXd_t> x4(
            &((spfWeights->storage())[0]), voxelCount, samplingDirs);

        // transpose of mode-4 unfolding of Ax
        Eigen::Map<typename SphericalFieldsTransform<data_t>::MatrixXd_t> Ax4(
            &((sphWeights->storage())[0]), voxelCount, sphCoeffsCount);

        typename SphericalFieldsTransform<data_t>::MatrixXd_t WVt =
            sft.getForwardTransformationMatrix().transpose();

        // perform multiplication in chunks
        index_t step = voxelCount / weightVolDesc.getNumberOfCoefficientsPerDimension()(0);
        for (index_t i = 0; i < voxelCount; i += step) {
            Ax4.middleRows(i, step) = x4.middleRows(i, step) * WVt;
        }

        return sphWeights;
    }

    template <typename data_t>
    std::unique_ptr<DataContainer<data_t>> AXDTOperator<data_t>::computeConeBeamWeights(
        const XGIDetectorDescriptor& rangeDescriptor,
        const SphericalFunctionInformation<data_t>& sf_info)
    {
        // obtain number of reconstructed directions, number and size of measured images
        const index_t dn = sf_info.weights.size();
        const index_t px = rangeDescriptor.getNumberOfCoefficientsPerDimension()[0];
        const index_t py = rangeDescriptor.getNumberOfCoefficientsPerDimension()[1];
        const index_t pn = rangeDescriptor.getNumberOfCoefficientsPerDimension()[2];

        // setup complete block descriptor
        auto projBlockDesc = std::make_unique<IdenticalBlocksDescriptor>(dn, rangeDescriptor);
        // setup factors block
        auto weights = std::make_unique<DataContainer<data_t>>(*projBlockDesc);

        // for every sampling angle, reconstruction volumes and factor caches
        for (index_t i = 0; i < dn; ++i) {
            // obtain dci direction
            DirVec e = sf_info.dirs[static_cast<size_t>(i)];

            if (abs(e.norm() - 1) > 1.0e-5)
                throw std::invalid_argument("direction vector at index " + std::to_string(i)
                                            + " not normalized");

            //#pragma omp parallel for
            for (index_t n = 0; n < pn; ++n) {
                // set index to base for currently processed image and direction
                index_t idx = n * px * py;

                // obtain geometry object for current image
                auto camera = rangeDescriptor.getGeometryAt(n).value();

                // allocate the grating's sensitivity vector
                DirVec t;

                // allocate helper objects for ray computation
                DirVec s;
                IndexVector_t pt(3);

                // allocate factor, to be computed from s, t, and e
                data_t factor = 0;

                // traverse image
                for (index_t y = 0; y < py; ++y) {
                    for (index_t x = 0; x < px; ++x) {
                        // dci direction: set above, normalized by design

                        // beam direction: get ray as provided by projection matrix, catch nans
                        // (yes, that can happen :()
                        pt << x, y, n;
                        auto ray = rangeDescriptor.computeRayFromDetectorCoord(pt);
                        s = ray.direction().cast<data_t>();
                        if (s.hasNaN())
                            throw std::invalid_argument(
                                "computation of ray produced nans (fall back to parallel mode?)");

                        // sensitivity direction: first column of rotation matrix (scaling is forced
                        // to be isotropic, x-direction/horizontal axis)
                        t = (camera.getRotationMatrix().transpose() * rangeDescriptor.getSensDir())
                                .cast<data_t>();
                        ;

                        // compute the factor: (|sxe|<e,t>)^2
                        factor = s.cross(e).norm() * e.dot(t);
                        factor = factor * factor;

                        // apply the factor (the location is x/y/n, but there is no need to compute
                        // that)
                        weights->getBlock(i)[idx++] = factor;
                    }
                }
            }
        }

        return weights;
    }

    template <typename data_t>
    std::unique_ptr<DataContainer<data_t>> AXDTOperator<data_t>::computeParallelWeights(
        const XGIDetectorDescriptor& rangeDescriptor,
        const SphericalFunctionInformation<data_t>& sf_info)
    {
        // obtain number of reconstructed directions, number of measured images
        const index_t dn = sf_info.weights.size();
        const index_t pn = rangeDescriptor.getNumberOfCoefficientsPerDimension()[2];

        // setup complete block descriptor
        IndexVector_t tmp_idx(1);
        tmp_idx << pn;
        auto weightsDesc =
            std::make_unique<IdenticalBlocksDescriptor>(dn, VolumeDescriptor(tmp_idx));
        // setup factors block
        auto weights = std::make_unique<DataContainer<data_t>>(*weightsDesc);

        // for every sampling angle, reconstruction volumes and factor caches
        for (index_t i = 0; i < dn; ++i) {
            // obtain dci direction
            DirVec e = sf_info.dirs[static_cast<size_t>(i)];
            if (abs(e.norm() - 1) > 1.0e-5)
                throw std::invalid_argument("direction vector at index " + std::to_string(i)
                                            + " not normalized");

            //#pragma omp parallel for
            for (index_t n = 0; n < pn; ++n) {
                // obtain geometry object for current image
                const Geometry camera = rangeDescriptor.getGeometryAt(n).value();

                // allocate the grating's sensitivity vector
                DirVec t;

                // allocate helper objects for ray computation
                DirVec s;

                // allocate factor, to be computed from s, t, and e
                data_t factor = 0;

                // dci direction: set above, normalized by design

                // beam direction: last column of rotation matrix (scaling is forced to be
                // isotropic, z-direction/viewing axis)
                s = camera.getRotationMatrix().block(2, 0, 1, 3).transpose().cast<data_t>();
                ;

                // sensitivity direction: first column of rotation matrix (scaling is forced to be
                // isotropic, x-direction/horizontal axis)
                t = (camera.getRotationMatrix().transpose() * rangeDescriptor.getSensDir())
                        .cast<data_t>();
                ;

                // compute the factor: (|sxe|<e,t>)^2
                factor = s.cross(e).norm() * e.dot(t);
                factor = factor * factor;

                // apply the factor (the location is x/y/n, but there is no need to compute that)
                weights->getBlock(i)[n] = factor;
            }
        }
        return weights;
    }

    template <typename data_t>
    SphericalFieldsTransform<data_t>::SphericalFieldsTransform(
        const SphericalFunctionInformation<data_t>& sf_info)
        : sf_info(sf_info)
    {
        index_t maxDegree = sf_info.maxDegree;
        size_t samplingTuplesCount = sf_info.dirs.size();
        index_t coeffColCount = sf_info.basisCnt;
        sphericalHarmonicsBasis = MatrixXd_t(samplingTuplesCount, coeffColCount);
        auto sym = sf_info.symmetry;

        data_t normFac = sqrt(static_cast<data_t>(4.0) * pi_t) / sqrt(sf_info.weights.sum());

        for (size_t i = 0; i < samplingTuplesCount; ++i) {
            index_t j = 0;

            auto dir = sf_info.dirs[i];

            // theta: atan2 returns the elevation, so to get theta = the inclination, we need:
            // inclination = pi/2-elevation azimuth = phi the boost implementation provides
            // spherical harmonics which are normalized under the STANDARD L2 norm on the sphere,
            // i.e. \int_{\mathbb{S}_2} 1 d\Omega(u) = 4pi interpreting the sum of the weights of
            // sphericalFunctionDesc as the measure used on the sphere, we need to adjust the
            // normalization such that the L_2 norm under the used measure is 1 as we want
            // orthonormal spherical harmonics
            auto sh_dir = SH_basis_real(
                maxDegree, (pi_t / static_cast<data_t>(2.0)) + atan2(dir[2], hypot(dir[0], dir[1])),
                atan2(dir[1], dir[0]));

            for (int l = 0; l <= maxDegree; ++l) {
                for (int m = -l; m <= l; ++m) {
                    if ((sym == Symmetry::even && (l % 2) != 0)
                        || (sym == Symmetry::odd && (l % 2) == 0)) {
                        continue;
                    }

                    sphericalHarmonicsBasis(static_cast<index_t>(i), j) =
                        normFac * sh_dir(l * l + l + m);
                    j++;
                }
            }
        }
    }

    template <typename data_t>
    typename SphericalFieldsTransform<data_t>::MatrixXd_t
        SphericalFieldsTransform<data_t>::SphericalFieldsTransform::getForwardTransformationMatrix()
            const
    {
        Eigen::DiagonalMatrix<data_t, Eigen::Dynamic> weights(sf_info.weights.size());
        weights = Eigen::DiagonalMatrix<data_t, Eigen::Dynamic>(sf_info.weights);

        return sphericalHarmonicsBasis.transpose() * weights;
    }

    template class AXDTOperator<float>;
    template class AXDTOperator<double>;
    template struct axdt::SphericalFieldsTransform<float>;
    template struct axdt::SphericalFieldsTransform<double>;
    template struct axdt::SphericalFunctionInformation<float>;
    template struct axdt::SphericalFunctionInformation<double>;
} // namespace elsa