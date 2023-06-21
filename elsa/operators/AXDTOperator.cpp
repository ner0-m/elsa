#include "AXDTOperator.h"
#include "IdenticalBlocksDescriptor.h"
#include "Scaling.h"
#include "Math.hpp"
#include "Timer.h"

using namespace elsa::axdt;

namespace elsa
{
    template <typename data_t>
    AXDTOperator<data_t>::AXDTOperator(const VolumeDescriptor& domainDescriptor,
                                       const XGIDetectorDescriptor& rangeDescriptor,
                                       const LinearOperator<data_t>& projector,
                                       const DirVecList& sphericalFuncDirs,
                                       const Vector_t<data_t>& sphericalFuncWeights,
                                       const Symmetry& sphericalHarmonicsSymmetry,
                                       index_t sphericalHarmonicsMaxDegree)
        : LinearOperator<data_t>{
            IdenticalBlocksDescriptor{SphericalFunctionInformation<data_t>::calculate_basis_cnt(
                                          sphericalHarmonicsSymmetry, sphericalHarmonicsMaxDegree),
                                      domainDescriptor},
            rangeDescriptor}
    {
        Timer timeguard("AXDTOperator", "Construction");

        auto sf_info = SphericalFunctionInformation<data_t>(sphericalFuncDirs, sphericalFuncWeights,
                                                            sphericalHarmonicsSymmetry,
                                                            sphericalHarmonicsMaxDegree);
        bl_op = std::make_unique<BlockLinearOperator<data_t>>(
            IdenticalBlocksDescriptor{sf_info.basisCnt, domainDescriptor}, rangeDescriptor,
            computeOperatorList(rangeDescriptor, sf_info, projector),
            BlockLinearOperator<data_t>::BlockType::COL);
    }

    template <typename data_t>
    AXDTOperator<data_t>::AXDTOperator(const AXDTOperator& other)
        : LinearOperator<data_t>(*other._domainDescriptor, *other._rangeDescriptor)
    {
        bl_op = other.bl_op->clone();
    }

    template <typename data_t>
    AXDTOperator<data_t>* AXDTOperator<data_t>::cloneImpl() const
    {
        return new AXDTOperator<data_t>(*this);
    }

    template <typename data_t>
    bool AXDTOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        // static_cast as type checked in base comparison
        auto otherOp = static_cast<const AXDTOperator<data_t>*>(&other);

        return *bl_op == *(otherOp->bl_op);
    }

    template <typename data_t>
    void AXDTOperator<data_t>::applyImpl(const DataContainer<data_t>& x,
                                         DataContainer<data_t>& Ax) const
    {
        Timer timeguard("AXDTOperator", "apply");
        spdlog::stopwatch timer;

        bl_op->apply(x, Ax);
        //        Logger::get("AXDTOperator")->info("Apply result {}", Ax.sum());
        Logger::get("AXDTOperator")->info("apply(), took {}s", timer);
    }

    template <typename data_t>
    void AXDTOperator<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                DataContainer<data_t>& Aty) const
    {
        Timer timeguard("AXDTOperator", "applyAdjoint");
        spdlog::stopwatch timer;

        bl_op->applyAdjoint(y, Aty);
        //        Logger::get("AXDTOperator")->info("ApplyAdjoint result {}", Aty.sum());
        Logger::get("AXDTOperator")->info("applyAdjoint(), took {}s", timer);
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
            // weights has shape (J, BasisCnt), but J = (#detectorPixels x #measurements) for
            // cone beams and J = (#measurements) for parallel beams (efficiency reasons).
            // At this point we can unify this interface as
            // J = (#detectorPixels x #measurements) by expanding the coalesced weights for
            // parallel beams

            // create a matching scaling operator
            auto scales = [&]() -> std::unique_ptr<Scaling<data_t>> {
                if (rangeDescriptor.isParallelBeam()) {
                    DataContainer<data_t> tmp{rangeDescriptor};
                    index_t totalCnt = rangeDescriptor.getNumberOfCoefficients();
                    index_t blkCnt = rangeDescriptor.getNumberOfCoefficientsPerDimension()
                                         [rangeDescriptor.getNumberOfDimensions() - 1];
                    index_t perBlkCnt = totalCnt / blkCnt;

                    index_t idx = 0;
                    for (index_t j = 0; j < blkCnt; ++j) {
                        for (index_t k = 0; k < perBlkCnt; ++k) {
                            tmp[idx++] = weights->getBlock(i)[j];
                        }
                    }

                    return std::make_unique<Scaling<data_t>>(rangeDescriptor, tmp);
                } else {
                    return std::make_unique<Scaling<data_t>>(rangeDescriptor,
                                                             materialize(weights->getBlock(i)));
                }
            }();

            ops.emplace_back((*scales * projector).clone());
            Logger::get("AXDTOperator")->info("constructed AXDTOperator Block {} / {}", i, numBlocks);
        }

        //        EDF::write<data_t>(*weights,"_weights.edf");
        return ops;
    }

    template <typename data_t>
    std::unique_ptr<DataContainer<data_t>> AXDTOperator<data_t>::computeSphericalHarmonicsWeights(
        const XGIDetectorDescriptor& rangeDescriptor,
        const SphericalFunctionInformation<data_t>& sf_info)
    {
        auto spfWeights = [&]() -> std::unique_ptr<DataContainer<data_t>> {
            if (rangeDescriptor.isParallelBeam())
                return computeParallelWeights(rangeDescriptor, sf_info);
            else
                return computeConeBeamWeights(rangeDescriptor, sf_info);
        }();

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
        auto samplingDirsCnt = static_cast<index_t>(sf_info.dirs.size());
        index_t sphCoeffsCount = sf_info.basisCnt;

        Eigen::Map<const typename SphericalFieldsTransform<data_t>::MatrixXd_t> x(
            &((spfWeights->storage())[0]), voxelCount, samplingDirsCnt);

        Eigen::Map<typename SphericalFieldsTransform<data_t>::MatrixXd_t> Ax(
            &((sphWeights->storage())[0]), voxelCount, sphCoeffsCount);

        typename SphericalFieldsTransform<data_t>::MatrixXd_t transformToSPH =
            sft.getForwardTransformationMatrix();

        // perform multiplication in chunks
        index_t step = voxelCount / weightVolDesc.getNumberOfCoefficientsPerDimension()(0);
        for (index_t i = 0; i < voxelCount; i += step) {
            Ax.middleRows(i, step) = x.middleRows(i, step) * transformToSPH;
        }

        return sphWeights;
    }

    template <typename data_t>
    std::unique_ptr<DataContainer<data_t>> AXDTOperator<data_t>::computeConeBeamWeights(
        const XGIDetectorDescriptor& rangeDescriptor,
        const SphericalFunctionInformation<data_t>& sf_info)
    {
        // obtain number of reconstructed directions, size and number of detector images
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
            // obtain sampling direction
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
                        // sampling direction: set above, normalized by design
                        // beam direction: get ray as provided by projection matrix
                        pt << x, y, n;
                        auto ray = rangeDescriptor.computeRayFromDetectorCoord(pt);
                        s = ray.direction().cast<data_t>();

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

        // set up the coalesced block descriptor
        IndexVector_t tmp_idx(1);
        tmp_idx << pn;
        auto weightsDesc =
            std::make_unique<IdenticalBlocksDescriptor>(dn, VolumeDescriptor(tmp_idx));
        // setup factors block
        auto weights = std::make_unique<DataContainer<data_t>>(*weightsDesc);

        // for every sampling angle, reconstruction volumes and factor caches
        for (index_t i = 0; i < dn; ++i) {
            // obtain sampling direction
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

                // sampling direction: set above, normalized by design

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

        for (index_t i = 0; static_cast<size_t>(i) < samplingTuplesCount; ++i) {
            index_t j = 0;

            auto dir = sf_info.dirs[static_cast<size_t>(i)];

            // theta: atan2 returns the elevation, so to get theta = the inclination, we need:
            // inclination = pi/2-elevation azimuth = phi the boost implementation provides
            // spherical harmonics which are normalized under the STANDARD L2 norm on the sphere,
            // i.e. \int_{\mathbb{S}_2} 1 d\Omega(u) = 4pi interpreting the sum of the weights of
            // sphericalFunctionDesc as the measure used on the sphere, we need to adjust the
            // normalization such that the L_2 norm under the used measure is 1 as we want
            // orthonormal spherical harmonics
            auto sh_dir = SH_basis_real<data_t>(
                maxDegree,
                (pi<data_t> / static_cast<data_t>(2.0) + atan2(dir[2], hypot(dir[0], dir[1]))),
                atan2(dir[1], dir[0]));

            for (int l = 0; l <= maxDegree; ++l) {
                for (int m = -l; m <= l; ++m) {
                    if ((sym == Symmetry::even && (l % 2) != 0)
                        || (sym == Symmetry::odd && (l % 2) == 0)) {
                        continue;
                    }

                    sphericalHarmonicsBasis(i, j) = normFac * sh_dir(l * l + l + m);
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
        auto weights = Eigen::DiagonalMatrix<data_t, Eigen::Dynamic>(sf_info.weights);
        return weights * sphericalHarmonicsBasis;
    }

    template class AXDTOperator<float>;
    template class AXDTOperator<double>;
    template struct axdt::SphericalFieldsTransform<float>;
    template struct axdt::SphericalFieldsTransform<double>;
    template struct axdt::SphericalFunctionInformation<float>;
    template struct axdt::SphericalFunctionInformation<double>;
} // namespace elsa