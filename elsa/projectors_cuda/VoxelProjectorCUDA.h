#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#include "elsaDefines.h"
#include "LinearOperator.h"
#include "Geometry.h"
#include "BoundingBox.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"
#include "Luts.hpp"
#include "XrayProjector.h"

#include "ProjectVoxelsCUDA.cuh"

namespace elsa
{
    template <typename data_t, typename Derived>
    class VoxelProjectorCUDA;

    template <typename data_t = real_t>
    class BlobVoxelProjectorCUDA;

    template <typename data_t = real_t>
    class BSplineVoxelProjectorCUDA;

    template <typename data_t = real_t>
    class PhaseContrastBlobVoxelProjectorCUDA;

    template <typename data_t = real_t>
    class PhaseContrastBSplineVoxelProjectorCUDA;

    template <typename data_t>
    struct XrayProjectorInnerTypes<BlobVoxelProjectorCUDA<data_t>> {
        using value_type = data_t;
        using forward_tag = any_projection_tag;
        using backward_tag = any_projection_tag;
    };

    template <typename data_t>
    struct XrayProjectorInnerTypes<BSplineVoxelProjectorCUDA<data_t>> {
        using value_type = data_t;
        using forward_tag = any_projection_tag;
        using backward_tag = any_projection_tag;
    };

    template <typename data_t>
    struct XrayProjectorInnerTypes<PhaseContrastBlobVoxelProjectorCUDA<data_t>> {
        using value_type = data_t;
        using forward_tag = any_projection_tag;
        using backward_tag = any_projection_tag;
    };

    template <typename data_t>
    struct XrayProjectorInnerTypes<PhaseContrastBSplineVoxelProjectorCUDA<data_t>> {
        using value_type = data_t;
        using forward_tag = any_projection_tag;
        using backward_tag = any_projection_tag;
    };

    /**
     * @brief GPU-operator representing the discretized X-ray transform in 2d/3d using Voxel
     * Projection
     *
     * @author Noah Dormann
     *
     * @cite based on SiddonsMethodCUDA
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * Every voxel in the volume is projected to the detector according to the Geometry and each hit
     * detector pixel is added the weighted voxel value.
     *
     * The geometry is represented as a list of projection matrices (see class Geometry), one for
     * each acquisition pose.
     *
     * Forward projection is accomplished using apply(), backward projection using applyAdjoint().
     *
     * Currently only utilizes a single GPU. Volume and images should both fit in device memory at
     * the same time.
     */
    template <typename data_t, typename Derived>
    class VoxelProjectorCUDA : public XrayProjector<Derived>
    {
    public:
        using self_type = VoxelProjectorCUDA<data_t, Derived>;
        using base_type = XrayProjector<Derived>;
        using value_type = typename base_type::value_type;
        using forward_tag = typename base_type::forward_tag;
        using backward_tag = typename base_type::backward_tag;

        /**
         * @brief Constructor for Voxel Projection.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        VoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                           const DetectorDescriptor& rangeDescriptor)
            : base_type(domainDescriptor, rangeDescriptor),
              _detectorDescriptor(static_cast<DetectorDescriptor&>(*_rangeDescriptor)),
              _volumeDescriptor(static_cast<VolumeDescriptor&>(*_domainDescriptor))
        {
            auto dim = static_cast<std::size_t>(_domainDescriptor->getNumberOfDimensions());
            if (dim != static_cast<std::size_t>(_rangeDescriptor->getNumberOfDimensions())) {
                throw LogicError(
                    std::string("VoxelProjectorCUDA: domain and range dimension need to match"));
            }

            if (dim != 2 && dim != 3) {
                throw LogicError("VoxelProjectorCUDA: only supporting 2d/3d operations");
            }

            if (_detectorDescriptor.getNumberOfGeometryPoses() == 0) {
                throw LogicError("VoxelProjectorCUDA: geometry list was empty");
            }

            // allocate device memory and copy extrinsic and projection matrices to device
            // height = dim + 1 because we are using homogenous coordinates
            size_t width = dim, height = dim + 1,
                   depth = asUnsigned(_detectorDescriptor.getNumberOfGeometryPoses());

            // allocate device memory and copy ray origins and the inverse of the significant part
            // of projection matrices to device
            _projMatrices.resize(width * height * depth);
            _extMatrices.resize(width * height * depth);
            auto projMatIter = _projMatrices.begin();
            auto extMatIter = _extMatrices.begin();

            for (const auto& geometry : _detectorDescriptor.getGeometry()) {

                RealMatrix_t P = geometry.getProjectionMatrix();
                RealMatrix_t E = geometry.getExtrinsicMatrix();

                // CUDA also uses a column-major representation, directly transfer matrix
                // transfer projection and extrinsic matrix
                projMatIter = thrust::copy(P.data(), P.data() + P.size(), projMatIter);
                extMatIter = thrust::copy(E.data(), E.data() + E.size(), extMatIter);
            }
        }

        /// destructor
        ~VoxelProjectorCUDA() override = default;

    protected:
        /// project Voxels (i.e. forward projection)
        void forward(const BoundingBox& aabb, const DataContainer<data_t>& x,
                     DataContainer<data_t>& Ax) const
        {
            (void) aabb;
            Timer timeGuard("ProjectVoxelsCUDA", "apply");

            // Set it to zero
            Ax = 0;
            projectVoxels<false>(x, Ax);
        }

        /// back-project Voxels (i.e. forward projection)
        void backward(const BoundingBox& aabb, const DataContainer<data_t>& y,
                      DataContainer<data_t>& Aty) const
        {
            (void) aabb;
            Timer timeGuard("ProjectVoxelsCUDA", "applyAdjoint");

            // Set it to zero
            Aty = 0;
            projectVoxels<true>(Aty, y);
        }

        /// implement the polymorphic clone operation
        VoxelProjectorCUDA<data_t, Derived>* _cloneImpl() const
        {
            return new VoxelProjectorCUDA<data_t, Derived>(_volumeDescriptor, _detectorDescriptor);
        }

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const
        {
            if (!LinearOperator<data_t>::isEqual(other))
                return false;

            auto otherOp = downcast_safe<VoxelProjectorCUDA>(&other);
            return static_cast<bool>(otherOp);
        }

    protected:
        template <bool adjoint>
        void projectVoxels(const DataContainer<data_t>& volumeContainer,
                           const DataContainer<data_t>& sinoContainer) const
        {
            auto dim = _volumeDescriptor.getNumberOfDimensions();
            auto domainDims = _domainDescriptor->getNumberOfCoefficientsPerDimension();
            auto domainDimsui = domainDims.template cast<unsigned int>();
            IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();
            auto rangeDimsui = rangeDims.template cast<unsigned int>();
            auto& volume = volumeContainer.storage();
            auto& sino = sinoContainer.storage();
            auto* dvolume = thrust::raw_pointer_cast(volume.data());
            auto* dsino = thrust::raw_pointer_cast(sino.data());
            const auto* projMat = thrust::raw_pointer_cast(_projMatrices.data());
            const auto* extMat = thrust::raw_pointer_cast(_extMatrices.data());

            auto& lutArray = this->self().getLutArray(dim);
            auto* dlut = thrust::raw_pointer_cast(lutArray.data());

            // assume that all geometries are from the same setup
            const DetectorDescriptor& detectorDesc =
                downcast<DetectorDescriptor>(sinoContainer.getDataDescriptor());
            const Geometry& geometry = detectorDesc.getGeometry()[0];
            real_t sourceDetectorDistance = geometry.getSourceDetectorDistance();

            // Prefetch unified memory
            int device = -1;
            cudaGetDevice(&device);
            cudaMemPrefetchAsync(dvolume, volume.size() * sizeof(data_t), device);
            cudaMemPrefetchAsync(dsino, sino.size() * sizeof(data_t), device);

            // advice lut is readonly
            cudaMemAdvise(dlut, lutArray.size(), cudaMemAdviseSetReadMostly, device);

            if constexpr (adjoint) {
                // Advice, that sinogram is read only
                cudaMemAdvise(dsino, sino.size(), cudaMemAdviseSetReadMostly, device);
                // Advice, that volume is not read only
                cudaMemAdvise(dvolume, volume.size(), cudaMemAdviseUnsetReadMostly, device);
            } else {
                // Advice, that sinogram is not read only
                cudaMemAdvise(dsino, sino.size(), cudaMemAdviseUnsetReadMostly, device);
                // Advice, that volume is read only
                cudaMemAdvise(dvolume, volume.size(), cudaMemAdviseSetReadMostly, device);
            }

            if (dim == 3) {
                dim3 sinogramDims(rangeDimsui[0], rangeDimsui[1], rangeDimsui[2]);
                dim3 volumeDims(domainDimsui[0], domainDimsui[1], domainDimsui[2]);
                if (this->self().projector_type == VoxelHelperCUDA::CLASSIC)
                    ProjectVoxelsCUDA<data_t, 3, adjoint, VoxelHelperCUDA::CLASSIC>::project(
                        volumeDims, sinogramDims, THREADS_PER_BLOCK, const_cast<data_t*>(dvolume),
                        const_cast<data_t*>(dsino), projMat, extMat, const_cast<data_t*>(dlut),
                        this->self().radius(), sourceDetectorDistance);
                else
                    ProjectVoxelsCUDA<data_t, 3, adjoint, VoxelHelperCUDA::DIFFERENTIAL>::project(
                        volumeDims, sinogramDims, THREADS_PER_BLOCK, const_cast<data_t*>(dvolume),
                        const_cast<data_t*>(dsino), projMat, extMat, const_cast<data_t*>(dlut),
                        this->self().radius(), sourceDetectorDistance);
            } else {
                dim3 sinogramDims(rangeDimsui[0], 1, rangeDimsui[1]);
                dim3 volumeDims(domainDimsui[0], domainDimsui[1], 1);
                if (this->self().projector_type == VoxelHelperCUDA::CLASSIC)
                    ProjectVoxelsCUDA<data_t, 2, adjoint, VoxelHelperCUDA::CLASSIC>::project(
                        volumeDims, sinogramDims, THREADS_PER_BLOCK, const_cast<data_t*>(dvolume),
                        const_cast<data_t*>(dsino), projMat, extMat, const_cast<data_t*>(dlut),
                        this->self().radius(), sourceDetectorDistance);
                else
                    ProjectVoxelsCUDA<data_t, 2, adjoint, VoxelHelperCUDA::DIFFERENTIAL>::project(
                        volumeDims, sinogramDims, THREADS_PER_BLOCK, const_cast<data_t*>(dvolume),
                        const_cast<data_t*>(dsino), projMat, extMat, const_cast<data_t*>(dlut),
                        this->self().radius(), sourceDetectorDistance);
            }
            // synchonize because we are using multiple streams
            cudaDeviceSynchronize();
        }

        /// threads per dimension used in the kernel execution configuration
        static const unsigned int THREADS_PER_BLOCK =
            ProjectVoxelsCUDA<data_t>::MAX_THREADS_PER_BLOCK;

        /// Reference to DetectorDescriptor stored in LinearOperator
        DetectorDescriptor& _detectorDescriptor;

        /// Reference to VolumeDescriptor stored in LinearOperator
        VolumeDescriptor& _volumeDescriptor;

    private:
        /// projection matrices; stored column-wise on GPU
        thrust::device_vector<data_t> _projMatrices;

        /// extrinsic matrices; stored column-wise on GPU
        thrust::device_vector<data_t> _extMatrices;

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;

        friend class XrayProjector<self_type>;
    };

    template <typename data_t>
    class BlobVoxelProjectorCUDA : public VoxelProjectorCUDA<data_t, BlobVoxelProjectorCUDA<data_t>>
    {
    public:
        using self_type = BlobVoxelProjectorCUDA<data_t>;

        /**
         * @brief Constructor for Blob Voxel Projection.
         *
         * @param[in] radius the blob radius
         * @param[in] alpha blob hyperparameter
         * @param[in] order blob order
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        BlobVoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                               const DetectorDescriptor& rangeDescriptor, data_t radius,
                               data_t alpha, index_t order);

        /**
         * @brief Constructor for Blob Voxel Projection using default blob parameters.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        BlobVoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                               const DetectorDescriptor& rangeDescriptor);

        data_t radius() const { return _lut.radius(); }

        data_t weight(data_t distance) const { return _lut(std::abs(distance)); }

        auto& getLutArray(int dim) const
        {
            (void) dim;
            return _lutArray;
        }

        /// implement the polymorphic clone operation
        BlobVoxelProjectorCUDA<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

        /// define the projector type to be classic (not differential)
        const VoxelHelperCUDA::PROJECTOR_TYPE projector_type = VoxelHelperCUDA::CLASSIC;

    private:
        thrust::device_vector<data_t> _lutArray;
        ProjectedBlobLut<data_t, 100> _lut;

        using Base = VoxelProjectorCUDA<data_t, BlobVoxelProjectorCUDA<data_t>>;

        friend class XrayProjector<self_type>;
    };

    template <typename data_t>
    class PhaseContrastBlobVoxelProjectorCUDA
        : public VoxelProjectorCUDA<data_t, PhaseContrastBlobVoxelProjectorCUDA<data_t>>
    {
    public:
        using self_type = PhaseContrastBlobVoxelProjectorCUDA<data_t>;

        /**
         * @brief Constructor for Phase Contrast Blob Voxel Projection.
         *
         * @param[in] radius the blob radius
         * @param[in] alpha blob hyperparameter
         * @param[in] order blob order
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        PhaseContrastBlobVoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                                            const DetectorDescriptor& rangeDescriptor,
                                            data_t radius, data_t alpha, index_t order);

        /**
         * @brief Constructor for Phase Contrast Blob Voxel Projection using default blob
         * parameters.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        PhaseContrastBlobVoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                                            const DetectorDescriptor& rangeDescriptor);

        data_t radius() const { return _lut.radius(); }

        data_t weight(data_t distance) const
        {
            return _lut(std::abs(distance)) * math::sgn(distance);
        }

        auto& getLutArray(int dim) const
        {
            if (dim == 2)
                return _lutArray;
            else
                return _lut3DArray;
        }

        /// implement the polymorphic clone operation
        PhaseContrastBlobVoxelProjectorCUDA<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

        /// define the projector type to be differential
        const VoxelHelperCUDA::PROJECTOR_TYPE projector_type = VoxelHelperCUDA::DIFFERENTIAL;

    private:
        ProjectedBlobDerivativeLut<data_t, 100> _lut;
        ProjectedBlobNormalizedGradientLut<data_t, 100> _lut3D;

        /// lut array; stored on GPU
        thrust::device_vector<data_t> _lutArray;
        thrust::device_vector<data_t> _lut3DArray;

        using Base = VoxelProjectorCUDA<data_t, PhaseContrastBlobVoxelProjectorCUDA<data_t>>;

        friend class XrayProjector<self_type>;
    };

    template <typename data_t>
    class BSplineVoxelProjectorCUDA
        : public VoxelProjectorCUDA<data_t, BSplineVoxelProjectorCUDA<data_t>>
    {
    public:
        using self_type = BSplineVoxelProjectorCUDA<data_t>;

        /**
         * @brief Constructor for BSpline Voxel Projection.
         *
         * @param[in] order bspline order
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        BSplineVoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                                  const DetectorDescriptor& rangeDescriptor, index_t order);

        /**
         * @brief Constructor for BSpline Voxel Projection using default order.
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        BSplineVoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                                  const DetectorDescriptor& rangeDescriptor);

        data_t radius() const { return _lut.radius(); }

        data_t weight(data_t distance) const { return _lut(std::abs(distance)); }

        auto& getLutArray(int dim) const
        {
            (void) dim;
            return _lutArray;
        }

        /// implement the polymorphic clone operation
        BSplineVoxelProjectorCUDA<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

        /// define the projector type to be classic (not differential)
        const VoxelHelperCUDA::PROJECTOR_TYPE projector_type = VoxelHelperCUDA::CLASSIC;

    private:
        thrust::device_vector<data_t> _lutArray;
        ProjectedBSplineLut<data_t, 100> _lut;

        using Base = VoxelProjectorCUDA<data_t, BSplineVoxelProjectorCUDA<data_t>>;

        friend class XrayProjector<self_type>;
    };

    template <typename data_t>
    class PhaseContrastBSplineVoxelProjectorCUDA
        : public VoxelProjectorCUDA<data_t, PhaseContrastBSplineVoxelProjectorCUDA<data_t>>
    {
    public:
        using self_type = PhaseContrastBSplineVoxelProjectorCUDA<data_t>;

        /**
         * @brief Constructor for Phase Contrast BSpline Voxel Projection.
         *
         * @param[in] order bspline order
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        PhaseContrastBSplineVoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                                               const DetectorDescriptor& rangeDescriptor,
                                               index_t order);

        /**
         * @brief Constructor for Phase Contrast BSpline Voxel Projection using default order
         *
         * @param[in] domainDescriptor describing the domain of the operator (the volume)
         * @param[in] rangeDescriptor describing the range of the operator (the sinogram)
         *
         * The domain is expected to be 2 or 3 dimensional (volSizeX, volSizeY, [volSizeZ]),
         * the range is expected to be matching the domain (detSizeX, [detSizeY], acqPoses).
         */
        PhaseContrastBSplineVoxelProjectorCUDA(const VolumeDescriptor& domainDescriptor,
                                               const DetectorDescriptor& rangeDescriptor);

        data_t radius() const { return _lut.radius(); }

        data_t weight(data_t distance) const
        {
            return _lut(std::abs(distance)) * math::sgn(distance);
        }

        auto& getLutArray(int dim) const
        {
            if (dim == 2)
                return _lutArray;
            else
                return _lut3DArray;
        }

        /// implement the polymorphic clone operation
        PhaseContrastBSplineVoxelProjectorCUDA<data_t>* _cloneImpl() const;

        /// implement the polymorphic comparison operation
        bool _isEqual(const LinearOperator<data_t>& other) const;

        /// define the projector type to be differential
        const VoxelHelperCUDA::PROJECTOR_TYPE projector_type = VoxelHelperCUDA::DIFFERENTIAL;

    private:
        ProjectedBSplineDerivativeLut<data_t, 100> _lut;
        ProjectedBSplineNormalizedGradientLut<data_t, 100> _lut3D;

        /// lut array; stored on GPU
        thrust::device_vector<data_t> _lutArray;
        thrust::device_vector<data_t> _lut3DArray;

        using Base = VoxelProjectorCUDA<data_t, PhaseContrastBSplineVoxelProjectorCUDA<data_t>>;

        friend class XrayProjector<self_type>;
    };
} // namespace elsa
