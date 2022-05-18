#pragma once

#include "LinearOperator.h"
#include "BoundingBox.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"
#include "CUDAWrappers.h"

namespace elsa
{
    /// wrapper for CUDA variables required by traversal kernel
    struct CUDAVariablesTraversal {
        CudaStreamWrapper stream;

        virtual ~CUDAVariablesTraversal() = default;
    };

    /// wrapper for CUDA variables required by forward traversal kernel
    struct CUDAVariablesForward : public CUDAVariablesTraversal {
        ~CUDAVariablesForward() override = default;
    };

    /// wrapper for CUDA variables required by adjoint traversal kernel
    struct CUDAVariablesAdjoint : public CUDAVariablesTraversal {
        ~CUDAVariablesAdjoint() override = default;
    };

    struct ForwardProjectionTask {
        BoundingBox _volumeBox;
        BoundingBox _imagePatch;
        std::vector<Interval> _poses;
        index_t _numPoses;
    };

    template <typename data_t>
    class CUDAProjector : public LinearOperator<data_t>
    {
    public:
        /**
         * \brief Determine which part of the volume is responsible for the data measured in the
         * specified part of the image
         *
         * \param[in] startCoordinate the start coordinate of the image part
         * \param[in] endCoordinate the end coordinate of the image part
         *
         * \returns the AABB of the reponsible part of the volume
         */
        virtual BoundingBox
            constrainProjectionSpace(const BoundingBox& sinogramBoundingBox) const = 0;

        // TODO does not check the boxes' bounds -> probably best to make this protected
        virtual void applyConstrained(const DataContainer<data_t>& x, DataContainer<data_t>& Ax,
                                      const ForwardProjectionTask& task,
                                      CUDAVariablesForward& cudaVars) const = 0;

        static inline std::mutex deviceLock = std::mutex();

        /// create and allocate memory for all CUDA variables required by forward traversal kernel
        /// arguments should specify the sizes of the largest chunks which will be traversed
        virtual std::unique_ptr<CUDAVariablesForward>
            setupCUDAVariablesForwardConstrained(IndexVector_t chunkSizeDomain,
                                                 IndexVector_t chunkSizeRange) const = 0;

    protected:
        CUDAProjector(const VolumeDescriptor& domainDescriptor,
                      const DetectorDescriptor& rangeDescriptor, int device = 0)
            : LinearOperator<data_t>{domainDescriptor, rangeDescriptor},
              _volumeDescriptor{domainDescriptor},
              _detectorDescriptor{rangeDescriptor},
              _device{device}
        {
            int deviceCount;
            cudaGetDeviceCount(&deviceCount);

            if (_device >= deviceCount)
                throw std::invalid_argument("CUDAProjector: Tried to select device number "
                                            + std::to_string(_device) + " but only "
                                            + std::to_string(deviceCount) + " devices available");
        }

        /// default copy constructor, hidden from non-derived classes to prevent potential
        /// slicing
        CUDAProjector(const CUDAProjector<data_t>& other)
            : LinearOperator<data_t>{other._volumeDescriptor, other._detectorDescriptor},
              _volumeDescriptor{other._volumeDescriptor},
              _detectorDescriptor{other._detectorDescriptor},
              _device{other._device}
        {
        }

        /// create and allocate memory for all CUDA variables required by forward traversal kernel
        /// arguments should specify the sizes of the largest chunks which will be traversed
        virtual std::unique_ptr<CUDAVariablesForward>
            setupCUDAVariablesForward(IndexVector_t chunkSizeDomain,
                                      IndexVector_t chunkSizeRange) const = 0;

        void containerChunksToPinned(const DataContainer<data_t>& dc, PinnedArray<data_t>& pinned,
                                     const BoundingBox& patchBoundingBox,
                                     const std::vector<Interval>& intervals) const
        {
            containerChunkPinnedMemoryTransfer<PAGEABLE_TO_PINNED>(
                const_cast<DataContainer<data_t>&>(dc), pinned, patchBoundingBox, intervals);
        }

        void containerChunkToPinned(const DataContainer<data_t>& dc, PinnedArray<data_t>& pinned,
                                    const BoundingBox& volumeBoundingBox) const
        {
            BoundingBox patchBoundingBox{RealVector_t(volumeBoundingBox._min.topRows(2)),
                                         RealVector_t(volumeBoundingBox._max.topRows(2))};
            std::vector<Interval> intervals{{static_cast<index_t>(volumeBoundingBox._min[2]),
                                             static_cast<index_t>(volumeBoundingBox._max[2])}};

            containerChunkPinnedMemoryTransfer<PAGEABLE_TO_PINNED>(
                const_cast<DataContainer<data_t>&>(dc), pinned, patchBoundingBox, intervals);
        }

        void pinnedToContainerChunks(DataContainer<data_t>& dc, const PinnedArray<data_t>& pinned,
                                     const BoundingBox& patchBoundingBox,
                                     const std::vector<Interval>& intervals) const
        {
            containerChunkPinnedMemoryTransfer<PINNED_TO_PAGEABLE>(
                dc, const_cast<PinnedArray<data_t>&>(pinned), patchBoundingBox, intervals);
        }

        /// convenience typedef for the Eigen::Matrix data vector
        using DataVector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

        /// Reference to VolumeDescriptor stored in LinearOperator
        const VolumeDescriptor& _volumeDescriptor;

        /// Reference to DetectorDescriptor stored in LinearOperator
        const DetectorDescriptor& _detectorDescriptor;

        const int _device;

    private:
        enum MemcpyDirection { PINNED_TO_PAGEABLE, PAGEABLE_TO_PINNED };

        template <MemcpyDirection dir>
        void containerChunkPinnedMemoryTransfer(DataContainer<data_t>& dc,
                                                PinnedArray<data_t>& pinnedChunk,
                                                const BoundingBox& patchBoundingBox,
                                                const std::vector<Interval>& intervals) const
        {
            const auto& dcDesc = dc.getDataDescriptor();
            IndexVector_t dcSize = dcDesc.getNumberOfCoefficientsPerDimension();

            const auto& arrSize = pinnedChunk._dims;
            IndexVector_t patchMin = IndexVector_t::Zero(3);
            patchMin.topRows(2) = patchBoundingBox._min.template cast<index_t>();

            auto patchOffset = dcDesc.getIndexFromCoordinate(patchMin);
            auto dcColumnSize = dcSize[0];
            auto dcSliceSize = dcSize[1] * dcColumnSize;

            data_t* pinnedData = pinnedChunk.get();
            data_t* dcData = &dc[0];
            data_t* src = dir == PINNED_TO_PAGEABLE ? pinnedData : dcData + patchOffset;
            data_t* dst = dir == PAGEABLE_TO_PINNED ? pinnedData : dcData + patchOffset;
            auto srcColumnSize = dir == PINNED_TO_PAGEABLE ? arrSize[0] : dcColumnSize;
            auto srcSliceSize = dir == PINNED_TO_PAGEABLE ? arrSize[0] * arrSize[1] : dcSliceSize;
            auto dstColumnSize = dir == PAGEABLE_TO_PINNED ? arrSize[0] : dcColumnSize;
            auto dstSliceSize = dir == PAGEABLE_TO_PINNED ? arrSize[0] * arrSize[1] : dcSliceSize;
            index_t srcZ, dstZ;
            index_t arrZ = 0;

            IndexVector_t patchSize =
                (patchBoundingBox._max.topRows(2) - patchBoundingBox._min.topRows(2))
                    .template cast<index_t>();
            // assumes 3d container
            for (const auto& [iStart, iEnd] : intervals) {
                for (index_t dcZ = iStart; dcZ < iEnd; dcZ++) {
                    for (index_t y = 0; y < patchSize[1]; y++) {
                        srcZ = dir == PINNED_TO_PAGEABLE ? arrZ : dcZ;
                        dstZ = dir == PAGEABLE_TO_PINNED ? arrZ : dcZ;
                        // copy chunk to pinned memory column by column
                        std::memcpy((void*) (dst + dstZ * dstSliceSize + y * dstColumnSize),
                                    (const void*) (src + srcZ * srcSliceSize + y * srcColumnSize),
                                    patchSize[0] * sizeof(data_t));
                    }
                    arrZ++;
                }
            }
        }

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;
    };
} // namespace elsa