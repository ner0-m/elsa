#pragma once

#include "LinearOperator.h"
#include "BoundingBox.h"
#include "VolumeDescriptor.h"
#include "DetectorDescriptor.h"
#include "CUDAWrappers.h"

namespace elsa
{
    static std::mutex deviceLock;

    /// wrapper for CUDA variables required by traversal kernel
    template <typename data_t>
    struct CUDAVariablesTraversal {
        CudaStreamWrapper stream;
        PinnedArray<data_t>* _pvolume;
        PinnedArray<data_t>* _psino;

        virtual ~CUDAVariablesTraversal() = default;
    };

    /// wrapper for CUDA variables required by forward traversal kernel
    template <typename data_t>
    struct CUDAVariablesForward : public CUDAVariablesTraversal<data_t> {
        ~CUDAVariablesForward() override = default;
    };

    /// wrapper for CUDA variables required by adjoint traversal kernel
    template <typename data_t>
    struct CUDAVariablesAdjoint : public CUDAVariablesTraversal<data_t> {
        ~CUDAVariablesAdjoint() override = default;
    };

    struct ForwardProjectionTask {
        BoundingBox _volumeBox;
        BoundingBox _imagePatch;
        std::vector<Interval> _poses;
        index_t _numPoses;
        bool _zeroInit{true};
        bool _fetchResult{true};
        index_t _paddedDim{-1};
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
                                      CUDAVariablesForward<data_t>& cudaVars) const = 0;

        /// create and allocate memory for all CUDA variables required by forward traversal kernel
        /// arguments should specify the sizes of the largest chunks which will be traversed
        /// does NOT allocate pinned memory buffers, those should be allocated and attached to the
        /// CUDAVariables object when needed
        virtual std::unique_ptr<CUDAVariablesForward<data_t>>
            setupCUDAVariablesForward(IndexVector_t chunkSizeDomain,
                                      IndexVector_t chunkSizeRange) const = 0;

        virtual std::vector<ForwardProjectionTask>
            getSubtasks(const ForwardProjectionTask& task, const IndexVector_t& maxVolumeDims,
                        const IndexVector_t& maxImageDims) const = 0;

        int getDevice() { return _device; }

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

        void containerChunksToPinned(const DataContainer<data_t>& dc, PinnedArray<data_t>& pinned,
                                     const BoundingBox& patchBoundingBox,
                                     const std::vector<Interval>& intervals) const
        {
            containerChunkPinnedMemoryTransfer<PAGEABLE_TO_PINNED>(
                const_cast<DataContainer<data_t>&>(dc), pinned, patchBoundingBox, intervals);
        }

        void containerChunkToPinned(const DataContainer<data_t>& dc, PinnedArray<data_t>& pinned,
                                    const BoundingBox& volumeBoundingBox, cudaStream_t stream) const
        {
            BoundingBox patchBoundingBox{RealVector_t(volumeBoundingBox._min.topRows(2)),
                                         RealVector_t(volumeBoundingBox._max.topRows(2))};
            std::vector<Interval> intervals{{static_cast<index_t>(volumeBoundingBox._min[2]),
                                             static_cast<index_t>(volumeBoundingBox._max[2])}};

            containerChunkPinnedMemoryTransfer<PAGEABLE_TO_PINNED>(
                const_cast<DataContainer<data_t>&>(dc), pinned, patchBoundingBox, intervals,
                stream);
        }

        void pinnedToContainerChunks(DataContainer<data_t>& dc, const PinnedArray<data_t>& pinned,
                                     const BoundingBox& patchBoundingBox,
                                     const std::vector<Interval>& intervals,
                                     cudaStream_t stream) const
        {
            containerChunkPinnedMemoryTransfer<PINNED_TO_PAGEABLE>(
                dc, const_cast<PinnedArray<data_t>&>(pinned), patchBoundingBox, intervals, stream);
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
                                                const std::vector<Interval>& intervals,
                                                cudaStream_t stream) const
        {
            const auto& dcDesc = dc.getDataDescriptor();
            IndexVector_t dcSize = dcDesc.getNumberOfCoefficientsPerDimension();

            // IndexVector_t patchMin = IndexVector_t::Zero(3);
            // patchMin.topRows(2) = patchBoundingBox._min.template cast<index_t>();
            IndexVector_t patchSize =
                (patchBoundingBox._max - patchBoundingBox._min).template cast<index_t>();

            IndexVector_t zeroPadMin = (patchBoundingBox._min.array() < 0)
                                           .select(-patchBoundingBox._min.template cast<index_t>(),
                                                   IndexVector_t::Zero(2));

            IndexVector_t zeroPadMax =
                patchBoundingBox._max.template cast<index_t>() - dcSize.topRows(2);
            zeroPadMax = (zeroPadMax.array() > 0).select(zeroPadMax, IndexVector_t::Zero(2));
            zeroPadMin.conservativeResize(3);
            zeroPadMax.conservativeResize(3);

            //     auto patchOffset = dcDesc.getIndexFromCoordinate(patchMin);
            //     auto dcColumnSize = dcSize[0];
            //     auto dcSliceSize = dcSize[1] * dcColumnSize;

            //     data_t* pinnedData = pinnedChunk.get();
            //     data_t* dcData = &dc[0];
            //     data_t* src = dir == PINNED_TO_PAGEABLE ? pinnedData : dcData +
            //     patchOffset; data_t* dst = dir == PAGEABLE_TO_PINNED ? pinnedData :
            //     dcData + patchOffset; auto srcColumnSize = dir == PINNED_TO_PAGEABLE
            //     ? patchSize[0] : dcColumnSize; auto srcSliceSize =
            //         dir == PINNED_TO_PAGEABLE ? patchSize[0] * patchSize[1] :
            //         dcSliceSize;
            //     auto dstColumnSize = dir == PAGEABLE_TO_PINNED ? patchSize[0] :
            //     dcColumnSize; auto dstSliceSize =
            //         dir == PAGEABLE_TO_PINNED ? patchSize[0] * patchSize[1] :
            //         dcSliceSize;
            //     index_t srcZ, dstZ;
            //     index_t arrZ = 0;

            //     // assumes 3d container
            //     for (const auto& [iStart, iEnd] : intervals) {
            //         for (index_t dcZ = iStart; dcZ < iEnd; dcZ++) {
            //             for (index_t y = 0; y < patchSize[1]; y++) {
            //                 srcZ = dir == PINNED_TO_PAGEABLE ? arrZ : dcZ;
            //                 dstZ = dir == PAGEABLE_TO_PINNED ? arrZ : dcZ;
            //                 // copy chunk to pinned memory column by column
            //                 std::memcpy((void*) (dst + dstZ * dstSliceSize + y *
            //                 dstColumnSize),
            //                             (const void*) (src + srcZ * srcSliceSize
            //                             + y * srcColumnSize), patchSize[0] *
            //                             sizeof(data_t));
            //             }
            //             arrZ++;
            //         }
            //     }

            auto dcPtr = make_cudaPitchedPtr(static_cast<void*>(&dc[0]), dcSize[0] * sizeof(data_t),
                                             dcSize[0] * sizeof(data_t), dcSize[1]);

            auto pinnedPtr = make_cudaPitchedPtr(static_cast<void*>(pinnedChunk.get()),
                                                 patchSize[0] * sizeof(data_t),
                                                 patchSize[0] * sizeof(data_t), patchSize[1]);

            patchSize.conservativeResize(3);
            IndexVector_t unpaddedSize = patchSize - zeroPadMin - zeroPadMax;
            index_t pinnedZ = 0;
            for (const auto& [iStart, iEnd] : intervals) {
                patchSize[2] = iEnd - iStart;
                zeroPadMin[2] = iStart < 0 ? -iStart : 0;
                zeroPadMax[2] = iEnd > dcSize[2] ? iEnd - dcSize[2] : 0;
                unpaddedSize[2] = patchSize[2] - zeroPadMin[2] - zeroPadMax[2];

                // zero pad regions outside dc bounding box
                for (index_t i = 0; i < 3; i++) {
                    IndexVector_t paddingShape = patchSize;
                    if (zeroPadMin[i] > 0) {
                        paddingShape[i] = zeroPadMin[i];
                        cudaMemset3DAsync(pinnedPtr, 0,
                                          make_cudaExtent(paddingShape[0] * sizeof(data_t),
                                                          paddingShape[1], paddingShape[2]),
                                          stream);
                    }
                    if (zeroPadMax[i] > 0) {
                        paddingShape[i] = zeroPadMax[i];
                        IndexVector_t paddingStart = patchSize;
                        paddingStart[i] = patchSize[i] - zeroPadMax[i];
                        cudaPitchedPtr paddingPtr = pinnedPtr;
                        paddingPtr.ptr =
                            (void*) ((char*) pinnedPtr.ptr
                                     + paddingStart.topRows(i + 1).prod() * sizeof(data_t));
                        cudaMemset3DAsync(paddingPtr, 0,
                                          make_cudaExtent(paddingShape[0] * sizeof(data_t),
                                                          paddingShape[1], paddingShape[2]),
                                          stream);
                    }
                }

                auto ext = make_cudaExtent(unpaddedSize[0] * sizeof(data_t), unpaddedSize[1],
                                           unpaddedSize[2]);
                auto dcPos =
                    make_cudaPos((patchBoundingBox._min[0] + zeroPadMin[0]) * sizeof(data_t),
                                 patchBoundingBox._min[1] + zeroPadMin[1], iStart + zeroPadMin[2]);
                auto pinnedPos =
                    make_cudaPos(zeroPadMin[0], zeroPadMin[1], pinnedZ + zeroPadMin[2]);

                cudaMemcpy3DParms parms = {0};
                parms.dstPtr = dir == PAGEABLE_TO_PINNED ? pinnedPtr : dcPtr;
                parms.dstPos = dir == PAGEABLE_TO_PINNED ? pinnedPos : dcPos;
                parms.srcPtr = dir == PINNED_TO_PAGEABLE ? pinnedPtr : dcPtr;
                parms.srcPos = dir == PINNED_TO_PAGEABLE ? pinnedPos : dcPos;
                parms.extent = ext;
                cudaMemcpy3DAsync(&parms, stream);

                pinnedZ += patchSize[2];
            }
        }

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;
    };
} // namespace elsa