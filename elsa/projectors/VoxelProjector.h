#pragma once

#include "LinearOperator.h"
#include "DetectorDescriptor.h"
#include "VolumeDescriptor.h"
#include "VoxelComputation.h"
#include "Blobs.h"
#include "BSplines.h"

namespace elsa
{
    template <typename data_t = real_t>
    class BlobVoxelProjector : public LinearOperator<data_t>
    {
    public:
        BlobVoxelProjector(const VolumeDescriptor& domainDescriptor,
                           const DetectorDescriptor& rangeDescriptor,
                           data_t radius = blobs::DEFAULT_RADIUS,
                           data_t alpha = blobs::DEFAULT_ALPHA,
                           index_t order = blobs::DEFAULT_ORDER);

    protected:
        void applyImpl(const elsa::DataContainer<data_t>& x,
                       elsa::DataContainer<data_t>& Ax) const override;

        void applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                              elsa::DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const elsa::LinearOperator<data_t>& other) const override;

        /// implement the polymorphic comparison operation
        BlobVoxelProjector<data_t>* cloneImpl() const override;

    public:
        const ProjectedBlob<data_t> blob;

    private:
        index_t _dim;
    };

    template <typename data_t = real_t>
    class BSplineVoxelProjector : public LinearOperator<data_t>
    {
    public:
        BSplineVoxelProjector(const VolumeDescriptor& domainDescriptor,
                              const DetectorDescriptor& rangeDescriptor,
                              const index_t order = bspline::DEFAULT_ORDER);

    protected:
        void applyImpl(const elsa::DataContainer<data_t>& x,
                       elsa::DataContainer<data_t>& Ax) const override;

        void applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                              elsa::DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const elsa::LinearOperator<data_t>& other) const override;

        /// implement the polymorphic comparison operation
        BSplineVoxelProjector<data_t>* cloneImpl() const override;

    private:
        index_t _dim;

    public:
        const ProjectedBSpline<data_t> bspline;
    };
} // namespace elsa