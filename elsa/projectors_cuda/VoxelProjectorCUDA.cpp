#include "LogGuard.h"
#include "Timer.h"
#include "TypeCasts.hpp"

#include "Logger.h"
#include "VoxelProjectorCUDA.h"

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace elsa
{

    template <typename data_t>
    BlobVoxelProjectorCUDA<data_t>::BlobVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        data_t radius, data_t alpha, index_t order)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          blob(radius, alpha, order),
          _dim(domainDescriptor.getNumberOfDimensions())
    {
        transferGeometries<data_t>(rangeDescriptor.getGeometry(), _projMatrices, _extMatrices);
    }

    template <typename data_t>
    void BlobVoxelProjectorCUDA<data_t>::applyImpl(const elsa::DataContainer<data_t>& x,
                                                   elsa::DataContainer<data_t>& Ax) const
    {
        // Zero out the result
        Ax = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, false, VoxelHelperCUDA::CLASSIC>(
                x, Ax, blob.get_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, false, VoxelHelperCUDA::CLASSIC>(
                x, Ax, blob.get_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t>
    void BlobVoxelProjectorCUDA<data_t>::applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                                                          elsa::DataContainer<data_t>& Aty) const
    {
        // Zero out the result
        Aty = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, true, VoxelHelperCUDA::CLASSIC>(
                Aty, y, blob.get_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, true, VoxelHelperCUDA::CLASSIC>(
                Aty, y, blob.get_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t>
    BlobVoxelProjectorCUDA<data_t>* BlobVoxelProjectorCUDA<data_t>::cloneImpl() const
    {
        return new BlobVoxelProjectorCUDA<data_t>(
            downcast<VolumeDescriptor>(*this->_domainDescriptor),
            downcast<DetectorDescriptor>(*this->_rangeDescriptor), this->blob.radius(),
            this->blob.alpha(), this->blob.order());
    }

    template <typename data_t>
    bool BlobVoxelProjectorCUDA<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherOp = downcast_safe<BlobVoxelProjectorCUDA>(&other);
        return static_cast<bool>(otherOp);
    }

    template <typename data_t>
    PhaseContrastBlobVoxelProjectorCUDA<data_t>::PhaseContrastBlobVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        data_t radius, data_t alpha, index_t order)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          blob(radius, alpha, order),
          _dim(domainDescriptor.getNumberOfDimensions())
    {
        transferGeometries<data_t>(rangeDescriptor.getGeometry(), _projMatrices, _extMatrices);
    }

    template <typename data_t>
    void PhaseContrastBlobVoxelProjectorCUDA<data_t>::applyImpl(
        const elsa::DataContainer<data_t>& x, elsa::DataContainer<data_t>& Ax) const
    {
        // Zero out the result
        Ax = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, false, VoxelHelperCUDA::DIFFERENTIAL>(
                x, Ax, blob.get_derivative_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, false, VoxelHelperCUDA::DIFFERENTIAL>(
                x, Ax, blob.get_normalized_gradient_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t>
    void PhaseContrastBlobVoxelProjectorCUDA<data_t>::applyAdjointImpl(
        const elsa::DataContainer<data_t>& y, elsa::DataContainer<data_t>& Aty) const
    {
        // Zero out the result
        Aty = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, true, VoxelHelperCUDA::DIFFERENTIAL>(
                Aty, y, blob.get_derivative_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, true, VoxelHelperCUDA::DIFFERENTIAL>(
                Aty, y, blob.get_normalized_gradient_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t>
    PhaseContrastBlobVoxelProjectorCUDA<data_t>*
        PhaseContrastBlobVoxelProjectorCUDA<data_t>::cloneImpl() const
    {
        return new PhaseContrastBlobVoxelProjectorCUDA<data_t>(
            downcast<VolumeDescriptor>(*this->_domainDescriptor),
            downcast<DetectorDescriptor>(*this->_rangeDescriptor), this->blob.radius(),
            this->blob.alpha(), this->blob.order());
    }

    template <typename data_t>
    bool PhaseContrastBlobVoxelProjectorCUDA<data_t>::isEqual(
        const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherOp = downcast_safe<PhaseContrastBlobVoxelProjectorCUDA>(&other);
        return static_cast<bool>(otherOp);
    }

    template <typename data_t>
    BSplineVoxelProjectorCUDA<data_t>::BSplineVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        index_t order)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          bspline(domainDescriptor.getNumberOfDimensions(), order),
          _dim(domainDescriptor.getNumberOfDimensions())
    {
        transferGeometries<data_t>(rangeDescriptor.getGeometry(), _projMatrices, _extMatrices);
    }

    template <typename data_t>
    void BSplineVoxelProjectorCUDA<data_t>::applyImpl(const elsa::DataContainer<data_t>& x,
                                                      elsa::DataContainer<data_t>& Ax) const
    {
        // Zero out the result
        Ax = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, false, VoxelHelperCUDA::CLASSIC>(
                x, Ax, bspline.get_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, false, VoxelHelperCUDA::CLASSIC>(
                x, Ax, bspline.get_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t>
    void BSplineVoxelProjectorCUDA<data_t>::applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                                                             elsa::DataContainer<data_t>& Aty) const
    {
        // Zero out the result
        Aty = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, true, VoxelHelperCUDA::CLASSIC>(
                Aty, y, bspline.get_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, true, VoxelHelperCUDA::CLASSIC>(
                Aty, y, bspline.get_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t>
    BSplineVoxelProjectorCUDA<data_t>* BSplineVoxelProjectorCUDA<data_t>::cloneImpl() const
    {
        return new BSplineVoxelProjectorCUDA<data_t>(
            downcast<VolumeDescriptor>(*this->_domainDescriptor),
            downcast<DetectorDescriptor>(*this->_rangeDescriptor), this->bspline.order());
    }

    template <typename data_t>
    bool BSplineVoxelProjectorCUDA<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherOp = downcast_safe<BSplineVoxelProjectorCUDA>(&other);
        return static_cast<bool>(otherOp);
    }

    template <typename data_t>
    PhaseContrastBSplineVoxelProjectorCUDA<data_t>::PhaseContrastBSplineVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        index_t order)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          bspline(domainDescriptor.getNumberOfDimensions(), order),
          _dim(domainDescriptor.getNumberOfDimensions())
    {
        transferGeometries<data_t>(rangeDescriptor.getGeometry(), _projMatrices, _extMatrices);
    }

    template <typename data_t>
    void PhaseContrastBSplineVoxelProjectorCUDA<data_t>::applyImpl(
        const elsa::DataContainer<data_t>& x, elsa::DataContainer<data_t>& Ax) const
    {
        // Zero out the result
        Ax = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, false, VoxelHelperCUDA::DIFFERENTIAL>(
                x, Ax, bspline.get_derivative_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, false, VoxelHelperCUDA::DIFFERENTIAL>(
                x, Ax, bspline.get_normalized_gradient_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t>
    void PhaseContrastBSplineVoxelProjectorCUDA<data_t>::applyAdjointImpl(
        const elsa::DataContainer<data_t>& y, elsa::DataContainer<data_t>& Aty) const
    {
        // Zero out the result
        Aty = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, true, VoxelHelperCUDA::DIFFERENTIAL>(
                Aty, y, bspline.get_derivative_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, true, VoxelHelperCUDA::DIFFERENTIAL>(
                Aty, y, bspline.get_normalized_gradient_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t>
    PhaseContrastBSplineVoxelProjectorCUDA<data_t>*
        PhaseContrastBSplineVoxelProjectorCUDA<data_t>::cloneImpl() const
    {
        return new PhaseContrastBSplineVoxelProjectorCUDA<data_t>(
            downcast<VolumeDescriptor>(*this->_domainDescriptor),
            downcast<DetectorDescriptor>(*this->_rangeDescriptor), this->bspline.order());
    }

    template <typename data_t>
    bool PhaseContrastBSplineVoxelProjectorCUDA<data_t>::isEqual(
        const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherOp = downcast_safe<PhaseContrastBSplineVoxelProjectorCUDA>(&other);
        return static_cast<bool>(otherOp);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class BlobVoxelProjectorCUDA<float>;
    template class BlobVoxelProjectorCUDA<double>;

    template class BSplineVoxelProjectorCUDA<float>;
    template class BSplineVoxelProjectorCUDA<double>;

    template class PhaseContrastBlobVoxelProjectorCUDA<float>;
    template class PhaseContrastBlobVoxelProjectorCUDA<double>;

    template class PhaseContrastBSplineVoxelProjectorCUDA<float>;
    template class PhaseContrastBSplineVoxelProjectorCUDA<double>;
} // namespace elsa
