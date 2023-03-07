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
        : VoxelProjectorCUDA<data_t, BlobVoxelProjectorCUDA<data_t>>(domainDescriptor,
                                                                     rangeDescriptor),
          _lut(radius, alpha, order)
    {
        // copy lut to device
        auto lutData = _lut.data();
        _lutArray.resize(sizeof(lutData));
        thrust::copy(lutData.begin(), lutData.end(), _lutArray.begin());
    }

    template <typename data_t>
    BlobVoxelProjectorCUDA<data_t>::BlobVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor)
        : BlobVoxelProjectorCUDA(domainDescriptor, rangeDescriptor, 2.f, 10.83f, 2)
    {
    }

    template <typename data_t>
    BlobVoxelProjectorCUDA<data_t>* BlobVoxelProjectorCUDA<data_t>::_cloneImpl() const
    {
        return new BlobVoxelProjectorCUDA<data_t>(this->_volumeDescriptor,
                                                  this->_detectorDescriptor, this->_lut.radius(),
                                                  this->_lut.alpha(), this->_lut.order());
    }

    template <typename data_t>
    bool BlobVoxelProjectorCUDA<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
            return false;

        auto otherOp = downcast_safe<BlobVoxelProjectorCUDA>(&other);
        return static_cast<bool>(otherOp);
    }

    template <typename data_t>
    PhaseContrastBlobVoxelProjectorCUDA<data_t>::PhaseContrastBlobVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        data_t radius, data_t alpha, index_t order)
        : VoxelProjectorCUDA<data_t, PhaseContrastBlobVoxelProjectorCUDA<data_t>>(domainDescriptor,
                                                                                  rangeDescriptor),
          _lut(radius, alpha, order),
          _lut3D(radius, alpha, order)
    {
        // copy lut to device
        auto lutData = _lut.data();
        _lutArray.resize(sizeof(lutData));
        thrust::copy(lutData.begin(), lutData.end(), _lutArray.begin());
        // copy lut3D to device
        auto lut3DData = _lut3D.data();
        _lut3DArray.resize(sizeof(lut3DData));
        thrust::copy(lut3DData.begin(), lut3DData.end(), _lut3DArray.begin());
    }

    template <typename data_t>
    PhaseContrastBlobVoxelProjectorCUDA<data_t>::PhaseContrastBlobVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor)
        : PhaseContrastBlobVoxelProjectorCUDA(domainDescriptor, rangeDescriptor, 2, 10.83, 2)
    {
    }

    template <typename data_t>
    PhaseContrastBlobVoxelProjectorCUDA<data_t>*
        PhaseContrastBlobVoxelProjectorCUDA<data_t>::_cloneImpl() const
    {
        return new PhaseContrastBlobVoxelProjectorCUDA<data_t>(
            this->_volumeDescriptor, this->_detectorDescriptor, this->_lut.radius(),
            this->_lut.alpha(), this->_lut.order());
    }

    template <typename data_t>
    bool PhaseContrastBlobVoxelProjectorCUDA<data_t>::_isEqual(
        const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
            return false;

        auto otherOp = downcast_safe<PhaseContrastBlobVoxelProjectorCUDA>(&other);
        return static_cast<bool>(otherOp);
    }

    template <typename data_t>
    BSplineVoxelProjectorCUDA<data_t>::BSplineVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        index_t order)
        : VoxelProjectorCUDA<data_t, BSplineVoxelProjectorCUDA<data_t>>(domainDescriptor,
                                                                        rangeDescriptor),
          _lut(domainDescriptor.getNumberOfDimensions(), order)
    {
        // copy lut to device
        auto lutData = _lut.data();
        _lutArray.resize(sizeof(lutData));
        thrust::copy(lutData.begin(), lutData.end(), _lutArray.begin());
    }

    template <typename data_t>
    BSplineVoxelProjectorCUDA<data_t>::BSplineVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor)
        : BSplineVoxelProjectorCUDA(domainDescriptor, rangeDescriptor, 3)
    {
    }

    template <typename data_t>
    BSplineVoxelProjectorCUDA<data_t>* BSplineVoxelProjectorCUDA<data_t>::_cloneImpl() const
    {
        return new BSplineVoxelProjectorCUDA<data_t>(this->_volumeDescriptor,
                                                     this->_detectorDescriptor, this->_lut.order());
    }

    template <typename data_t>
    bool BSplineVoxelProjectorCUDA<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
            return false;

        auto otherOp = downcast_safe<BSplineVoxelProjectorCUDA>(&other);
        return static_cast<bool>(otherOp);
    }

    template <typename data_t>
    PhaseContrastBSplineVoxelProjectorCUDA<data_t>::PhaseContrastBSplineVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        index_t order)
        : VoxelProjectorCUDA<data_t, PhaseContrastBSplineVoxelProjectorCUDA<data_t>>(
            domainDescriptor, rangeDescriptor),
          _lut(domainDescriptor.getNumberOfDimensions(), order),
          _lut3D(domainDescriptor.getNumberOfDimensions(), order)
    {
        // copy lut to device
        auto lutData = _lut.data();
        _lutArray.resize(sizeof(lutData));
        thrust::copy(lutData.begin(), lutData.end(), _lutArray.begin());
        // copy lut3D to device
        auto lut3DData = _lut3D.data();
        _lut3DArray.resize(sizeof(lut3DData));
        thrust::copy(lut3DData.begin(), lut3DData.end(), _lut3DArray.begin());
    }

    template <typename data_t>
    PhaseContrastBSplineVoxelProjectorCUDA<data_t>::PhaseContrastBSplineVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor)
        : PhaseContrastBSplineVoxelProjectorCUDA(domainDescriptor, rangeDescriptor, 3)
    {
    }

    template <typename data_t>
    PhaseContrastBSplineVoxelProjectorCUDA<data_t>*
        PhaseContrastBSplineVoxelProjectorCUDA<data_t>::_cloneImpl() const
    {
        return new PhaseContrastBSplineVoxelProjectorCUDA<data_t>(
            this->_volumeDescriptor, this->_detectorDescriptor, this->_lut.order());
    }

    template <typename data_t>
    bool PhaseContrastBSplineVoxelProjectorCUDA<data_t>::_isEqual(
        const LinearOperator<data_t>& other) const
    {
        if (!Base::isEqual(other))
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
