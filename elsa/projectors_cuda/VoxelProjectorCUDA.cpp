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

    template <typename data_t, size_t N>
    BlobVoxelProjectorCUDA<data_t, N>::BlobVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        data_t radius, data_t alpha, index_t order)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          blob(radius, alpha, order),
          _dim(domainDescriptor.getNumberOfDimensions())
    {
        transferGeometries<data_t>(rangeDescriptor.getGeometry(), _projMatrices, _extMatrices);
    }

    template <typename data_t, size_t N>
    void BlobVoxelProjectorCUDA<data_t, N>::applyImpl(const elsa::DataContainer<data_t>& x,
                                                      elsa::DataContainer<data_t>& Ax) const
    {
        // Zero out the result
        Ax = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, false, VoxelHelperCUDA::CLASSIC, N>(
                x, Ax, blob.get_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, false, VoxelHelperCUDA::CLASSIC, N>(
                x, Ax, blob.get_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t, size_t N>
    void BlobVoxelProjectorCUDA<data_t, N>::applyAdjointImpl(const elsa::DataContainer<data_t>& y,
                                                             elsa::DataContainer<data_t>& Aty) const
    {
        // Zero out the result
        Aty = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, true, VoxelHelperCUDA::CLASSIC, N>(
                Aty, y, blob.get_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, true, VoxelHelperCUDA::CLASSIC, N>(
                Aty, y, blob.get_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t, size_t N>
    PhaseContrastBlobVoxelProjectorCUDA<data_t, N>::PhaseContrastBlobVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        data_t radius, data_t alpha, index_t order)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          blob(radius, alpha, order),
          _dim(domainDescriptor.getNumberOfDimensions())
    {
        transferGeometries<data_t>(rangeDescriptor.getGeometry(), _projMatrices, _extMatrices);
    }

    template <typename data_t, size_t N>
    void PhaseContrastBlobVoxelProjectorCUDA<data_t, N>::applyImpl(
        const elsa::DataContainer<data_t>& x, elsa::DataContainer<data_t>& Ax) const
    {
        // Zero out the result
        Ax = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, false, VoxelHelperCUDA::DIFFERENTIAL, N>(
                x, Ax, blob.get_derivative_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, false, VoxelHelperCUDA::DIFFERENTIAL, N>(
                x, Ax, blob.get_normalized_gradient_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t, size_t N>
    void PhaseContrastBlobVoxelProjectorCUDA<data_t, N>::applyAdjointImpl(
        const elsa::DataContainer<data_t>& y, elsa::DataContainer<data_t>& Aty) const
    {
        // Zero out the result
        Aty = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, true, VoxelHelperCUDA::DIFFERENTIAL, N>(
                Aty, y, blob.get_derivative_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, true, VoxelHelperCUDA::DIFFERENTIAL, N>(
                Aty, y, blob.get_normalized_gradient_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t, size_t N>
    BSplineVoxelProjectorCUDA<data_t, N>::BSplineVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        index_t order)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          bspline(domainDescriptor.getNumberOfDimensions(), order),
          _dim(domainDescriptor.getNumberOfDimensions())
    {
        transferGeometries<data_t>(rangeDescriptor.getGeometry(), _projMatrices, _extMatrices);
    }

    template <typename data_t, size_t N>
    void BSplineVoxelProjectorCUDA<data_t, N>::applyImpl(const elsa::DataContainer<data_t>& x,
                                                         elsa::DataContainer<data_t>& Ax) const
    {
        // Zero out the result
        Ax = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, false, VoxelHelperCUDA::CLASSIC, N>(
                x, Ax, bspline.get_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, false, VoxelHelperCUDA::CLASSIC, N>(
                x, Ax, bspline.get_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t, size_t N>
    void BSplineVoxelProjectorCUDA<data_t, N>::applyAdjointImpl(
        const elsa::DataContainer<data_t>& y, elsa::DataContainer<data_t>& Aty) const
    {
        // Zero out the result
        Aty = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, true, VoxelHelperCUDA::CLASSIC, N>(
                Aty, y, bspline.get_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, true, VoxelHelperCUDA::CLASSIC, N>(
                Aty, y, bspline.get_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t, size_t N>
    PhaseContrastBSplineVoxelProjectorCUDA<data_t, N>::PhaseContrastBSplineVoxelProjectorCUDA(
        const VolumeDescriptor& domainDescriptor, const DetectorDescriptor& rangeDescriptor,
        index_t order)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          bspline(domainDescriptor.getNumberOfDimensions(), order),
          _dim(domainDescriptor.getNumberOfDimensions())
    {
        transferGeometries<data_t>(rangeDescriptor.getGeometry(), _projMatrices, _extMatrices);
    }

    template <typename data_t, size_t N>
    void PhaseContrastBSplineVoxelProjectorCUDA<data_t, N>::applyImpl(
        const elsa::DataContainer<data_t>& x, elsa::DataContainer<data_t>& Ax) const
    {
        // Zero out the result
        Ax = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, false, VoxelHelperCUDA::DIFFERENTIAL, N>(
                x, Ax, bspline.get_derivative_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, false, VoxelHelperCUDA::DIFFERENTIAL, N>(
                x, Ax, bspline.get_normalized_gradient_lut(), _projMatrices, _extMatrices);
    }

    template <typename data_t, size_t N>
    void PhaseContrastBSplineVoxelProjectorCUDA<data_t, N>::applyAdjointImpl(
        const elsa::DataContainer<data_t>& y, elsa::DataContainer<data_t>& Aty) const
    {
        // Zero out the result
        Aty = 0;

        if (_dim == 2)
            projectVoxelsCUDA<data_t, 2, true, VoxelHelperCUDA::DIFFERENTIAL, N>(
                Aty, y, bspline.get_derivative_lut(), _projMatrices, _extMatrices);
        else
            projectVoxelsCUDA<data_t, 3, true, VoxelHelperCUDA::DIFFERENTIAL, N>(
                Aty, y, bspline.get_normalized_gradient_lut(), _projMatrices, _extMatrices);
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
