#include "SiddonsMethodCUDA.h"
#include "LogGuard.h"
#include "Timer.h"
#include "TypeCasts.hpp"
#include "Logger.h"

#include <thrust/copy.h>
#include <thrust/host_vector.h>

namespace elsa
{
    template <typename data_t>
    SiddonsMethodCUDA<data_t>::SiddonsMethodCUDA(const VolumeDescriptor& domainDescriptor,
                                                 const DetectorDescriptor& rangeDescriptor)
        : base_type(domainDescriptor, rangeDescriptor)
    {
        const auto& detectorDescriptor = static_cast<const DetectorDescriptor&>(*_rangeDescriptor);
        auto dim = static_cast<std::size_t>(_domainDescriptor->getNumberOfDimensions());
        if (dim != static_cast<std::size_t>(_rangeDescriptor->getNumberOfDimensions())) {
            throw LogicError(
                std::string("SiddonsMethodCUDA: domain and range dimension need to match"));
        }

        if (dim != 2 && dim != 3) {
            throw LogicError("SiddonsMethodCUDA: only supporting 2d/3d operations");
        }

        if (detectorDescriptor.getNumberOfGeometryPoses() == 0) {
            throw LogicError("SiddonsMethodCUDA: geometry list was empty");
        }

        const auto numGeometry = asUnsigned(detectorDescriptor.getNumberOfGeometryPoses());

        // allocate device memory and copy ray origins and the inverse of the significant part of
        // projection matrices to device
        _invProjMatrices.resize(dim * dim * numGeometry);
        auto matricesIter = _invProjMatrices.begin();

        // Allocate memory for ray origins
        _rayOrigins.resize(dim * numGeometry);
        auto originsIter = _rayOrigins.begin();

        const auto& poses = detectorDescriptor.getGeometry();
        for (std::size_t i = 0; i < poses.size(); ++i) {
            const auto& geometry = poses[i];

            RealMatrix_t P = geometry.getInverseProjectionMatrix().block(0, 0, dim, dim);

            // CUDA also uses a column-major representation, directly transfer matrix
            // transfer inverse of projection matrix
            matricesIter = thrust::copy(P.data(), P.data() + P.size(), matricesIter);

            // get camera center using direct inverse
            RealVector_t ro = geometry.getCameraCenter();

            // transfer ray origin
            originsIter = thrust::copy(ro.begin(), ro.end(), originsIter);
        }
    }

    template <typename data_t>
    SiddonsMethodCUDA<data_t>* SiddonsMethodCUDA<data_t>::_cloneImpl() const
    {
        return new SiddonsMethodCUDA<data_t>(static_cast<VolumeDescriptor&>(*_domainDescriptor),
                                             static_cast<DetectorDescriptor&>(*_rangeDescriptor));
    }

    template <typename data_t>
    void SiddonsMethodCUDA<data_t>::forward(const BoundingBox& aabb, const DataContainer<data_t>& x,
                                            DataContainer<data_t>& Ax) const
    {
        Timer timeGuard("SiddonsMethodCUDA", "apply");

        // Set it to zero, just to be sure
        Ax = 0;

        traverseForward(aabb, x.storage(), Ax.storage());
    }

    template <typename data_t>
    void SiddonsMethodCUDA<data_t>::backward(const BoundingBox& aabb,
                                             const DataContainer<data_t>& y,
                                             DataContainer<data_t>& Aty) const
    {
        Timer timeguard("SiddonsMethodCUDA", "applyAdjoint");

        // Set it to zero, just to be sure
        Aty = 0;

        traverseBackward(aabb, Aty.storage(), y.storage());
    }

    template <typename data_t>
    bool SiddonsMethodCUDA<data_t>::_isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherSM = downcast_safe<SiddonsMethodCUDA>(&other);
        return static_cast<bool>(otherSM);
    }

    template <typename data_t>
    void SiddonsMethodCUDA<data_t>::traverseForward(const BoundingBox& aabb,
                                                    const thrust::universal_vector<data_t>& volume,
                                                    thrust::universal_vector<data_t>& sino) const
    {
        auto domainDims = _domainDescriptor->getNumberOfCoefficientsPerDimension();
        auto domainDimsui = domainDims.template cast<unsigned int>();
        IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();
        auto rangeDimsui = rangeDims.template cast<unsigned int>();

        if (_domainDescriptor->getNumberOfDimensions() == 3) {
            typename TraverseSiddonsCUDA<data_t, 3>::BoundingBox boundingBox;
            boundingBox._max[0] = aabb.max().template cast<uint32_t>()[0];
            boundingBox._max[1] = aabb.max().template cast<uint32_t>()[1];
            boundingBox._max[2] = aabb.max().template cast<uint32_t>()[2];

            dim3 sinogramDims(rangeDimsui[2], rangeDimsui[1], rangeDimsui[0]);

            TraverseSiddonsCUDA<data_t, 3>::traverseForward(
                sinogramDims, THREADS_PER_BLOCK,
                const_cast<data_t*>(thrust::raw_pointer_cast(volume.data())), domainDimsui[0],
                thrust::raw_pointer_cast(sino.data()), rangeDimsui[0],
                thrust::raw_pointer_cast(_rayOrigins.data()),
                thrust::raw_pointer_cast(_invProjMatrices.data()), boundingBox);
        } else {
            typename TraverseSiddonsCUDA<data_t, 2>::BoundingBox boundingBox;
            boundingBox._max[0] = aabb.max().template cast<uint32_t>()[0];
            boundingBox._max[1] = aabb.max().template cast<uint32_t>()[1];

            // perform projection
            dim3 sinogramDims(rangeDimsui[1], 1, rangeDimsui[0]);

            TraverseSiddonsCUDA<data_t, 2>::traverseForward(
                sinogramDims, THREADS_PER_BLOCK,
                // sinogramDims, 1,
                const_cast<data_t*>(thrust::raw_pointer_cast(volume.data())), domainDimsui[0],
                thrust::raw_pointer_cast(sino.data()), rangeDimsui[0],
                thrust::raw_pointer_cast(_rayOrigins.data()),
                thrust::raw_pointer_cast(_invProjMatrices.data()), boundingBox);
        }
        // synchronize because we are using multiple streams
        cudaDeviceSynchronize();
    }

    template <typename data_t>
    void SiddonsMethodCUDA<data_t>::traverseBackward(
        const BoundingBox& aabb, thrust::universal_vector<data_t>& volume,
        const thrust::universal_vector<data_t>& sino) const
    {
        auto domainDims = _domainDescriptor->getNumberOfCoefficientsPerDimension();
        auto domainDimsui = domainDims.template cast<unsigned int>();
        IndexVector_t rangeDims = _rangeDescriptor->getNumberOfCoefficientsPerDimension();
        auto rangeDimsui = rangeDims.template cast<unsigned int>();

        if (_domainDescriptor->getNumberOfDimensions() == 3) {
            typename TraverseSiddonsCUDA<data_t, 3>::BoundingBox boundingBox;
            boundingBox._max[0] = aabb.max().template cast<uint32_t>()[0];
            boundingBox._max[1] = aabb.max().template cast<uint32_t>()[1];
            boundingBox._max[2] = aabb.max().template cast<uint32_t>()[2];

            dim3 sinogramDims(rangeDimsui[2], rangeDimsui[1], rangeDimsui[0]);

            TraverseSiddonsCUDA<data_t, 3>::traverseAdjoint(
                sinogramDims, THREADS_PER_BLOCK, thrust::raw_pointer_cast(volume.data()),
                domainDimsui[0], const_cast<data_t*>(thrust::raw_pointer_cast(sino.data())),
                rangeDimsui[0], thrust::raw_pointer_cast(_rayOrigins.data()),
                thrust::raw_pointer_cast(_invProjMatrices.data()), boundingBox);
        } else {
            typename TraverseSiddonsCUDA<data_t, 2>::BoundingBox boundingBox;
            boundingBox._max[0] = aabb.max().template cast<uint32_t>()[0];
            boundingBox._max[1] = aabb.max().template cast<uint32_t>()[1];

            // perform projection
            dim3 sinogramDims(rangeDimsui[1], 1, rangeDimsui[0]);

            TraverseSiddonsCUDA<data_t, 2>::traverseAdjoint(
                sinogramDims, THREADS_PER_BLOCK, thrust::raw_pointer_cast(volume.data()),
                domainDimsui[0], const_cast<data_t*>(thrust::raw_pointer_cast(sino.data())),
                rangeDimsui[0], thrust::raw_pointer_cast(_rayOrigins.data()),
                thrust::raw_pointer_cast(_invProjMatrices.data()), boundingBox);
        }
        // synchronize because we are using multiple streams
        cudaDeviceSynchronize();
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SiddonsMethodCUDA<float>;
    template class SiddonsMethodCUDA<double>;
} // namespace elsa
