#pragma once

#include "ContiguousStorage.h"

#include <type_traits>
#include <cublas_v2.h>
#include <limits>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <mutex>

namespace elsa::cublas
{
    namespace detail
    {
        /* No matter if there exist mutliple cublas-handles, even per thread. Documentation states
         *  that a single handle can be shared by multiple threads, but care should be taken when
         *  modifying the cuda-configuration.  */
        struct CublasState {
        public:
            cublasHandle_t handle = 0;
            std::mutex lock;
            bool initialized = false;

        public:
            ~CublasState()
            {
                if (initialized)
                    cublasDestroy(handle);
                initialized = false;
            }
        };
        thread_local CublasState _cublasState;

        inline bool tryInitCublas()
        {
            std::scoped_lock<std::mutex> _lock(_cublasState.lock);
            if (_cublasState.initialized)
                return true;
            if (cublasCreate(&_cublasState.handle) == CUBLAS_STATUS_SUCCESS)
                _cublasState.initialized = true;
            return _cublasState.initialized;
        }

        template <class ItType>
        using iret_type = thrust::iterator_value_t<ItType>;
        template <class ItType>
        static constexpr bool is_float = std::is_same<iret_type<ItType>, float>::value;
        template <class ItType>
        static constexpr bool is_double = std::is_same<iret_type<ItType>, double>::value;
        template <class ItType>
        static constexpr bool is_fcomplex =
            std::is_same<iret_type<ItType>, thrust::complex<float>>::value;
        template <class ItType>
        static constexpr bool is_dcomplex =
            std::is_same<iret_type<ItType>, thrust::complex<double>>::value;
        template <class ItType>
        static constexpr bool is_cublas_type =
            (is_float<ItType> || is_double<ItType> || is_fcomplex<ItType>
             || is_dcomplex<ItType>) &&thrust::is_contiguous_iterator<ItType>::value;
        template <class ItType, class Scalar>
        static constexpr bool is_element = std::is_same<iret_type<ItType>, Scalar>::value;
        template <class AItType, class BItType>
        static constexpr bool is_compatible = std::is_same<iret_type<AItType>, iret_type<BItType>>::value && thrust::is_contiguous_iterator<BItType>::value;
    } // namespace detail

    template <class ItType, class Scalar>
    bool inplaceMulScalar(ItType begin, ItType end, const Scalar& scalar) 
    {
        if constexpr (!detail::is_cublas_type<ItType> || !detail::is_element<ItType, Scalar>)
            return false;

        size_t count = std::distance(begin, end);
        if (count > std::numeric_limits<int>::max())
            return false;
        if (!detail::tryInitCublas())
            return false;

        cublasStatus_t out = CUBLAS_STATUS_EXECUTION_FAILED;
        if constexpr (detail::is_float<ItType>)
            out = cublasSscal(detail::_cublasState.handle, static_cast<int>(count),
                              (const float*) &scalar, (float*) &(*begin), 1);
        else if constexpr (detail::is_double<ItType>)
            out = cublasDscal(detail::_cublasState.handle, static_cast<int>(count),
                              (const double*) &scalar, (double*) &(*begin), 1);
        else if constexpr (detail::is_fcomplex<ItType>)
            out = cublasCscal(detail::_cublasState.handle, static_cast<int>(count),
                              (const cuComplex*) &scalar, (cuComplex*) &(*begin), 1);
        else if constexpr (detail::is_dcomplex<ItType>)
            out = cublasZscal(detail::_cublasState.handle, static_cast<int>(count),
                              (const cuDoubleComplex*) &scalar, (cuDoubleComplex*) &(*begin), 1);
        return (out == CUBLAS_STATUS_SUCCESS);
    }

    template <class AItType, class BItType, class Scalar>
    bool inplaceDotProduct(AItType aBegin, AItType aEnd, BItType bBegin, Scalar& result)
    {
        if constexpr (!detail::is_cublas_type<AItType> || !detail::is_element<AItType, Scalar>)
            return false;
        if constexpr (!detail::is_compatible<AItType, BItType>)
            return false;

        size_t count = std::distance(aBegin, aEnd);
        if (count > std::numeric_limits<int>::max())
            return false;
        if (!detail::tryInitCublas())
            return false;

        cublasStatus_t out = CUBLAS_STATUS_EXECUTION_FAILED;
        if constexpr (detail::is_float<AItType>)
            out = cublasSdot(detail::_cublasState.handle, static_cast<int>(count), (const float*)&(*aBegin), 1, (const float*)&(*bBegin), 1, (float*)&result);
        else if constexpr (detail::is_double<AItType>)
            out = cublasDdot(detail::_cublasState.handle, static_cast<int>(count), (const double*)&(*aBegin), 1, (const double*)&(*bBegin), 1, (double*)&result);
        else if constexpr (detail::is_fcomplex<AItType>)
            out = cublasCdotc(detail::_cublasState.handle, static_cast<int>(count), (const cuComplex*)&(*aBegin), 1, (const cuComplex*)&(*bBegin), 1, (cuComplex*)&result);
        else if constexpr (detail::is_dcomplex<AItType>)
            out = cublasZdotc(detail::_cublasState.handle, static_cast<int>(count), (const cuDoubleComplex*)&(*aBegin), 1, (const cuDoubleComplex*)&(*bBegin), 1, (cuDoubleComplex*)&result);
        return (out == CUBLAS_STATUS_SUCCESS);
    }
} // namespace elsa::cublas
