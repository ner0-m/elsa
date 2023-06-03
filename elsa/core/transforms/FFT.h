#pragma once

#ifdef ELSA_CUDA_ENABLED
#include <cuda_runtime.h>
#include <cufftXt.h>

#include <thrust/universal_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>

#include <list>
#include <unordered_map>
#include <optional>
#endif

#include "Complex.h"
#include "elsaDefines.h"
#include "Error.h"
#include "DataDescriptor.h"
#include "ContiguousStorage.h"

#if WITH_FFTW
#define EIGEN_FFTW_DEFAULT
#endif
#include <unsupported/Eigen/FFT>

namespace elsa
{
    namespace detail
    {
#ifdef ELSA_CUDA_ENABLED
        cufftResult createPlan(cufftHandle* plan, cufftType type, const IndexVector_t& shape);

        template <typename data_t>
        void fftNormalize(thrust::universal_ptr<data_t> ptr, index_t size, bool applySqrt)
        {
            data_t normalizingFactor = static_cast<data_t>(applySqrt ? std::sqrt(size) : size);
            thrust::transform(ptr, ptr + size, thrust::make_constant_iterator(normalizingFactor),
                              ptr, thrust::divides<data_t>());
        }

        /* Cache is not thread safe, also plans should not be used across multiple threads at the
           same time! Hence, we use a thread local instance. Potential optimizations, should the
           caches GPU memory consumption ever be a problem: Aside from disabling the caching
           mechanism, there are two potential optimizations.
            - Currently, when no new plan can be allocated, the cache is flushed. This could be
           extended also flush the caches of all other threads.
            - cuFFT allows us to manage the work area memory. Before planning,
           cufftSetAutoAllocation(false) could be called, then the work area can be explicitely set.
           This way, all elements of the cache could share a work area, fitting the requirements of
           the largest cached plan. This would increase the management overhead, but reduce memory
           consumption. Currently, the cache has to few elements for this to be worth it.
           */
        class CuFFTPlanCache
        {
        private:
            using CacheElement = std::tuple<cufftHandle, IndexVector_t, cufftType>;
            using CacheList = std::list<CacheElement>;
            CacheList _cache;
            /* this should be very low, to conserve GPU memory! */
            size_t _limit;

            void flush();
            void evict();

        public:
            CuFFTPlanCache();

            CuFFTPlanCache(const CuFFTPlanCache& other) = delete;
            CuFFTPlanCache& operator=(const CuFFTPlanCache& other) = delete;
            /* This performs linear search, because that should be less overhead than a
               map lookup for very small cache sizes*/
            std::optional<cufftHandle> get(cufftType type, const IndexVector_t& shape);
        };

        extern thread_local CuFFTPlanCache cufftCache;

        template <class data_t, bool is_forward>
        bool fftDevice(thrust::universal_ptr<data_t> this_data, const IndexVector_t& src_shape,
                       index_t src_dims, FFTNorm norm)
        {
            /* According to this example:
             * https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuFFT/1d_c2c/1d_c2c_example.cpp
             * it is fine to reinterpret_cast std::complex to cufftComplex
             */

            cufftType type;
            if constexpr (std::is_same<data_t, elsa::complex<float>>::value) {
                type = CUFFT_C2C;
            } else if constexpr (std::is_same<data_t, elsa::complex<double>>::value) {
                type = CUFFT_Z2Z;
            } else {
                /* only single and double precision are supported */
                static_cast<void>(this_data);
                static_cast<void>(src_shape);
                static_cast<void>(src_dims);
                static_cast<void>(norm);
                return false;
            }

            for (size_t i = 0; i < src_dims; i++) {
                if (unlikely(std::numeric_limits<int>::max() < src_shape(i))) {
                    /* e.g. 16GiB one dimensional FFT with complex floats would overflow a signed
                       integer. unlikely, but not impossible. */
                    return false;
                }
            }

            int direction = is_forward ? CUFFT_FORWARD : CUFFT_INVERSE;

            cufftHandle plan;
#ifdef ELSA_CUFFT_CACHE_SIZE != 0
            std::optional<cufftHandle> planOpt = cufftCache.get(type, src_shape);
            if (planOpt) {
                plan = *planOpt;
            } else {
                return false;
            }
#else
            if (createPlan(&plan, type, src_shape) != CUFFT_SUCCESS) {
                return false;
            }
#endif

            bool success;
            if constexpr (std::is_same<data_t, elsa::complex<float>>::value) {
                auto data_ptr = reinterpret_cast<cufftComplex*>(this_data.get());
                /* cuFFT can handle in-place transforms */
                success = cufftExecC2C(plan, data_ptr, data_ptr, direction) == CUFFT_SUCCESS;
            } else {
                auto data_ptr = reinterpret_cast<cufftDoubleComplex*>(this_data.get());
                success = cufftExecZ2Z(plan, data_ptr, data_ptr, direction) == CUFFT_SUCCESS;
            }

            if (likely(success)) {
                /* cuFFT performs unnormalized FFTs, therefor we are left to do scaling according to
                 * FFTNorm */
                if constexpr (is_forward) {
                    if (norm == FFTNorm::FORWARD) {
                        fftNormalize(this_data, src_shape.prod(), false);
                    } else if (norm == FFTNorm::ORTHO) {
                        fftNormalize(this_data, src_shape.prod(), true);
                    }
                } else {
                    if (norm == FFTNorm::BACKWARD) {
                        fftNormalize(this_data, src_shape.prod(), false);
                    } else if (norm == FFTNorm::ORTHO) {
                        fftNormalize(this_data, src_shape.prod(), true);
                    }
                }
            }

#if ELSA_CUFFT_CACHE_SIZE == 0
            cufftDestroy(plan);
#endif

            return success;
        }
#endif

        template <class data_t, bool is_forward>
        void fftHost(data_t* this_data, const IndexVector_t& src_shape, index_t src_dims,
                     FFTNorm norm)
        {
            using DataVector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

            if constexpr (isComplex<data_t>) {
                // TODO: fftw variant

                // generalization of an 1D-FFT
                // walk over each dimension and 1d-fft one 'line' of data
                for (index_t dim_idx = 0; dim_idx < src_dims; ++dim_idx) {
                    // jumps in the data for the current dimension's data
                    // dim_size[0] * dim_size[1] * ...
                    // 1 for dim_idx == 0.
                    const index_t stride = src_shape.head(dim_idx).prod();

                    // number of coefficients for the current dimension
                    const index_t dim_size = src_shape(dim_idx);

                    // number of coefficients for the other dimensions
                    // this is the number of 1d-ffts we'll do
                    // e.g. shape=[2, 3, 4] and we do dim_idx=2 (=shape 4)
                    //   -> src_shape.prod() == 24 / 4 = 6 == 2*3
                    const index_t other_dims_size = src_shape.prod() / dim_size;

#ifndef EIGEN_FFTW_DEFAULT
// when using eigen+fftw, this corrupts the memory, so don't parallelize.
// error messages may include:
// * double free or corruption (fasttop)
// * malloc_consolidate(): unaligned fastbin chunk detected
#pragma omp parallel for
#endif
                    // do all the 1d-ffts along the current dimensions axis
                    for (index_t i = 0; i < other_dims_size; ++i) {

                        index_t ray_start = i;
                        // each time i is a multiple of stride,
                        // jump forward the current+previous dimensions' shape product
                        // (draw an indexed 3d cube to visualize this)
                        ray_start += (stride * (dim_size - 1)) * ((i - (i % stride)) / stride);

                        // this is one "ray" through the volume
                        Eigen::Map<DataVector_t, Eigen::AlignmentType::Unaligned,
                                   Eigen::InnerStride<>>
                            input_map(this_data + ray_start, dim_size,
                                      Eigen::InnerStride<>(stride));

                        using inner_t = GetFloatingPointType_t<typename DataVector_t::Scalar>;

                        Eigen::FFT<inner_t> fft_op;

                        // disable any scaling in eigen - normally it does 1/n for ifft
                        fft_op.SetFlag(Eigen::FFT<inner_t>::Flag::Unscaled);

                        Eigen::Matrix<std::complex<inner_t>, Eigen::Dynamic, 1> fft_in{dim_size};
                        Eigen::Matrix<std::complex<inner_t>, Eigen::Dynamic, 1> fft_out{dim_size};

                        // eigen internally copies the fwd input matrix anyway if
                        // it doesn't have stride == 1
                        fft_in = input_map.block(0, 0, dim_size, 1)
                                     .template cast<std::complex<inner_t>>();

                        if (unlikely(dim_size == 1)) {
                            // eigen kiss-fft crashes for size=1...
                            fft_out = fft_in;
                        } else {
                            // arguments for in and out _must not_ be the same matrix!
                            // they will corrupt wildly otherwise.
                            if constexpr (is_forward) {
                                fft_op.fwd(fft_out, fft_in);
                                if (norm == FFTNorm::FORWARD) {
                                    fft_out /= dim_size;
                                } else if (norm == FFTNorm::ORTHO) {
                                    fft_out /= std::sqrt(dim_size);
                                }
                            } else {
                                fft_op.inv(fft_out, fft_in);
                                if (norm == FFTNorm::BACKWARD) {
                                    fft_out /= dim_size;
                                } else if (norm == FFTNorm::ORTHO) {
                                    fft_out /= std::sqrt(dim_size);
                                }
                            }
                        }

                        // we can't directly use the map as fft output,
                        // since Eigen internally just uses the pointer to
                        // the map's first element, and doesn't respect stride at all..
                        input_map.block(0, 0, dim_size, 1) = fft_out.template cast<data_t>();
                    }
                }
            } else {
                static_cast<void>(this_data);
                static_cast<void>(src_shape);
                static_cast<void>(src_dims);
                static_cast<void>(norm);
                throw Error{"fft with non-complex input container not supported"};
            }
        }
    } // namespace detail

    template <class data_t>
    void fft(ContiguousStorage<data_t>& x, const DataDescriptor& desc, FFTNorm norm,
             bool forceCPU = false)
    {
        const auto& src_shape = desc.getNumberOfCoefficientsPerDimension();
        const auto& src_dims = desc.getNumberOfDimensions();

#ifdef ELSA_CUDA_ENABLED
        if (!forceCPU && detail::fftDevice<data_t, true>(x.data(), src_shape, src_dims, norm)) {
            return;
        }
#else
        static_cast<void>(forceCPU);
#endif
        detail::fftHost<data_t, true>(x.data().get(), src_shape, src_dims, norm);
    }

    template <class data_t>
    void ifft(ContiguousStorage<data_t>& x, const DataDescriptor& desc, FFTNorm norm,
              bool forceCPU = false)
    {
        const auto& src_shape = desc.getNumberOfCoefficientsPerDimension();
        const auto& src_dims = desc.getNumberOfDimensions();
#ifdef ELSA_CUDA_ENABLED
        if (!forceCPU && detail::fftDevice<data_t, false>(x.data(), src_shape, src_dims, norm)) {
            return;
        }
#else
        static_cast<void>(forceCPU);
#endif
        detail::fftHost<data_t, false>(x.data().get(), src_shape, src_dims, norm);
    }
} // namespace elsa
