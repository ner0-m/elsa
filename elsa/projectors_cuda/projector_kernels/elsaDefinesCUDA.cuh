#pragma once

#include "elsaDefines.h"
#include "initializer_list"

#include <cuda_runtime.h>

#include <cstring>
#include <type_traits>
#include <memory>

#ifdef __CUDACC__
#define CUDA_HOST_DEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#define CUDA_INLINE __forceinline__
#else
#define CUDA_HOST_DEV
#define CUDA_HOST
#define CUDA_DEV
#define CUDA_INLINE
#endif

namespace elsa
{
    constexpr static uint32_t MAX_THREADS_PER_BLOCK = 32;

    using Interval = std::pair<index_t, index_t>;

    template <typename Scalar, uint32_t dim>
    class VectorCUDA
    {
    public:
        CUDA_HOST_DEV VectorCUDA(){};

        CUDA_HOST VectorCUDA(std::initializer_list<Scalar> l)
        {
            if (l.size() != dim) {
                throw std::invalid_argument("VectorCUDA of size " + std::to_string(dim)
                                            + " initialized with " + std::to_string(l.size())
                                            + " arguments");
            }

            for (uint32_t i = 0; i < dim; i++)
                _data[i] = l[i];
        }

        CUDA_HOST VectorCUDA(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v)
        {
            if (v.size() > dim)
                throw std::invalid_argument("VectorCUDA of size " + std::to_string(dim)
                                            + " initialized with Eigen::Vector of size "
                                            + std::to_string(v.size()));

            std::memcpy(_data, v.data(), sizeof(Scalar) * v.size());

            // initialize all further elements to 0
            for (auto i = v.size(); i < dim; i++)
                _data[i] = 0;
        }

        VectorCUDA(const VectorCUDA<Scalar, dim>&) = default;

        template <typename int_t, typename = std::enable_if_t<std::is_integral<int_t>::value>>
        CUDA_HOST_DEV CUDA_INLINE const Scalar& operator[](int_t index) const
        {
            return _data[index];
        }

        template <typename int_t, typename = std::enable_if_t<std::is_integral<int_t>::value>>
        CUDA_HOST_DEV CUDA_INLINE Scalar& operator[](int_t index)
        {
            return _data[index];
        }

        CUDA_DEV CUDA_INLINE VectorCUDA<Scalar, dim>&
            operator-=(const VectorCUDA<Scalar, dim>& other)
        {
#pragma unroll
            for (uint32_t i = 0; i < dim; i++)
                _data[i] -= other._data[i];

            return *this;
        }

        CUDA_DEV CUDA_INLINE friend VectorCUDA<Scalar, dim>
            operator-(const VectorCUDA<Scalar, dim>& minuend,
                      const VectorCUDA<Scalar, dim>& subtrahend)
        {
            VectorCUDA<Scalar, dim> difference(minuend);
            difference -= subtrahend;

            return difference;
        }

    private:
        Scalar _data[dim];
    };

    /**
     * \brief A CUDA variant of the BoundingBox class, independent from Eigen
     *
     * \tparam dim the dimensionality of the box
     *
     * Allows for the bounding box to be passed to the kernel by value.
     * Kernel arguments are stored in constant memory, and should generally
     * provide faster access to the variables than via global memory.
     */
    template <uint dim = 3>
    struct BoundingBoxCUDA {
        CUDA_HOST BoundingBoxCUDA(IndexVector_t max)
            : _min(RealVector_t::Zero(dim)), _max(max.template cast<real_t>())
        {
        }

        CUDA_HOST BoundingBoxCUDA(IndexVector_t min, IndexVector_t max)
            : _min(min.template cast<real_t>()), _max(max.template cast<real_t>())
        {
        }

        CUDA_HOST BoundingBoxCUDA(RealVector_t min, RealVector_t max) : _min(min), _max(max) {}

        CUDA_HOST BoundingBoxCUDA(const VectorCUDA<real_t, dim>& min,
                                  const VectorCUDA<real_t, dim>& max)
            : _min(min), _max(max)
        {
        }

        CUDA_HOST BoundingBoxCUDA(const BoundingBoxCUDA<dim>& other)
            : _min(other._min), _max(other._max)
        {
        }

        VectorCUDA<real_t, dim> _min;
        VectorCUDA<real_t, dim> _max;
    };

    template <typename data_t, uint32_t size>
    struct EasyAccessSharedArray {
        data_t* const __restrict__ _p;

        CUDA_DEV EasyAccessSharedArray(data_t* p);

        CUDA_DEV CUDA_INLINE const data_t& operator[](uint32_t index) const
        {
            return _p[index * MAX_THREADS_PER_BLOCK];
        }

        CUDA_DEV CUDA_INLINE data_t& operator[](uint32_t index)
        {
            return _p[index * MAX_THREADS_PER_BLOCK];
        }
    };

} // namespace elsa