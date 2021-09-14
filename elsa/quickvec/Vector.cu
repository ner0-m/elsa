// The following ugly workaround is necessary to compile thrust with clang
// =============================

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/complex.h>

#include "Vector.cuh"
#include "Defines.cuh"
#include <complex>
#include <functional>

namespace quickvec
{
    template <typename data_t>
    Vector<data_t>::Vector(Eigen::Matrix<data_t, Eigen::Dynamic, 1> const& data)
        : _size(static_cast<size_t>(data.size())),
          _data(static_cast<size_t>(data.size()) * sizeof(data_t))
    {
        int device = -1;
        gpuErrchk(cudaGetDevice(&device));
        gpuErrchk(
            cudaMemcpy(_data.get(), data.data(), _size * sizeof(data_t), cudaMemcpyHostToDevice));
    };

    template <typename data_t>
    Vector<data_t>::Vector(size_t size) : _size(size), _data(size * sizeof(data_t))
    {
    }

    template <typename data_t>
    Vector<data_t>::Vector(data_t* data, size_t size, bool owning)
        : _size(size), _data(data, owning)
    {
    }

    template <typename data_t>
    Vector<data_t> Vector<data_t>::clone() const
    {
        Vector<data_t> vec(this->size());
        thrust::copy(_data.get(), _data.get() + _size, vec._data.get());
        return vec;
    }

    template <typename data_t>
    Vector<data_t>& Vector<data_t>::operator=(const Vector& other)
    {
        thrust::copy(other._data.get(), other._data.get() + _size, _data.get());
        return *this;
    }

    template <typename data_t>
    Vector<data_t>& Vector<data_t>::operator=(Vector&& other)
    {
        _data = std::move(other._data);
        _size = std::move(other._size);
        return *this;
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> Vector<data_t>::l2Norm() const
    {
        auto norm = squaredl2Norm();
        return std::sqrt(norm);
    }

    // helper struct for L2-norm (note that generic lambdas are not working with thrust)
    template <typename T>
    struct L2Norm {
        __host__ __device__ GetFloatingPointType_t<T> operator()(T const& lhs, T const& rhs)
        {
            return abs(lhs) * abs(rhs);
        }
    };

    template <typename data_t>
    GetFloatingPointType_t<data_t> Vector<data_t>::squaredl2Norm() const
    {
        return thrust::inner_product(_data.get(), _data.get() + _size, _data.get(),
                                     GetFloatingPointType_t<data_t>{0},
                                     std::plus<GetFloatingPointType_t<data_t>>(), L2Norm<data_t>());
    }

    template <typename T>
    struct L1Norm {
        __host__ __device__ GetFloatingPointType_t<T> operator()(T const& lhs, T const& rhs)
        {
            return abs(lhs) + abs(rhs);
        }
    };

    template <typename data_t>
    GetFloatingPointType_t<data_t> Vector<data_t>::l1Norm() const
    {
        return thrust::reduce(_data.get(), _data.get() + _size, GetFloatingPointType_t<data_t>{0},
                              L1Norm<data_t>());
    }

    // helper struct for L0-"norm" for index_t (note that generic lambdas are not working with
    // thrust)
    template <typename T>
    struct L0PseudoNorm {
        static constexpr real_t margin = 0.000001f;
        __host__ __device__ index_t operator()(const T& val) { return abs(val) > margin; }
    };

    template <typename data_t>
    index_t Vector<data_t>::l0PseudoNorm() const
    {
        // return _size - thrust::count(_data.get(), _data.get() + _size, 0);
        return thrust::count_if(_data.get(), _data.get() + _size, L0PseudoNorm<data_t>());
    }

    // helper struct for LInf-norm (note generic lambdas are not working with thrust)
    template <typename T>
    struct LInfNorm {
        __host__ __device__ bool operator()(T const& lhs, T const& rhs)
        {
            return abs(lhs) < abs(rhs);
        }
    };

    template <typename data_t>
    GetFloatingPointType_t<data_t> Vector<data_t>::lInfNorm() const
    {
        return abs(*thrust::max_element(_data.get(), _data.get() + _size, LInfNorm<data_t>()));
    }

    template <typename data_t>
    data_t Vector<data_t>::sum() const
    {
        return thrust::reduce(_data.get(), _data.get() + _size);
    }

    template <typename data_t>
    struct a_dot_conj_b : public thrust::binary_function<data_t, data_t, data_t> {
        __host__ __device__ data_t operator()(data_t a, data_t b) { return a * thrust::conj(b); };
    };

    template <typename data_t>
    data_t Vector<data_t>::dot(const Vector<data_t>& v) const
    {
        if constexpr (isComplex<data_t>) {
            return thrust::conj(
                thrust::inner_product(_data.get(), _data.get() + _size, v._data.get(), data_t{0},
                                      thrust::plus<data_t>(), a_dot_conj_b<data_t>()));
        }
        return thrust::inner_product(_data.get(), _data.get() + _size, v._data.get(), data_t{0});
    }

    template <typename data_t>
    Vector<data_t>& Vector<data_t>::operator+=(const Vector<data_t>& v)
    {
        eval(*this + v);
        return *this;
    }

    template <typename data_t>
    Vector<data_t>& Vector<data_t>::operator-=(const Vector<data_t>& v)
    {
        eval(*this - v);
        return *this;
    }

    template <typename data_t>
    Vector<data_t>& Vector<data_t>::operator*=(const Vector<data_t>& v)
    {
        eval(*this * v);
        return *this;
    }

    template <typename data_t>
    Vector<data_t>& Vector<data_t>::operator/=(const Vector<data_t>& v)
    {
        eval(*this / v);
        return *this;
    }

    template <typename data_t>
    Vector<data_t>& Vector<data_t>::operator+=(data_t scalar)
    {
        eval(*this + scalar);
        return *this;
    }

    template <typename data_t>
    Vector<data_t>& Vector<data_t>::operator-=(data_t scalar)
    {
        eval(*this - scalar);
        return *this;
    }

    template <typename data_t>
    Vector<data_t>& Vector<data_t>::operator*=(data_t scalar)
    {
        eval(*this * scalar);
        return *this;
    }

    template <typename data_t>
    Vector<data_t>& Vector<data_t>::operator/=(data_t scalar)
    {
        eval(*this / scalar);
        return *this;
    }

    template <typename data_t>
    Vector<data_t>& Vector<data_t>::operator=(data_t scalar)
    {
        unsigned int blockSize = 256;
        auto numBlocks = static_cast<unsigned int>((_size + blockSize - 1) / blockSize);

        set<<<numBlocks, blockSize>>>(_size, scalar, _data.get());

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        return *this;
    }

    template <typename data_t>
    bool Vector<data_t>::operator==(Vector<data_t> const& other) const
    {
        return thrust::equal(_data.get(), _data.get() + _size, other._data.get());
    }

    template class Vector<float>;
    template class Vector<double>;
    template class Vector<thrust::complex<float>>;
    template class Vector<thrust::complex<double>>;
    template class Vector<index_t>;

} // namespace quickvec
