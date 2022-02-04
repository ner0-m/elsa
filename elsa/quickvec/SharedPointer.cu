#include "Defines.cuh"
#include "SharedPointer.cuh"
#include <thrust/complex.h>

namespace quickvec
{

    template <typename T>
    SharedPointer<T>::SharedPointer(size_t size) : _counter(new Counter)
    {
        if (size == 0) {
            _ptr = nullptr;
        } else {
            gpuErrchk(cudaMallocManaged(&_ptr, size));
            cudaDeviceSynchronize();
            (*_counter)++;
        }
    }

    template <typename T>
    SharedPointer<T>::SharedPointer(T* pointer, bool owning) : _ptr(pointer), _counter(new Counter)
    {
        if (_ptr) {
            (*_counter)++;

            // for a non-owning vector increase the counter by one additional which will prevent
            // free of the memory
            if (!owning)
                (*_counter)++;
        }
    }

    template <typename T>
    SharedPointer<T>::SharedPointer(SharedPointer const& other)
        : _ptr(other._ptr), _counter(other._counter)
    {
        (*_counter)++;
    }

    template <typename T>
    SharedPointer<T>::SharedPointer(SharedPointer&& other)
        : _ptr(other._ptr), _counter(other._counter)
    {
        other._counter = nullptr;
        other._ptr = nullptr;
    }

    template <typename T>
    SharedPointer<T>& SharedPointer<T>::operator=(const SharedPointer& other)
    {
        SharedPointer<T>(other).swap(*this);
        return *this;
    }

    template <typename T>
    SharedPointer<T>& SharedPointer<T>::operator=(SharedPointer&& other)
    {
        SharedPointer<T>(other).swap(*this);
        return *this;
    }

    template <typename T>
    SharedPointer<T>::~SharedPointer()
    {
        // TODO: Think about delete for counter of non-owning constructed pointer
        if (_counter && _ptr) {
            (*_counter)--;
            if (_counter->get() == 0) {
                delete _counter;
                gpuErrchk(cudaFree(_ptr));
                cudaDeviceSynchronize();
            }
        }
    }

    template <typename T>
    T& SharedPointer<T>::operator*() const
    {
        assert(_ptr != nullptr);
        return *_ptr;
    }

    template <typename T>
    T* SharedPointer<T>::operator->() const
    {
        assert(_ptr != nullptr);
        return _ptr;
    }

    template <typename T>
    void SharedPointer<T>::swap(SharedPointer& other) noexcept
    {
        std::swap(_ptr, other._ptr);
        std::swap(_counter, other._counter);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SharedPointer<float>;
    template class SharedPointer<thrust::complex<float>>;
    template class SharedPointer<double>;
    template class SharedPointer<thrust::complex<double>>;
    template class SharedPointer<index_t>;

} // namespace quickvec
