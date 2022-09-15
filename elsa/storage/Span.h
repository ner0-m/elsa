#pragma once

#include "ContiguousStorage.h"

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace elsa
{
    template <class T>
    class ContiguousStorageView
    {
    public:
        using element_type = T;
        using value_type = typename std::remove_cv<T>::type;

        using size_type = typename ContiguousStorage<element_type>::size_type;
        using difference_type = typename ContiguousStorage<element_type>::difference_type;

        using pointer = typename ContiguousStorage<element_type>::pointer;
        using const_pointer = typename ContiguousStorage<element_type>::const_pointer;

        using reference = typename ContiguousStorage<element_type>::reference;
        using const_reference = typename ContiguousStorage<element_type>::const_reference;

        using iterator = typename ContiguousStorage<element_type>::iterator;
        using const_iterator = typename ContiguousStorage<element_type>::const_iterator;

        ContiguousStorageView(ContiguousStorage<T>& storage) noexcept;

        ContiguousStorageView(ContiguousStorage<T>& storage, size_type start,
                              size_type end) noexcept;

        ContiguousStorageView(const ContiguousStorageView<T>& other)
            : ref_(other.ref_), startIdx_(other.startIdx_), endIdx_(other.endIdx_)
        {
        }
        ContiguousStorageView& operator=(const ContiguousStorageView<T>& other)
        {
            ref_ = other.ref_;
            startIdx_ = other.startIdx_;
            endIdx_ = other.endIdx_;
            return *this;
        }

        ContiguousStorageView(ContiguousStorageView<T>&& storage) noexcept = default;
        ContiguousStorageView& operator=(ContiguousStorageView<T>&& storage) noexcept = default;

        template <typename InputIterator>
        void assign(InputIterator first, InputIterator last);

        bool empty() const noexcept;
        size_type size() const noexcept;

        ContiguousStorage<T>& storage() noexcept;
        const ContiguousStorage<T>& storage() const noexcept;

        reference operator[](size_type idx) noexcept;
        const_reference operator[](size_type idx) const noexcept;

        pointer data() noexcept;
        const_pointer data() const noexcept;

        iterator begin() noexcept;
        iterator end() noexcept;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;

        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        reference front() noexcept;
        const_reference front() const noexcept;

        reference back() noexcept;
        const_reference back() const noexcept;

    private:
        ContiguousStorage<T>& ref_;
        size_type startIdx_;
        size_type endIdx_;
    };

    template <class T>
    ContiguousStorageView<T>::ContiguousStorageView(ContiguousStorage<T>& storage) noexcept
        : ContiguousStorageView(storage, 0, storage.size())
    {
    }

    template <class T>
    ContiguousStorageView<T>::ContiguousStorageView(ContiguousStorage<T>& storage, size_type start,
                                                    size_type end) noexcept
        : ref_(storage), startIdx_(start), endIdx_(end)
    {
    }

    template <class T>
    template <typename InputIterator>
    void ContiguousStorageView<T>::assign(InputIterator first, InputIterator last)
    {
        thrust::copy(first, last, begin());
    }

    template <class T>
    bool ContiguousStorageView<T>::empty() const noexcept
    {
        return size() == 0;
    }

    template <class T>
    typename ContiguousStorageView<T>::size_type ContiguousStorageView<T>::size() const noexcept
    {
        return endIdx_ - startIdx_;
    }

    template <class T>
    ContiguousStorage<T>& ContiguousStorageView<T>::storage() noexcept
    {
        return ref_;
    }

    template <class T>
    const ContiguousStorage<T>& ContiguousStorageView<T>::storage() const noexcept
    {
        return ref_;
    }

    template <class T>
    typename ContiguousStorageView<T>::pointer ContiguousStorageView<T>::data() noexcept
    {
        return storage().data() + startIdx_;
    }

    template <class T>
    typename ContiguousStorageView<T>::const_pointer ContiguousStorageView<T>::data() const noexcept
    {
        return storage().data() + startIdx_;
    }

    template <class T>
    typename ContiguousStorageView<T>::reference
        ContiguousStorageView<T>::operator[](size_type idx) noexcept
    {
        return data()[idx];
    }

    template <class T>
    typename ContiguousStorageView<T>::const_reference
        ContiguousStorageView<T>::operator[](size_type idx) const noexcept
    {
        return data()[idx];
    }

    template <class T>
    typename ContiguousStorageView<T>::const_reference
        ContiguousStorageView<T>::front() const noexcept
    {
        return data()[0];
    }

    template <class T>
    typename ContiguousStorageView<T>::reference ContiguousStorageView<T>::front() noexcept
    {
        return data()[0];
    }

    template <class T>
    typename ContiguousStorageView<T>::reference ContiguousStorageView<T>::back() noexcept
    {
        return data()[size() - 1];
    }

    template <class T>
    typename ContiguousStorageView<T>::const_reference
        ContiguousStorageView<T>::back() const noexcept
    {
        return data()[size() - 1];
    }

    template <class T>
    typename ContiguousStorageView<T>::iterator ContiguousStorageView<T>::begin() noexcept
    {
        return thrust::next(storage().begin(), startIdx_);
    }

    template <class T>
    typename ContiguousStorageView<T>::iterator ContiguousStorageView<T>::end() noexcept
    {
        return thrust::next(storage().begin(), endIdx_);
    }

    template <class T>
    typename ContiguousStorageView<T>::const_iterator
        ContiguousStorageView<T>::begin() const noexcept
    {
        return cbegin();
    }

    template <class T>
    typename ContiguousStorageView<T>::const_iterator ContiguousStorageView<T>::end() const noexcept
    {
        return cend();
    }

    template <class T>
    typename ContiguousStorageView<T>::const_iterator
        ContiguousStorageView<T>::cbegin() const noexcept
    {
        return thrust::next(storage().cbegin(), startIdx_);
    }

    template <class T>
    typename ContiguousStorageView<T>::const_iterator
        ContiguousStorageView<T>::cend() const noexcept
    {
        return thrust::next(storage().cbegin(), endIdx_);
    }
} // namespace elsa
