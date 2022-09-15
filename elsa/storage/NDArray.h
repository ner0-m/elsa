#pragma once

#include "ContiguousStorage.h"
#include <type_traits>
#include <iterator>

namespace elsa
{
    template <class T>
    class NDArray
    {
    public:
        using element_type = T;
        using value_type = typename std::remove_cv<T>::type;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = element_type*;
        using const_pointer = const element_type*;
        using reference = element_type&;
        using const_reference = const element_type&;
        using iterator = pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;

        NDArray() = default;
        NDArray(const NDArray&) = default;
        NDArray& operator=(const NDArray&) = default;

        NDArray(NDArray&&) = default;
        NDArray& operator=(NDArray&&) = default;

    private:
        ContiguousStorage<T> storage_;
    };
} // namespace elsa
