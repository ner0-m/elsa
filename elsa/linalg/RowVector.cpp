#include "RowVector.h"
#include "Vector.h"
#include "transforms/Assign.h"

#include <thrust/complex.h>

namespace elsa::linalg
{
    template <class data_t>
    RowView<data_t>::RowView(iterator first, iterator last) : first(first), last(last)
    {
    }

    template <class data_t>
    typename RowView<data_t>::iterator RowView<data_t>::begin()
    {
        return first;
    }

    template <class data_t>
    typename RowView<data_t>::iterator RowView<data_t>::end()
    {
        return last;
    }

    template <class data_t>
    typename RowView<data_t>::iterator RowView<data_t>::begin() const
    {
        return first;
    }

    template <class data_t>
    typename RowView<data_t>::iterator RowView<data_t>::end() const
    {
        return last;
    }

    template <class data_t>
    typename RowView<data_t>::iterator RowView<data_t>::cbegin() const
    {
        return first;
    }

    template <class data_t>
    typename RowView<data_t>::iterator RowView<data_t>::cend() const
    {
        return last;
    }

    template <class data_t>
    typename RowView<data_t>::size_type RowView<data_t>::size() const
    {
        return thrust::distance(first, last);
    }

    template <class data_t>
    typename RowView<data_t>::reference RowView<data_t>::operator()(size_type idx)
    {
        return *thrust::next(first, idx);
    }

    template <class data_t>
    typename RowView<data_t>::const_reference RowView<data_t>::operator()(size_type idx) const
    {
        return *thrust::next(first, idx);
    }

    template <class data_t>
    typename RowView<data_t>::reference RowView<data_t>::operator[](size_type idx)
    {
        return *thrust::next(first, idx);
    }

    template <class data_t>
    typename RowView<data_t>::const_reference RowView<data_t>::operator[](size_type idx) const
    {
        return *thrust::next(first, idx);
    }

    template <class data_t>
    RowView<data_t>& RowView<data_t>::operator=(value_type val)
    {
        elsa::fill(begin(), end(), val);
        return *this;
    }

    template <class data_t>
    RowView<data_t>& RowView<data_t>::operator=(const Vector<data_t>& v)
    {
        if (size() != v.size()) {
            throw std::invalid_argument("RowView: Assigning vector of different size");
        }

        elsa::assign(v.begin(), v.end(), begin());
        return *this;
    }

    template <class data_t>
    ConstRowView<data_t>::ConstRowView(iterator first, iterator last) : first(first), last(last)
    {
    }

    template <class data_t>
    typename ConstRowView<data_t>::iterator ConstRowView<data_t>::begin() const
    {
        return first;
    }

    template <class data_t>
    typename ConstRowView<data_t>::iterator ConstRowView<data_t>::end() const
    {
        return last;
    }

    template <class data_t>
    typename ConstRowView<data_t>::iterator ConstRowView<data_t>::cbegin() const
    {
        return first;
    }

    template <class data_t>
    typename ConstRowView<data_t>::iterator ConstRowView<data_t>::cend() const
    {
        return last;
    }

    template <class data_t>
    typename ConstRowView<data_t>::size_type ConstRowView<data_t>::size() const
    {
        return thrust::distance(first, last);
    }

    template <class data_t>
    typename ConstRowView<data_t>::const_reference
        ConstRowView<data_t>::operator()(size_type idx) const
    {
        return *thrust::next(first, idx);
    }

    template <class data_t>
    typename ConstRowView<data_t>::const_reference
        ConstRowView<data_t>::operator[](size_type idx) const
    {
        return *thrust::next(first, idx);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class RowView<float>;
    template class RowView<double>;
    template class RowView<std::ptrdiff_t>;
    template class RowView<thrust::complex<float>>;
    template class RowView<thrust::complex<double>>;

    template class ConstRowView<float>;
    template class ConstRowView<double>;
    template class ConstRowView<std::ptrdiff_t>;
    template class ConstRowView<thrust::complex<float>>;
    template class ConstRowView<thrust::complex<double>>;
} // namespace elsa::linalg
