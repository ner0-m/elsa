#include "ColumnVector.h"
#include "Vector.h"
#include "transforms/Assign.h"

#include <thrust/complex.h>

namespace elsa::linalg
{
    template <class data_t>
    typename ColumnView<data_t>::iterator ColumnView<data_t>::begin()
    {
        return PermutationIterator(first,
                                   TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    template <class data_t>
    typename ColumnView<data_t>::iterator ColumnView<data_t>::end()
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

    template <class data_t>
    typename ColumnView<data_t>::iterator ColumnView<data_t>::begin() const
    {
        return cbegin();
    }

    template <class data_t>
    typename ColumnView<data_t>::iterator ColumnView<data_t>::end() const
    {
        return cend();
    }

    template <class data_t>
    typename ColumnView<data_t>::iterator ColumnView<data_t>::cbegin() const
    {
        return PermutationIterator(first,
                                   TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    template <class data_t>
    typename ColumnView<data_t>::iterator ColumnView<data_t>::cend() const
    {
        return begin() + size();
    }

    template <class data_t>
    typename ColumnView<data_t>::size_type ColumnView<data_t>::size() const
    {
        // `(x + y - 1) / y` is a cheap ceil for integer division
        // Partition size of matrix from first element of the column till the end of the matrix, in
        // sizes of stride
        return (thrust::distance(first, last) + stride - 1) / stride;
    }

    template <class data_t>
    typename ColumnView<data_t>::reference ColumnView<data_t>::operator()(size_type idx)
    {
        return *thrust::next(first, idx);
    }

    template <class data_t>
    typename ColumnView<data_t>::const_reference ColumnView<data_t>::operator()(size_type idx) const
    {
        return *thrust::next(first, idx);
    }

    template <class data_t>
    typename ColumnView<data_t>::reference ColumnView<data_t>::operator[](size_type idx)
    {
        return *thrust::next(first, idx);
    }

    template <class data_t>
    typename ColumnView<data_t>::const_reference ColumnView<data_t>::operator[](size_type idx) const
    {
        return *thrust::next(first, idx);
    }

    template <class data_t>
    ColumnView<data_t>& ColumnView<data_t>::operator=(value_type val)
    {
        elsa::fill(begin(), end(), val);
        return *this;
    }

    template <class data_t>
    ColumnView<data_t>& ColumnView<data_t>::operator=(const Vector<data_t>& v)
    {
        if (size() != v.size()) {
            throw std::invalid_argument("ColumnView: Assigning vector of different size");
        }

        elsa::assign(v.begin(), v.end(), begin());
        return *this;
    }

    template <class data_t>
    typename ConstColumnView<data_t>::iterator ConstColumnView<data_t>::begin() const
    {
        return cbegin();
    }

    template <class data_t>
    typename ConstColumnView<data_t>::iterator ConstColumnView<data_t>::end() const
    {
        return cend();
    }

    template <class data_t>
    typename ConstColumnView<data_t>::iterator ConstColumnView<data_t>::cbegin() const
    {
        return PermutationIterator(first,
                                   TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    template <class data_t>
    typename ConstColumnView<data_t>::iterator ConstColumnView<data_t>::cend() const
    {
        return begin() + size();
    }

    template <class data_t>
    typename ConstColumnView<data_t>::size_type ConstColumnView<data_t>::size() const
    {
        // `(x + y - 1) / y` is a cheap ceil for integer division
        // Partition size of matrix from first element of the column till the end of the matrix, in
        // sizes of stride
        return (thrust::distance(first, last) + stride - 1) / stride;
    }

    template <class data_t>
    typename ConstColumnView<data_t>::const_reference
        ConstColumnView<data_t>::operator()(size_type idx) const
    {
        return *thrust::next(first, idx);
    }

    template <class data_t>
    typename ConstColumnView<data_t>::const_reference
        ConstColumnView<data_t>::operator[](size_type idx) const
    {
        return *thrust::next(first, idx);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ColumnView<float>;
    template class ColumnView<double>;
    template class ColumnView<std::ptrdiff_t>;
    template class ColumnView<thrust::complex<float>>;
    template class ColumnView<thrust::complex<double>>;

    template class ConstColumnView<float>;
    template class ConstColumnView<double>;
    template class ConstColumnView<std::ptrdiff_t>;
    template class ConstColumnView<thrust::complex<float>>;
    template class ConstColumnView<thrust::complex<double>>;
} // namespace elsa::linalg
