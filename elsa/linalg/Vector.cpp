#include "Vector.h"
#include "ContiguousStorage.h"

#include "reductions/L2.h"
#include "reductions/DotProduct.h"

#include "transforms/Assign.h"
#include "transforms/InplaceAdd.h"
#include "transforms/InplaceSub.h"
#include "transforms/InplaceMul.h"
#include "transforms/InplaceDiv.h"

#include <thrust/complex.h>

#include <ostream>

namespace elsa::linalg
{
    template <class data_t>
    Vector<data_t>::Vector(size_type size) : storage_(size)
    {
    }

    template <class data_t>
    Vector<data_t>::Vector(size_type size, const_reference val) : storage_(size, val)
    {
    }

    template <class data_t>
    Vector<data_t>::Vector(std::initializer_list<data_t> list) : storage_(list.begin(), list.end())
    {
    }

    template <class data_t>
    Vector<data_t>::Vector(const ContiguousStorage<data_t>& storage) : storage_(storage)
    {
    }

    template <class data_t>
    Vector<data_t>::Vector(iterator first, iterator last) : storage_(first, last)
    {
    }

    template <class data_t>
    Vector<data_t>::Vector(RowView<data_t> row) : storage_(row.begin(), row.end())
    {
    }

    template <class data_t>
    Vector<data_t>::Vector(ConstRowView<data_t> row) : storage_(row.begin(), row.end())
    {
    }

    template <class data_t>
    Vector<data_t>::Vector(ColumnView<data_t> col) : storage_(col.begin(), col.end())
    {
    }

    template <class data_t>
    Vector<data_t>::Vector(ConstColumnView<data_t> col) : storage_(col.begin(), col.end())
    {
    }

    template <class data_t>
    Vector<data_t>::Vector(Eigen::Matrix<data_t, Eigen::Dynamic, 1> mat)
        : storage_(mat.data(), mat.data() + mat.size())
    {
    }

    template <class data_t>
    typename Vector<data_t>::size_type Vector<data_t>::size() const
    {
        return storage_.size();
    }

    template <class data_t>
    typename Vector<data_t>::reference Vector<data_t>::operator()(size_type idx)
    {
        return *(begin() + idx);
    }

    template <class data_t>
    typename Vector<data_t>::const_reference Vector<data_t>::operator()(size_type idx) const
    {
        return *(begin() + idx);
    }

    template <class data_t>
    typename Vector<data_t>::reference Vector<data_t>::operator[](size_type idx)
    {
        return *(begin() + idx);
    }

    template <class data_t>
    typename Vector<data_t>::const_reference Vector<data_t>::operator[](size_type idx) const
    {
        return *(begin() + idx);
    }

    template <class data_t>
    typename Vector<data_t>::iterator Vector<data_t>::begin()
    {
        return storage_.begin();
    }

    template <class data_t>
    typename Vector<data_t>::iterator Vector<data_t>::end()
    {
        return storage_.end();
    }

    template <class data_t>
    typename Vector<data_t>::const_iterator Vector<data_t>::begin() const
    {
        return cbegin();
    }

    template <class data_t>
    typename Vector<data_t>::const_iterator Vector<data_t>::end() const
    {
        return cend();
    }

    template <class data_t>
    typename Vector<data_t>::const_iterator Vector<data_t>::cbegin() const
    {
        return storage_.cbegin();
    }

    template <class data_t>
    typename Vector<data_t>::const_iterator Vector<data_t>::cend() const
    {
        return storage_.cend();
    }

    template <class data_t>
    Vector<data_t>& Vector<data_t>::operator=(value_type val)
    {
        elsa::fill(begin(), end(), val);
        return *this;
    }

    template <class data_t>
    Vector<data_t>& Vector<data_t>::operator+=(data_t s)
    {
        elsa::inplaceAddScalar(begin(), end(), s);
        return *this;
    }

    template <class data_t>
    Vector<data_t>& Vector<data_t>::operator-=(data_t s)
    {
        elsa::inplaceSubScalar(begin(), end(), s);
        return *this;
    }

    template <class data_t>
    Vector<data_t>& Vector<data_t>::operator*=(data_t s)
    {
        elsa::inplaceMulScalar(begin(), end(), s);
        return *this;
    }

    template <class data_t>
    Vector<data_t>& Vector<data_t>::operator/=(data_t s)
    {
        elsa::inplaceDivScalar(begin(), end(), s);
        return *this;
    }

    template <class data_t>
    Vector<data_t>& Vector<data_t>::operator+=(const Vector<data_t>& v)
    {
        elsa::inplaceAdd(begin(), end(), v.begin());
        return *this;
    }

    template <class data_t>
    Vector<data_t>& Vector<data_t>::operator-=(const Vector<data_t>& v)
    {
        elsa::inplaceSub(begin(), end(), v.begin());
        return *this;
    }

    template <class data_t>
    Vector<data_t>& Vector<data_t>::operator*=(const Vector<data_t>& v)
    {
        elsa::inplaceMul(begin(), end(), v.begin());
        return *this;
    }

    template <class data_t>
    Vector<data_t>& Vector<data_t>::operator/=(const Vector<data_t>& v)
    {
        elsa::inplaceDiv(begin(), end(), v.begin());
        return *this;
    }

    template <class data_t>
    Vector<data_t> operator+(const Vector<data_t>& v, data_t s)
    {
        Vector<data_t> copy = v;
        copy += s;
        return copy;
    }

    template <class data_t>
    Vector<data_t> operator-(const Vector<data_t>& v, data_t s)
    {
        auto copy = v;
        copy -= s;
        return copy;
    }

    template <class data_t>
    Vector<data_t> operator*(const Vector<data_t>& v, data_t s)
    {
        auto copy = v;
        copy *= s;
        return copy;
    }

    template <class data_t>
    Vector<data_t> operator/(const Vector<data_t>& v, data_t s)
    {
        auto copy = v;
        copy /= s;
        return copy;
    }

    template <class data_t>
    Vector<data_t> operator+(const Vector<data_t>& x, const Vector<data_t>& y)
    {
        auto copy = x;
        copy += y;
        return copy;
    }

    template <class data_t>
    Vector<data_t> operator-(const Vector<data_t>& x, const Vector<data_t>& y)
    {
        auto copy = x;
        copy -= y;
        return copy;
    }

    template <class data_t>
    Vector<data_t> operator*(const Vector<data_t>& x, const Vector<data_t>& y)
    {
        auto copy = x;
        copy *= y;
        return copy;
    }

    template <class data_t>
    Vector<data_t> operator/(const Vector<data_t>& x, const Vector<data_t>& y)
    {
        auto copy = x;
        copy /= y;
        return copy;
    }

    template <class data_t>
    void normalize(Vector<data_t>& v)
    {
        v /= norm(v);
    }

    template <class data_t>
    Vector<data_t> normalized(const Vector<data_t>& v)
    {
        return v / norm(v);
    }

    template <class data_t>
    data_t norm(const Vector<data_t>& v)
    {
        return elsa::l2Norm(v.begin(), v.end());
    }

    template <class data_t>
    data_t dot(const Vector<data_t>& x, const Vector<data_t>& y)
    {
        return elsa::dot(x.begin(), x.end(), y.begin());
    }

    template <class data_t>
    std::ostream& operator<<(std::ostream& stream, const Vector<data_t>& v)
    {
        Eigen::IOFormat fmt(4, 0, ", ", "\n", "[", "]");
        using Map = Eigen::Map<const Eigen::Vector<data_t, Eigen::Dynamic>>;
        Map map(v.data(), v.size(), 1);

        stream << map.format(fmt);

        return stream;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Vector<float>;
    template class Vector<double>;
    template class Vector<std::ptrdiff_t>;
    template class Vector<thrust::complex<float>>;
    template class Vector<thrust::complex<double>>;

    template std::ostream& operator<< <float>(std::ostream&, const Vector<float>&);
    template std::ostream& operator<< <double>(std::ostream&, const Vector<double>&);
    template std::ostream& operator<< <std::ptrdiff_t>(std::ostream&,
                                                       const Vector<std::ptrdiff_t>&);
    template void normalize<float>(Vector<float>&);
    template void normalize<double>(Vector<double>&);
    template void normalize<std::ptrdiff_t>(Vector<std::ptrdiff_t>&);

    template Vector<float> normalized<float>(const Vector<float>&);
    template Vector<double> normalized<double>(const Vector<double>&);
    template Vector<std::ptrdiff_t> normalized<std::ptrdiff_t>(const Vector<std::ptrdiff_t>&);

    template float norm<float>(const Vector<float>&);
    template double norm<double>(const Vector<double>&);
    template std::ptrdiff_t norm<std::ptrdiff_t>(const Vector<std::ptrdiff_t>&);

    template float dot<float>(const Vector<float>&, const Vector<float>&);
    template double dot<double>(const Vector<double>&, const Vector<double>&);
    template std::ptrdiff_t dot<std::ptrdiff_t>(const Vector<std::ptrdiff_t>&,
                                                const Vector<std::ptrdiff_t>&);

#define ELSA_INSTANTIATE_SCALAR_OP(op, type) \
    template Vector<type> operator op<type>(const Vector<type>&, type);

#define ELSA_INSTANTIATE_SCALAR_OPS(op)                     \
    ELSA_INSTANTIATE_SCALAR_OP(op, float)                   \
    ELSA_INSTANTIATE_SCALAR_OP(op, double)                  \
    ELSA_INSTANTIATE_SCALAR_OP(op, thrust::complex<float>)  \
    ELSA_INSTANTIATE_SCALAR_OP(op, thrust::complex<double>) \
    ELSA_INSTANTIATE_SCALAR_OP(op, std::ptrdiff_t)

    ELSA_INSTANTIATE_SCALAR_OPS(+)
    ELSA_INSTANTIATE_SCALAR_OPS(-)
    ELSA_INSTANTIATE_SCALAR_OPS(*)
    ELSA_INSTANTIATE_SCALAR_OPS(/)

#undef ELSA_INSTANTIATE_SCALAR_OP
#undef ELSA_INSTANTIATE_SCALAR_OPS

#define ELSA_INSTANTIATE_OP(op, type) \
    template Vector<type> operator op<type>(const Vector<type>&, const Vector<type>&);

#define ELSA_INSTANTIATE_OPS(op)                     \
    ELSA_INSTANTIATE_OP(op, float)                   \
    ELSA_INSTANTIATE_OP(op, double)                  \
    ELSA_INSTANTIATE_OP(op, thrust::complex<float>)  \
    ELSA_INSTANTIATE_OP(op, thrust::complex<double>) \
    ELSA_INSTANTIATE_OP(op, std::ptrdiff_t)

    ELSA_INSTANTIATE_OPS(+);
    ELSA_INSTANTIATE_OPS(-);
    ELSA_INSTANTIATE_OPS(*);
    ELSA_INSTANTIATE_OPS(/);

#undef ELSA_INSTANTIATE_OP
#undef ELSA_INSTANTIATE_OPS
} // namespace elsa::linalg
