#include "Matrix.h"

#include "CUDADefines.h"
#include "transforms/Assign.h"

#include <thrust/complex.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>

#include <stdexcept>

namespace elsa::linalg
{
    template <class data_t>
    Matrix<data_t>::Matrix(const Matrix<data_t>& m) = default;

    template <class data_t>
    Matrix<data_t>& Matrix<data_t>::operator=(const Matrix<data_t>& m) = default;

    template <class data_t>
    Matrix<data_t>::Matrix(Matrix<data_t>&& m) noexcept = default;

    template <class data_t>
    Matrix<data_t>& Matrix<data_t>::operator=(Matrix<data_t>&& m) noexcept = default;

    template <class data_t>
    Matrix<data_t>::Matrix(size_type rows, size_type cols)
        : rows_(rows), cols_(cols), storage_(rows * cols)
    {
    }

    template <class data_t>
    Matrix<data_t>::Matrix(size_type rows, size_type cols, const_reference val)
        : rows_(rows), cols_(cols), storage_(rows * cols, val)
    {
    }

    template <class data_t>
    Matrix<data_t>::Matrix(size_type rows, size_type cols, std::initializer_list<data_t> list)
        : rows_(rows), cols_(cols), storage_(list.begin(), list.end())
    {
        if (list.size() != rows * cols) {
            throw std::invalid_argument(
                "Matrix: initializer list doesn not fit given rows and columns");
        }
    }

    template <class data_t>
    Matrix<data_t>::Matrix(
        const Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat)
        : rows_(mat.rows()), cols_(mat.cols()), storage_(mat.data(), mat.data() + mat.size())
    {
    }

    template <class data_t>
    Matrix<data_t>::Matrix(
        const Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& mat)
        : Matrix(Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(mat))
    {
    }

    template <class data_t>
    Matrix<data_t>& Matrix<data_t>::operator=(value_type val)
    {
        elsa::fill(begin(), end(), val);
        return *this;
    }

    template <class data_t>
    typename Matrix<data_t>::pointer Matrix<data_t>::data()
    {
        return thrust::raw_pointer_cast(storage_.data());
    }

    template <class data_t>
    typename Matrix<data_t>::const_pointer Matrix<data_t>::data() const
    {
        return thrust::raw_pointer_cast(storage_.data());
    }

    template <class data_t>
    typename Matrix<data_t>::iterator Matrix<data_t>::begin()
    {
        return storage_.begin();
    }

    template <class data_t>
    typename Matrix<data_t>::iterator Matrix<data_t>::end()
    {
        return storage_.end();
    }

    template <class data_t>
    typename Matrix<data_t>::const_iterator Matrix<data_t>::begin() const
    {
        return cbegin();
    }

    template <class data_t>
    typename Matrix<data_t>::const_iterator Matrix<data_t>::end() const
    {
        return cend();
    }

    template <class data_t>
    typename Matrix<data_t>::const_iterator Matrix<data_t>::cbegin() const
    {
        return storage_.cbegin();
    }

    template <class data_t>
    typename Matrix<data_t>::const_iterator Matrix<data_t>::cend() const
    {
        return storage_.cend();
    }

    template <class data_t>
    typename Matrix<data_t>::size_type Matrix<data_t>::cols() const
    {
        return cols_;
    }

    template <class data_t>
    typename Matrix<data_t>::size_type Matrix<data_t>::rows() const
    {
        return rows_;
    }

    template <class data_t>
    typename Matrix<data_t>::reference Matrix<data_t>::operator()(size_type i, size_type j)
    {
        return storage_[index(i, j)];
    }

    template <class data_t>
    typename Matrix<data_t>::const_reference Matrix<data_t>::operator()(size_type i,
                                                                        size_type j) const
    {
        return storage_[index(i, j)];
    }

    template <class data_t>
    RowView<data_t> Matrix<data_t>::row(size_type i)
    {
        if (i >= rows_) {
            throw std::invalid_argument("Matrix: trying to access out of bound row");
        }

        auto first = storage_.begin() + index(i, 0);
        return {first, first + cols_};
    }

    template <class data_t>
    ConstRowView<data_t> Matrix<data_t>::row(size_type i) const
    {
        if (i >= rows_) {
            throw std::invalid_argument("Matrix: trying to access out of bound row");
        }

        auto first = storage_.begin() + index(i, 0);
        return {first, first + cols_};
    }

    template <class data_t>
    ColumnView<data_t> Matrix<data_t>::col(size_type i)
    {
        if (i >= cols_) {
            throw std::invalid_argument("Matrix: trying to access out of bound column");
        }

        return {storage_.begin() + i, storage_.end(), cols_};
    }

    template <class data_t>
    ConstColumnView<data_t> Matrix<data_t>::col(size_type i) const
    {
        if (i >= cols_) {
            throw std::invalid_argument("Matrix: trying to access out of bound column");
        }
        return {storage_.begin() + i, storage_.end(), cols_};
    }

    template <class data_t>
    void Matrix<data_t>::reshape(size_type newrows, size_type newcols)
    {
        if (newrows * newcols != rows_ * cols_) {
            throw std::invalid_argument("Matrix: reshape must not change the size of the matrix");
        }

        rows_ = newrows;
        cols_ = newcols;
    }

    template <class data_t>
    void Matrix<data_t>::resize(size_type newrows, size_type newcols)
    {
        // Resize first, if this throws, the size of the matrix hasn't changed
        storage_.resize(newrows * newcols);

        rows_ = newrows;
        cols_ = newcols;
    }

    template <class data_t>
    Matrix<data_t> Matrix<data_t>::transpose() const
    {
        Matrix<data_t> transposed(cols(), rows());

        for (int i = 0; i < rows(); ++i) {
            auto col = transposed.col(i);
            auto row = this->row(i);
            elsa::assign(row.begin(), row.end(), col.begin());
        }

        return transposed;
    }

    template <class data_t>
    std::ostream& operator<<(std::ostream& stream, const Matrix<data_t>& mat)
    {
        Eigen::IOFormat fmt(4, 0, ", ", "\n", "[", "]");
        using Map = Eigen::Map<
            const Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
        Map map(mat.data(), mat.rows(), mat.cols());

        stream << map.format(fmt);

        return stream;
    }

    template <class data_t, class T>
    Vector<data_t> operator*(const Matrix<data_t>& mat, const T& x)
    {
        if (mat.cols() != x.size()) {
            throw std::invalid_argument(
                "Matrix-Vector product: Matrix size doesn not fit the given vector");
        }

        // Allocate memory for result vector
        Vector<data_t> prod(mat.rows());

        // As the structs we are using aren't device ready, we need to handle iterators here, kind
        // of meh, but works for now
        auto matfirst = mat.begin();
        auto cols = mat.cols();
        auto vecfirst = x.begin();

        // In parallel go over each row of the matrix and perform a sequential scalar product for
        // each row with the given vector
        thrust::transform(thrust::device, thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(mat.rows()), prod.begin(),
                          [=] __device__ __host__(std::size_t idx) {
                              // Works as we are row-major
                              auto row = matfirst + idx * cols;
                              return thrust::inner_product(thrust::seq, row, row + cols, vecfirst,
                                                           data_t(0));
                          });

        return prod;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Matrix<float>;
    template class Matrix<double>;
    template class Matrix<std::ptrdiff_t>;
    template class Matrix<thrust::complex<float>>;
    template class Matrix<thrust::complex<double>>;

    template std::ostream& operator<< <float>(std::ostream& stream, const Matrix<float>& mat);
    template std::ostream& operator<< <double>(std::ostream& stream, const Matrix<double>& mat);
    template std::ostream& operator<< <std::ptrdiff_t>(std::ostream& stream,
                                                       const Matrix<std::ptrdiff_t>& mat);

#define ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(type, vtype) \
    template Vector<type> operator*<type, vtype<type>>(const Matrix<type>&, const vtype<type>&);

    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(float, Vector);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(double, Vector);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(std::ptrdiff_t, Vector);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(thrust::complex<float>, Vector);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(thrust::complex<double>, Vector);

    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(float, RowView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(double, RowView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(std::ptrdiff_t, RowView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(thrust::complex<float>, RowView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(thrust::complex<double>, RowView);

    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(float, ConstRowView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(double, ConstRowView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(std::ptrdiff_t, ConstRowView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(thrust::complex<float>, ConstRowView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(thrust::complex<double>, ConstRowView);

    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(float, ColumnView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(double, ColumnView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(std::ptrdiff_t, ColumnView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(thrust::complex<float>, ColumnView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(thrust::complex<double>, ColumnView);

    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(float, ConstColumnView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(double, ConstColumnView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(std::ptrdiff_t, ConstColumnView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(thrust::complex<float>, ConstColumnView);
    ELSA_INSTANTIATE_MATRIX_VECTOR_PROD(thrust::complex<double>, ConstColumnView);

#undef ELSA_INSTANTIATE_MATRIX_VECTOR_PROD
} // namespace elsa::linalg
