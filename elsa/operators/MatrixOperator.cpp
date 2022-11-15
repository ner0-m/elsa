#include "MatrixOperator.h"
#include "Error.h"
#include "RowVector.h"

namespace elsa
{
    template <typename data_t>
    MatrixOperator<data_t>::MatrixOperator(const DataDescriptor& domain,
                                           const DataDescriptor& range,
                                           const linalg::Matrix<data_t>& mat)
        : LinearOperator<data_t>(domain, range), mat_(mat)
    {
        if (domain.getNumberOfCoefficients() != mat.cols()) {
            throw Error("MatrixOperator: domain size doesn't work with given matrix");
        }

        if (range.getNumberOfCoefficients() != mat.rows()) {
            throw Error("MatrixOperator: range size doesn't work with given matrix");
        }
    }

    template <typename data_t>
    MatrixOperator<data_t>::MatrixOperator(const DataDescriptor& domain,
                                           const DataDescriptor& range, const EigenRowMat& mat)
        : MatrixOperator(domain, range, linalg::Matrix(mat))
    {
    }

    template <typename data_t>
    MatrixOperator<data_t>::MatrixOperator(const DataDescriptor& domain,
                                           const DataDescriptor& range, const EigenColMat& mat)
        : MatrixOperator(domain, range, linalg::Matrix(mat))
    {
    }

    template <typename data_t>
    void MatrixOperator<data_t>::applyImpl(const DataContainer<data_t>& x,
                                           DataContainer<data_t>& Ax) const
    {
        linalg::ConstRowView<data_t> vec(x.begin(), x.end());

        auto prod = mat_ * vec;

        Ax = prod;
    }

    template <typename data_t>
    void MatrixOperator<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                                  DataContainer<data_t>& Aty) const
    {
        linalg::ConstRowView<data_t> vec(y.begin(), y.end());

        auto transposed = mat_.transpose();

        auto prod = transposed * vec;

        Aty = prod;
    }

    template <typename data_t>
    MatrixOperator<data_t>* MatrixOperator<data_t>::cloneImpl() const
    {
        return new MatrixOperator(this->getDomainDescriptor(), this->getRangeDescriptor(), mat_);
    }

    template <typename data_t>
    bool MatrixOperator<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherMat = downcast_safe<MatrixOperator>(&other);

        return mat_.rows() == otherMat->mat_.rows() && mat_.rows() == otherMat->mat_.rows()
               && std::equal(mat_.begin(), mat_.end(), otherMat->mat_.begin());
    }

    // ------------------------------------------
    // explicit template instantiation
    template class MatrixOperator<float>;
    template class MatrixOperator<double>;

} // namespace elsa
