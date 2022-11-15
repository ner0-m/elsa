#pragma once

#include "LinearOperator.h"
#include "Matrix.h"

#include <Eigen/Core>

namespace elsa
{

    /**
     * @brief Operator representing a dense matrix. Other operators are usually stored in some
     * sparse form. This operator is useful for small example operations where the matrix is known
     * or can be computed easily, or some other way.
     */
    template <class data_t>
    class MatrixOperator : public LinearOperator<data_t>
    {
        using EigenRowMat = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using EigenColMat = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    public:
        /// Construct from domain, range and `elsa::linalg::Matrix`
        MatrixOperator(const DataDescriptor& domain, const DataDescriptor& range,
                       const linalg::Matrix<data_t>& mat);

        /// Construct from domain, range and an Eigen row major stored matrix
        MatrixOperator(const DataDescriptor& domain, const DataDescriptor& range,
                       const EigenRowMat& mat);

        /// Construct from domain, range and an Eigen column major stored matrix
        MatrixOperator(const DataDescriptor& domain, const DataDescriptor& range,
                       const EigenColMat& mat);

        /// default destructor
        ~MatrixOperator() override = default;

    protected:
        /// apply the matrix operation
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint (i.e transpose) of the matrix operation
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

        /// implement the polymorphic clone operation
        MatrixOperator<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        linalg::Matrix<data_t> mat_;
    };
} // namespace elsa
