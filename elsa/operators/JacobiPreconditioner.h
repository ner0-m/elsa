#pragma once

#include "LinearOperator.h"
#include "Scaling.h"

namespace elsa
{
    /**
     * @brief Class representing a Jacobi Preconditioner
     *
     * @author Michael Loipführer - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents a Jacobi Precontion Operator for a given LinearOperator.
     */
    template <typename data_t = real_t>
    class JacobiPreconditioner : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for a Jacobi Preconditioner
         *
         * @param[in] op the LinearOperator for which to compute a Jacobi Preconditioner for
         * each subset
         * @param[in] inverse whether or not to invert the computed preconditioner
         */
        JacobiPreconditioner(const LinearOperator<data_t>& op, bool inverse);

        /// default destructor
        ~JacobiPreconditioner() override = default;

    protected:
        /// protected copy constructor; used for cloning
        JacobiPreconditioner(const JacobiPreconditioner& other);

        /// implement the polymorphic clone operation
        JacobiPreconditioner<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

        /// apply the block linear operator
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// apply the adjoint of the block linear operator
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aty) const override;

    private:
        /// the actual inverse diagonal representing a Jacobi Preconditioner for the given problem
        Scaling<data_t> _inverseDiagonal;

        /// lift from base class
        using LinearOperator<data_t>::_domainDescriptor;
        using LinearOperator<data_t>::_rangeDescriptor;

        /// utility to get the diagonal of a linear operator
        static DataContainer<data_t> diagonalFromOperator(const LinearOperator<data_t>& op);
    };

} // namespace elsa
