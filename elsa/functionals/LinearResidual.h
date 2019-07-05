#pragma once

#include "Residual.h"

namespace elsa
{
    /**
     * \brief Class representing a linear residual, i.e. Ax - b with operator A and vectors x, b.
     *
     * \author Matthias Wieczorek - initial code
     * \author Tobias Lasser - modularization, modernization
     *
     * \tparam data_t data type for the domain and range of the operator, default to real_t
     *
     * A linear residual is a vector-valued mapping \f$ \mathbb{R}^n\to\mathbb{R}^m \f$, namely
     * \f$ x \mapsto  Ax - b \f$, where A is a LinearOperator, b a constant data vector (DataContainer)
     * and x a variable (DataContainer).
     * This linear residual can be used as input to a Functional.
     */
    template <typename data_t = real_t>
    class LinearResidual : public Residual<data_t> {
    public:
        /**
         * \brief Constructor for a trivial residual \f$ x \mapsto x \f$
         *
         * \param[in] descriptor describing the domain = range of the residual
         */
        explicit LinearResidual(const DataDescriptor& descriptor);

        /**
         * \brief Constructor for a simple residual \f$ x \mapsto x - b \f$
         *
         * \param[in] b a vector (DataContainer) that will be subtracted from x
         */
        explicit LinearResidual(const DataContainer<data_t>& b);

        /** \brief Constructor for a residual \f$ x \mapsto Ax \f$
         *
         * \param[in] A a LinearOperator
         */
        explicit LinearResidual(const LinearOperator<data_t>& A);

        /**
         * \brief Constructor for a residual \f$ x \mapsto Ax - b \f$
         *
         * \param[in] A a LinearOperator
         * \param[in] b a vector (DataContainer)
         */
        LinearResidual(const LinearOperator<data_t>& A, const DataContainer<data_t>& b);

        /// default destructor
        ~LinearResidual() override = default;


        /// return true if the residual has an operator A
        bool hasOperator() const;

        /// return true if the residual has a data vector b
        bool hasDataVector() const;

        /// return the operator A (throws if the residual has none)
        LinearOperator<data_t>& getOperator() const;

        /// return the data vector b (throws if the residual has none)
        const DataContainer<data_t>& getDataVector() const;

    protected:
        /// implement the polymorphic clone operation
        LinearResidual<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Residual<data_t>& other) const override;

        /// the evaluate method, evaluating the residual at x and placing the value in result
        void _evaluate(const DataContainer<data_t>& x, DataContainer<data_t>& result) override;

        /**
         * \brief return the Jacobian (first derivative) of the linear residual at x.
         *
         * \param x input DataContainer (in the domain of the residual)
         *
         * \returns  a LinearOperator (the Jacobian)
         *
         * If A is set, then the Jacobian is A and this returns a copy of A.
         * If A is not set, then an Identity operator is returned.
         */
        LinearOperator<data_t> _getJacobian(const DataContainer<data_t>& x) override;


    private:
        /// flag if operator A is present
        bool _hasOperator;

        /// flag if data vector b is present
        bool _hasDataVector;

        /// the operator A
        std::unique_ptr<LinearOperator<data_t>> _operator{};

        /// the data vector b
        std::unique_ptr<DataContainer<data_t>> _dataVector{};
    };

} // namespace elsa
