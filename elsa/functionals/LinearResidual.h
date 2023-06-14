#pragma once

#include <optional>

#include "DataContainer.h"
#include "DataDescriptor.h"
#include "LinearOperator.h"

namespace elsa
{
    /**
     * @brief Class representing a linear residual, i.e. Ax - b with operator A and vectors x, b.
     *
     * A linear residual is a vector-valued mapping \f$ \mathbb{R}^n\to\mathbb{R}^m \f$, namely
     * \f$ x \mapsto  Ax - b \f$, where A is a LinearOperator, b a constant data vector
     * (DataContainer) and x a variable (DataContainer). This linear residual can be used as input
     * to a Functional.
     *
     * @tparam data_t data type for the domain and range of the operator, default to real_t
     *
     * @author
     * * Matthias Wieczorek - initial code
     * * Tobias Lasser - modularization, modernization
     */
    template <typename data_t = real_t>
    class LinearResidual
    {
    public:
        /**
         * @brief Constructor for a trivial residual \f$ x \mapsto x \f$
         *
         * @param[in] descriptor describing the domain = range of the residual
         */
        explicit LinearResidual(const DataDescriptor& descriptor);

        /**
         * @brief Constructor for a simple residual \f$ x \mapsto x - b \f$
         *
         * @param[in] b a vector (DataContainer) that will be subtracted from x
         */
        explicit LinearResidual(const DataContainer<data_t>& b);

        /** @brief Constructor for a residual \f$ x \mapsto Ax \f$
         *
         * @param[in] A a LinearOperator
         */
        explicit LinearResidual(const LinearOperator<data_t>& A);

        /**
         * @brief Constructor for a residual \f$ x \mapsto Ax - b \f$
         *
         * @param[in] A a LinearOperator
         * @param[in] b a vector (DataContainer)
         */
        LinearResidual(const LinearOperator<data_t>& A, const DataContainer<data_t>& b);

        // Copy constructor
        LinearResidual(const LinearResidual<data_t>&);

        // Copy assignment
        LinearResidual& operator=(const LinearResidual<data_t>&);

        // Move constructor
        LinearResidual(LinearResidual<data_t>&&) noexcept;

        // Move assignment
        LinearResidual& operator=(LinearResidual<data_t>&&) noexcept;

        /// default destructor
        ~LinearResidual() = default;

        /// return the domain descriptor of the residual
        const DataDescriptor& getDomainDescriptor() const;

        /// return the range descriptor of the residual
        const DataDescriptor& getRangeDescriptor() const;

        /// return true if the residual has an operator A
        bool hasOperator() const;

        /// return true if the residual has a data vector b
        bool hasDataVector() const;

        /// return the operator A (throws if the residual has none)
        const LinearOperator<data_t>& getOperator() const;

        /// return the data vector b (throws if the residual has none)
        const DataContainer<data_t>& getDataVector() const;

        /**
         * @brief evaluate the residual at x and return the result
         *
         * @param[in] x input DataContainer (in the domain of the residual)
         *
         * @returns result DataContainer (in the range of the residual) containing the result of
         * the evaluation of the residual at x
         */
        DataContainer<data_t> evaluate(const DataContainer<data_t>& x) const;

        /**
         * @brief evaluate the residual at x and store in result
         *
         * @param[in] x input DataContainer (in the domain of the residual)
         * @param[out] result output DataContainer (in the range of the residual)
         */
        void evaluate(const DataContainer<data_t>& x, DataContainer<data_t>& result) const;

        /**
         * @brief return the Jacobian (first derivative) of the linear residual at x.
         * If A is set, then the Jacobian is A and this returns a copy of A.
         * If A is not set, then an Identity operator is returned.
         *
         * @param x input DataContainer (in the domain of the residual)
         *
         * @returns  a LinearOperator (the Jacobian)
         */
        LinearOperator<data_t> getJacobian(const DataContainer<data_t>& x);

    private:
        /// Descriptor of domain
        std::unique_ptr<DataDescriptor> domainDesc_;

        /// Descriptor of range
        std::unique_ptr<DataDescriptor> rangeDesc_;

        /// the operator A, nullptr implies no operator present
        std::unique_ptr<LinearOperator<data_t>> _operator{};

        /// optional  data vector b
        std::optional<DataContainer<data_t>> _dataVector{};
    };

    template <class data_t>
    bool operator==(const LinearResidual<data_t>& lhs, const LinearResidual<data_t>& rhs)
    {
        if (lhs.getDomainDescriptor() != rhs.getDomainDescriptor()) {
            return false;
        }

        if (lhs.getRangeDescriptor() != rhs.getRangeDescriptor()) {
            return false;
        }

        if (lhs.hasOperator() && rhs.hasOperator() && lhs.getOperator() != rhs.getOperator()) {
            return false;
        }

        if (lhs.hasDataVector() && rhs.hasDataVector()
            && lhs.getDataVector() != rhs.getDataVector()) {
            return false;
        }

        return true;
    }

    template <class data_t>
    bool operator!=(const LinearResidual<data_t>& lhs, const LinearResidual<data_t>& rhs)
    {
        return !(lhs == rhs);
    }
} // namespace elsa
