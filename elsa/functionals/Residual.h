#pragma once

#include "Cloneable.h"
#include "DataDescriptor.h"
#include "LinearOperator.h"

#include <memory>

namespace elsa
{
    /**
     * @brief Abstract base class representing a residual, i.e. a vector-valued mapping.
     *
     * @author Matthias Wieczorek - initial code
     * @author Tobias Lasser - modularization, streamlining
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * A residual is a vector-valued mapping representing an error (or mismatch). For real numbers
     * this corresponds to \f$ \mathbb{R}^n\to\mathbb{R}^m \f$ (e.g. \f$ x \mapsto Ax-b \f$ for
     * linear residuals). In order to measure this error, the residual can be used as input to a
     * Functional.
     */
    template <typename data_t = real_t>
    class Residual : public Cloneable<Residual<data_t>>
    {
    public:
        /**
         * @brief Constructor for the residual, mapping from domain to range
         *
         * @param[in] domainDescriptor describing the domain of the residual
         * @param[in] rangeDescriptor describing the range of the residual
         */
        Residual(const DataDescriptor& domainDescriptor, const DataDescriptor& rangeDescriptor);

        /// default destructor
        ~Residual() override = default;

        /// return the domain descriptor
        const DataDescriptor& getDomainDescriptor() const;

        /// return the range descriptor
        const DataDescriptor& getRangeDescriptor() const;

        /**
         * @brief evaluate the residual at x and return the result
         *
         * @param[in] x input DataContainer (in the domain of the residual)
         *
         * @returns result DataContainer (in the range of the residual) containing the result of
         * the evaluation of the residual at x
         *
         * Please note: this method uses evaluate(x, result) to perform the actual operation.
         */
        DataContainer<data_t> evaluate(const DataContainer<data_t>& x) const;

        /**
         * @brief evaluate the residual at x and store in result
         *
         * @param[in] x input DataContainer (in the domain of the residual)
         * @param[out] result output DataContainer (in the range of the residual)
         *
         * Please note: this method calls the method evaluateImpl that has to be overridden in
         * derived classes. (Why is this method here not virtual itself? Because you cannot
         * have a non-virtual function overloading a virtual one [evaluate with one vs. two args].)
         */
        void evaluate(const DataContainer<data_t>& x, DataContainer<data_t>& result) const;

        /**
         * @brief return the Jacobian (first derivative) of the residual at x.
         *
         * @param[in] x input DataContainer (in the domain of the residual) at which the
         * Jacobian of the residual will be evaluated
         *
         * @returns a LinearOperator (the Jacobian)
         *
         * Please note: this method calls the method getJacobianImpl that has to be overridden in
         * derived classes. (This is not strictly necessary, it's just for consistency with
         * evaluate.)
         */
        LinearOperator<data_t> getJacobian(const DataContainer<data_t>& x);

    protected:
        /// the data descriptor of the domain of the residual
        std::unique_ptr<DataDescriptor> _domainDescriptor;

        /// the data descriptor of the range of the residual
        std::unique_ptr<DataDescriptor> _rangeDescriptor;

        /// implement the polymorphic comparison operation
        bool isEqual(const Residual<data_t>& other) const override;

        /// the evaluate method that has to be overridden in derived classes
        virtual void evaluateImpl(const DataContainer<data_t>& x,
                                  DataContainer<data_t>& result) const = 0;

        /// the getJacobian method that has to be overriden in derived classes
        virtual LinearOperator<data_t> getJacobianImpl(const DataContainer<data_t>& x) = 0;
    };
} // namespace elsa
