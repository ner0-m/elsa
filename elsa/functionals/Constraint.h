#pragma once

#include "Functional.h"

namespace elsa
{
    /**
     * @brief Class representing a constraint associated to an optimization problem.
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * Constraint is expressed in the form
     *  - @f$ Ax + Bz = c @f$.
     *
     * All three components must be present during construction. One can interchangeably utilize a
     * LinearResidual if either of the LinearOperators A or B is the zero matrix.
     */
    template <typename data_t = real_t>
    class Constraint : public Cloneable<Constraint<data_t>>
    {
    public:
        /**
         * @brief Constructor for the constraint, accepting the two LinearOperators and
         * DataContainer
         *
         * @param[in] A LinearOperator
         * @param[in] B LinearOperator
         * @param[in] c DataContainer
         */
        Constraint(const LinearOperator<data_t>& A, const LinearOperator<data_t>& B,
                   const DataContainer<data_t>& c);

        /// make copy constructor deletion explicit
        Constraint(const Constraint<data_t>&) = delete;

        /// default destructor
        ~Constraint() = default;

        /// return the operator A
        auto getOperatorA() const -> const LinearOperator<data_t>&;

        /// return the operator B
        auto getOperatorB() const -> const LinearOperator<data_t>&;

        /// return the data vector c
        auto getDataVectorC() const -> const DataContainer<data_t>&;

    protected:
        /// implement the clone operation
        auto cloneImpl() const -> Constraint<data_t>* override;

        /// overridden comparison method based on the LinearOperators and the DataContainer
        auto isEqual(const Constraint<data_t>& other) const -> bool override;

    private:
        /// @f$ A @f$ from the problem definition
        std::unique_ptr<LinearOperator<data_t>> _A;

        /// @f$ B @f$ from the problem definition
        std::unique_ptr<LinearOperator<data_t>> _B;

        /// @f$ c @f$ from the problem definition
        DataContainer<data_t> _c;
    };
} // namespace elsa
