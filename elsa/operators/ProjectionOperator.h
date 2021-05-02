#pragma once

#include "LinearOperator.h"

// TODO add logic, reword, improve
// TODO should this class exist? depends on the development of SHADMM
namespace elsa
{
    /**
     * @brief Operator representing the projection operation.
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * This class represents a WHAT that is HOW.
     */
    template <typename data_t = real_t>
    class ProjectionOperator : public LinearOperator<data_t> // TODO is it a LinearOperator?
    {
    public:
        /**
         * @brief Constructor for the projection operator, specifying the domain (= range).
         *
         * @param[in] descriptor DataDescriptor describing the domain and range of the operator
         */
        explicit ProjectionOperator(const DataDescriptor& descriptor);

        /// default destructor
        ~ProjectionOperator() override = default;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        ProjectionOperator(const ProjectionOperator<data_t>&) = default;

        /**
         * @brief apply the projection operator WHAT to WHO, i.e. Px = WHAT
         *
         * @param[in] x input DataContainer (in the domain of the operator)
         * @param[out] Px output DataContainer (in the range of the operator)
         */
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Px) const override;

        /**
         * @brief apply the adjoint of the projection operator A to y, i.e. P^ty = y
         *
         * @param[in] y input DataContainer (in the range of the operator)
         * @param[out] Pty output DataContainer (in the domain of the operator)
         */
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Pty) const override;

        /// implement the polymorphic clone operation
        ProjectionOperator<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;
    };
} // namespace elsa
