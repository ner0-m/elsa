#pragma once

#include "LinearOperator.h"

// TODO add logic, reword, improve
// TODO should this class exist? depends on the development of DiscreteShearletTransform
namespace elsa
{
    /**
     * @brief Operator representing the shearing operation.
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * This class represents a WHAT that is HOW.
     *
     * S_{s} is e.g. the matrix:
     * (1      s)
     * (0      1)
     *
     * References:
     * https://www.math.uh.edu/~dlabate/SHBookIntro.pdf
     */
    template <typename data_t = real_t>
    class ShearingOperator : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for the shearing operator, specifying the domain (= range).
         *
         * @param[in] descriptor DataDescriptor describing the domain and range of the operator
         */
        explicit ShearingOperator(const DataDescriptor& descriptor);

        /// default destructor
        ~ShearingOperator() override = default;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        ShearingOperator(const ShearingOperator<data_t>&) = default;

        /**
         * @brief apply the shearing operator WHAT to WHO, i.e. S_{s}x = WHAT
         *
         * @param[in] x input DataContainer (in the domain of the operator)
         * @param[out] S_{s}x output DataContainer (in the range of the operator)
         */
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ssx) const override;

        /**
         * @brief apply the adjoint of the shearing operator A to y, i.e. S_{s}^ty = y
         *
         * @param[in] y input DataContainer (in the range of the operator)
         * @param[out] S_{s}ty output DataContainer (in the domain of the operator)
         */
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Ssty) const override;

        /// implement the polymorphic clone operation
        ShearingOperator<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;
    };
} // namespace elsa
