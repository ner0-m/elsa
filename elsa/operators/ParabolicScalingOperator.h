#pragma once

#include "LinearOperator.h"
#include "VolumeDescriptor.h"
#include "Timer.h"

// TODO add logic, reword, improve
// TODO should this class exist? depends on the development of DiscreteShearletTransform
namespace elsa
{
    /**
     * @brief Operator representing the parabolic scaling operation.
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t//TODO
     *
     * This class represents a 2x2 parabolic scaling operation A_{a}. A_{a} is the symmetric matrix
     * defined as:
     *
     * (a        0)
     * (0  a^{1/2})
     *
     * where the scaling parameter a ∈ R_{+}.
     *
     * Its inverse matrix A_{a}^{-1} is defined as:
     *
     * (1/a        0)
     * (0  1/a^{1/2})
     *
     * References:
     * https://www.math.uh.edu/~dlabate/SHBookIntro.pdf
     */
    template <typename data_t = real_t>
    class ParabolicScalingOperator : public LinearOperator<data_t>
    {
    public:
        /**
         * @brief Constructor for the parabolic scaling operator, specifying the domain (= range).
         *
         * @param[in] descriptor DataDescriptor describing the domain and range of the operator
         */
        explicit ParabolicScalingOperator(real_t scalingParameter);

        /// default destructor
        ~ParabolicScalingOperator() override = default;

        /**
         * @brief apply the operator A to an element in the operator's domain
         *
         * @param[in] x input DataContainer (in the domain of the operator)
         * @param[out] Ax output DataContainer (in the range of the operator)
         *
         * Please note: this method calls the method applyImpl that has to be overridden in derived
         * classes. (Why is this method not virtual itself? Because you cannot have a non-virtual
         * function overloading a virtual one [apply with one vs. two arguments]).
         */
        void applyInverse(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const;

        /**
         * @brief apply the adjoint of operator A to an element of the operator's range
         *
         * @param[in] y input DataContainer (in the range of the operator)
         * @param[out] Aty output DataContainer (in the domain of the operator)
         *
         * Please note: this method calls the method applyAdjointImpl that has to be overridden in
         * derived classes. (Why is this method not virtual itself? Because you cannot have a
         * non-virtual function overloading a virtual one [applyAdjoint with one vs. two args]).
         */
        void applyInverseAdjoint(const DataContainer<data_t>& y, DataContainer<data_t>& Aty) const;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        ParabolicScalingOperator(const ParabolicScalingOperator<data_t>&) = default;

        /**
         * @brief apply the parabolic scaling operator WHAT to WHO, i.e. A_{a}x = WHAT
         *
         * @param[in] x input DataContainer (in the domain of the operator)
         * @param[out] A_{a}x output DataContainer (in the range of the operator)
         */
        void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Aax) const override;

        /**
         * @brief apply the adjoint of the parabolic scaling operator A_{a} to y, i.e. A_{a}^ty = y
         *
         * @param[in] y input DataContainer (in the range of the operator)
         * @param[out] Aaty output DataContainer (in the domain of the operator)
         */
        void applyAdjointImpl(const DataContainer<data_t>& y,
                              DataContainer<data_t>& Aaty) const override;

        /// implement the polymorphic clone operation
        ParabolicScalingOperator<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        real_t _scalingParameter;
    };
} // namespace elsa
