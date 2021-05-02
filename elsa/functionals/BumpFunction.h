#pragma once

#include "Functional.h"

// TODO might be used as a generating function in shearlet systems
namespace elsa
{
    /**
     * // TODO reword
     * @brief Class representing the bump function (there are many such functions)
     *
     * @author Andi Braimllari - initial code
     *
     * // TODO reword
     * @tparam data_t data type for the domain of the residual of the functional, defaulting to
     * real_t
     *
     * The bump function evaluates to ? this is a family of functions
     *
     * References: // TODO change
     * https://en.wikipedia.org/wiki/Bump_function
     */
    // TODO extends functional?
    template <typename data_t = real_t>
    class BumpFunction : public Functional<data_t>
    {
    public:
        /**
         * @brief Constructor for the bump function, mapping domain vector to a scalar
         *
         * @param[in] domainDescriptor describing the domain of the functional
         */
        explicit BumpFunction(const DataDescriptor& domainDescriptor);

        /// make copy constructor deletion explicit
        BumpFunction(const BumpFunction<data_t>&) = delete;

        /// default destructor
        ~BumpFunction() override = default;

    protected:
        /// the evaluation of the bump function
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// the computation of the gradient (in place)
        void getGradientInPlaceImpl(DataContainer<data_t>& Rx) override;

        /// the computation of the Hessian
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// implement the polymorphic clone operation
        BumpFunction<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Functional<data_t>& other) const override;
    };
} // namespace elsa
