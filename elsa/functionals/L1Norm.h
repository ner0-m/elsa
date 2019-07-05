#pragma once

#include "Functional.h"

namespace elsa
{
    /**
     * \brief Class representing the l1 norm functional.
     *
     * \author Matthias Wieczorek - initial code
     * \author Maximilian Hornung - modularization
     * \author Tobias Lasser - modernization
     *
     * \tparam data_t data type for the domain of the residual of the functional, defaulting to real_t
     *
     * The l1 norm functional evaluates to \f$ \sum_{i=1}^n |x_i| \f$ for \f$ x=(x_i)_{i=1}^n \f$.
     * Please note that it is not differentiable, hence getGradient and getHessian will throw exceptions.
     */
     template <typename data_t = real_t>
     class L1Norm : public Functional<data_t> {
     public:
         /**
          * \brief Constructor for the l1 norm functional, mapping domain vector to a scalar (without a residual)
          *
          * \param[in] domainDescriptor describing the domain of the functional
          */
         explicit L1Norm(const DataDescriptor& domainDescriptor);

         /**
          * \brief Constructor for the l1 norm functional, using a residual as input to map to a scalar
          *
          * \param[in] residual to be used when evaluating the functional (or its derivatives)
          */
         explicit L1Norm(const Residual<data_t>& residual);

         /// default destructor
         ~L1Norm() override = default;

     protected:
         /// the evaluation of the l1 norm
         data_t _evaluate(const DataContainer<data_t>& Rx) override;

         /// the computation of the gradient (in place)
         void _getGradientInPlace(DataContainer<data_t>& Rx) override;

         /// the computation of the Hessian
         LinearOperator<data_t> _getHessian(const DataContainer<data_t>& Rx) override;

         /// implement the polymorphic clone operation
         L1Norm<data_t>* cloneImpl() const override;

         /// implement the polymorphic comparison operation
         bool isEqual(const Functional<data_t>& other) const override;
     };

} // namespace elsa
