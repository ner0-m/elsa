#pragma once

#include "LinearOperator.h"

namespace elsa
{
    /**
     * \brief Operator representing the identity operation.
     *
     * \author Matthias Wieczorek - initial code
     * \author Tobias Lasser - modularization, rewrite
     *
     * This class represents a linear operator A that is the identity, i.e. Ax = x.
     */
     template <typename data_t = real_t>
     class Identity : public LinearOperator<data_t> {
     public:
         /**
          * \brief Constructor for the identity operator, specifying the domain (= range).
          *
          * \param[in] descriptor DataDescriptor describing the domain and range of the operator
          */
          Identity(const DataDescriptor& descriptor);

          /// default destructor
          ~Identity() override = default;

     protected:
         /**
          * \brief apply the identity operator A to x, i.e. Ax = x
          *
          * \param[in] x input DataContainer (in the domain of the operator)
          * \param[out] Ax output DataContainer (in the range of the operator)
          */
         void _apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) override;

         /**
          * \brief apply the adjoint of the identity operator A to y, i.e. A^ty = y
          *
          * \param[in] y input DataContainer (in the range of the operator)
          * \param[out] A^ty output DataContainer (in the domain of the operator)
          */
          void _applyAdjoint(const DataContainer<data_t>& y, DataContainer<data_t>& Aty) override;

         /// implement the polymorphic clone operation
         Identity<data_t>* cloneImpl() const override;
     };
} // namespace elsa
