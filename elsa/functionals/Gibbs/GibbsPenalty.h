#pragma once

#include "Functional.h"
#include "Scaling.h"
#include "GibbsUtils.h"

namespace elsa
{
    namespace Gibbs
    {
        template <typename data_t = real_t>
        class GibbsPenalty : public Functional<data_t>
        {
        public:
            explicit GibbsPenalty(const DataDescriptor& domainDescriptor);

            // explicit GibbsPenalty(const Residual<data_t>& residual);
            //  GibbsPotential(const LinearOperator<data_t>& A, const DataContainer<data_t>& b); do
            //  i need those?

            /// make copy constructor deletion explicit
            GibbsPenalty(const GibbsPenalty<data_t>&) = delete;

            /// default destructor
            ~GibbsPenalty() override = default;

        protected:
            /// the evaluation of the Gibbs penalty
            data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

            /// the computation of the gradient (in place)
            void getGradientInPlaceImpl(DataContainer<data_t>& Rx) override;

            /// the computation of the Hessian
            LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

            /// implement the polymorphic clone operation
            GibbsPenalty<data_t>* cloneImpl() const override;

            /// implement the polymorphic comparison operation
            bool isEqual(const Functional<data_t>& other) const override;

        private:
            /// helper function for one dim index vector
            IndexVector_t oneDimension(index_t val)
            {
                IndexVector_t res(1);
                res << val;
                return res;
            };
        };
    }; // namespace Gibbs

} // namespace elsa
