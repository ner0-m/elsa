#include "PowerIterations.h"
#include "DataContainer.h"
#include "Error.h"
#include "LinearOperator.h"

#include <iostream>

namespace elsa
{
    template <class data_t>
    data_t powerIterations(const LinearOperator<data_t>& op, index_t niters)
    {
        if (op.getDomainDescriptor().getNumberOfCoefficients()
            != op.getRangeDescriptor().getNumberOfCoefficients()) {
            throw LogicError("powerIterations: Operator for power iterations must be symmetric");
        }

        DataContainer<data_t> u(op.getDomainDescriptor());
        u = 1;

        for (index_t i = 0; i < niters; i++) {
            op.apply(u, u);
            u = u / u.l2Norm();
        }

        return u.dot(op.apply(u)) / u.l2Norm();
    }

    template float powerIterations(const LinearOperator<float>&, index_t);
    template double powerIterations(const LinearOperator<double>&, index_t);
    template complex<float> powerIterations(const LinearOperator<complex<float>>&, index_t);
    template complex<double> powerIterations(const LinearOperator<complex<double>>&, index_t);
} // namespace elsa
