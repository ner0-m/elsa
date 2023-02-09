#pragma once

#include "DataContainer.h"
#include "LinearOperator.h"

namespace elsa
{
    /// @brief The power iterations approximate the largest eigenvalue of the given operator.
    /// The operator must be symmetric, else an exception is thrown.
    template <class data_t>
    data_t powerIterations(const LinearOperator<data_t>& op, index_t niters = 5);
} // namespace elsa
