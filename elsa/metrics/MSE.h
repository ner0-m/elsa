#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"

namespace elsa
{
    /**
     * @brief Base class representing a proximity operator prox.
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the values of the operator, defaulting to real_t
     *
     * This class represents a proximity operator prox, expressed through its apply methods,
     * which implement the proximity operator of f with penalty r i.e.
     * @f$ prox_{f,\rho}(v) = argmin_{x}(f(x) + (\rho/2)·\| x - v \|^2_2). @f$
     *
     * Concrete implementations of proximity operators will derive from this class and override the
     * applyImpl method.
     *
     * References:
     * https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
     */
    template <typename data_t = real_t>
    class MSE
    {
    public:
        static data_t calculate(DataContainer<data_t> x, DataContainer<data_t> y);
    };
} // namespace elsa