#pragma once

#include "DataDescriptor.h"
#include "elsaDefines.h"
#include "DataContainer.h"
#include "Cloneable.h"
#include <optional>

namespace elsa
{
    /**
     * @brief Base class representing a solver for an optimization problem.
     *
     * @author Matthias Wieczorek - initial code
     * @author Maximilian Hornung - modularization
     * @author Tobias Lasser - rewrite, modernization
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents abstract (typically iterative) solvers acting on optimization problems.
     */
    template <typename data_t = real_t>
    class Solver : public Cloneable<Solver<data_t>>
    {
    public:
        /// Scalar alias
        using Scalar = data_t;

        Solver() = default;

        /// default destructor
        ~Solver() override = default;

        /**
         * @brief Solve the optimization problem (most likely iteratively)
         *
         * @param[in] iterations number of iterations to execute
         * @param[in] x0 optional initial solution, initial solution set to zero if not present
         *
         * @returns the current solution (after solving)
         */
        virtual DataContainer<data_t>
            solve(index_t iterations, std::optional<DataContainer<data_t>> x0 = std::nullopt) = 0;
    };

    /// Extract the default value from the optional if present, else create a
    /// new DataContainer with the given Descriptor and initial value
    template <class data_t>
    DataContainer<data_t> extract_or(std::optional<DataContainer<data_t>> x0,
                                     const DataDescriptor& domain,
                                     SelfType_t<data_t> val = data_t{0})
    {
        return x0.has_value() ? x0.value() : full<data_t>(domain, val);
    }
} // namespace elsa
