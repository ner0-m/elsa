#pragma once

#include <optional>

#include "Solver.h"
#include "Dictionary.h"

namespace elsa
{
    /**
     * @brief Class representing the Orthogonal Matching Pursuit.
     *
     * Orthogonal Matching Pursuit is a greedy algorithm to find a sparse representation. It starts
     * with the 0-vector and adds one non-zero entry per iteration. The algorithm works in the
     * following manner:
     * -# Find the next atom that should be used in the representation. This done by finding the
     * atom that is correlated the most with the current residual.
     * -# Construct a dictionary that only contains the atoms that are being used, defined as \f$
     * D_S \f$ (dictionary restricted to the support).
     * -# The representation is the solution to the least square problem \f$ min_x \|y-D_S*x\| \f$
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * @author Jonas Buerger - initial code
     *
     */
    template <typename data_t = real_t>
    class OrthogonalMatchingPursuit : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        /**
         * @brief Constructor for OrthogonalMatchingPursuit, accepting a dictionary representation
         * problem and, optionally, a value for epsilon
         *
         * @param[in] D dictionary operator
         * @param[in] y signal that should be sparsely represented
         * @param[in] epsilon affects the stopping condition
         */
        OrthogonalMatchingPursuit(const Dictionary<data_t>& D, const DataContainer<data_t>& y,
                                  data_t epsilon);

        /// make copy constructor deletion explicit
        OrthogonalMatchingPursuit(const OrthogonalMatchingPursuit<data_t>&) = delete;

        /// default destructor
        ~OrthogonalMatchingPursuit() override = default;

        /**
         * @brief Solve the representation problem, i.e. apply iterations number of iterations of
         * matching pursuit
         *
         * @param[in] iterations number of iterations to execute. As OrthogonalMatchingPursuit is a
         * greedy algorithm, this corresponds to the desired sparsity level
         * @param[in] x0 optional initial solution, initial solution set to zero if not present
         *
         * @returns the approximated solution
         */
        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

    private:
        /// helper method to find the index of the atom that is most correlated with the residual
        index_t mostCorrelatedAtom(const Dictionary<data_t>& dict,
                                   const DataContainer<data_t>& evaluatedResidual);

        /// implement the polymorphic clone operation
        OrthogonalMatchingPursuit<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;

        Dictionary<data_t> dict_;

        DataContainer<data_t> signal_;

        /// variable affecting the stopping condition
        data_t epsilon_;
    };
} // namespace elsa
