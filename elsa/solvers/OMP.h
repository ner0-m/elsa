#pragma once

#include "Solver.h"
#include "RepresentationProblem.h"
#include "WLSProblem.h"
#include "CG.h"

namespace elsa
{
    /**
     * @brief Class representing the Orthogonal Matching Pursuit.
     *
     * @author Jonas Buerger - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
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
     */
    template <typename data_t = real_t>
    class OMP : public Solver<data_t>
    {
    public:
        /**
         * @brief Constructor for OMP, accepting a dictionary representation problem and,
         * optionally, a value for epsilon
         *
         * @param[in] problem the representation problem that is supposed to be solved
         * @param[in] epsilon affects the stopping condition
         */
        OMP(const RepresentationProblem<data_t>& problem,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        OMP(const OMP<data_t>&) = delete;

        /// default destructor
        ~OMP() override = default;

    private:
        /// variable affecting the stopping condition
        data_t _epsilon;

        /// lift the base class method getCurrentSolution
        using Solver<data_t>::getCurrentSolution;

        /// lift the base class variable _problem
        using Solver<data_t>::_problem;

        /// helper method to find the index of the atom that is most correlated with the residual
        index_t mostCorrelatedAtom(const Dictionary<data_t>& dict,
                                   const DataContainer<data_t>& evaluatedResidual);

        /**
         * @brief Solve the representation problem, i.e. apply iterations number of iterations of
         * matching pursuit
         *
         * @param[in] iterations number of iterations to execute. As OMP is a greedy algorithm, this
         * corresponds to the desired sparsity level
         *
         * @returns a reference to the current solution
         */
        DataContainer<data_t>& solveImpl(index_t iterations) override;

        /// implement the polymorphic clone operation
        OMP<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa