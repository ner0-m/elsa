#pragma once

#include "Problem.h"
#include "Scaling.h"
#include "LinearResidual.h"

namespace elsa
{
    /**
     * @brief Class representing a generic Problem that can be split into ordered subsets.
     *
     * @author Michael Loipführer - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents a problem that can be split into smaller, ordered subsets for use in
     * ordered subset solvers like SQS.
     */
    template <typename data_t = real_t>
    class SubsetProblem : public Problem<data_t>
    {
    public:
        /**
         * @brief Constructor for a subset problem
         *
         * @param[in] fullProblem the generic problem
         * @param[in] subsetProblems the problem instances corresponding to each ordered subset
         *
         * NOTE: we also need to store the full problem as otherwise we would not be able to
         * easily compute the hessian of the problem.
         */
        SubsetProblem(const Problem<data_t>& fullProblem,
                      const std::vector<std::unique_ptr<Problem<data_t>>>& subsetProblems);

        /// default destructor
        ~SubsetProblem() override = default;

        /**
         * @brief return the gradient of the problem at the current estimated solution for the given
         * subset
         *
         * @param[in] subset is index of the subset the gradient is evaluated for
         *
         * @returns DataContainer (in the domain of the problem) containing the result of the
         * gradient at the current solution for the given subset
         *
         */
        DataContainer<data_t> getSubsetGradient(const DataContainer<data_t>& x, index_t subset);

        /**
         * @brief compute the gradient of the problem at the current estimated solution
         *
         * @param[in] subset is index of the subset the gradient is evaluated for
         * @param[out] result output DataContainer containing the gradient (in the domain of the
         * problem) evaluated for the given subset
         *
         */
        void getSubsetGradient(const DataContainer<data_t>& x, DataContainer<data_t>& result,
                               index_t subset);

        /**
         * @brief return the number of ordered subsets this problem is divided into
         */
        index_t getNumberOfSubsets() const;

    protected:
        /// copy constructor for use in cloning
        SubsetProblem<data_t>(const SubsetProblem<data_t>& subsetProblem);

        /// implement the polymorphic clone operation
        SubsetProblem<data_t>* cloneImpl() const override;

        /// the subset-problems
        std::vector<std::unique_ptr<Problem<data_t>>> _subsetProblems;
    };
} // namespace elsa
