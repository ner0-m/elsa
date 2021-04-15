#pragma once

#include "SubsetProblem.h"
#include "Scaling.h"
#include "LinearResidual.h"
#include "WLSProblem.h"

namespace elsa
{
    /**
     * @brief Class representing a WSL Problem that can be split into subsets.
     *
     * @author Michael Loipführer - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents a WSL problem that can be split into smaller, ordered subsets for use
     * in ordered subset solvers like SQS.
     */
    template <typename data_t = real_t>
    class WLSSubsetProblem : public SubsetProblem<data_t>
    {
    public:
        /**
         * @brief Constructor for a subset problem
         *
         * @param[in] A the full system matrix of the whole WSL problem
         * @param[in] b data vector
         * @param[in] subsetAs the system matrices corresponding to each subset
         * @param[in] subsetBs a data vector with a Block Descriptor containing the data vector vor
         * each subset
         */
        WLSSubsetProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                         const std::vector<std::unique_ptr<LinearOperator<data_t>>>& subsetAs);

        /// default destructor
        ~WLSSubsetProblem() override = default;

    protected:
        /// default copy constructor for cloning
        WLSSubsetProblem<data_t>(const WLSSubsetProblem<data_t>&) = default;

        /// implement the polymorphic clone operation
        WLSSubsetProblem<data_t>* cloneImpl() const override;

    private:
        /// converts a list of of operators corresponding to subsets and a data term with a
        /// BlockDescriptor to a list of WLSProblems
        static std::unique_ptr<std::vector<std::unique_ptr<Problem<data_t>>>>
            wlsProblemsFromOperators(
                const std::vector<std::unique_ptr<LinearOperator<data_t>>>& subsetAs,
                const DataContainer<data_t>& b);
    };
} // namespace elsa
