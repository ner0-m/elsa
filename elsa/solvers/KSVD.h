#pragma once

#include "Solver.h"
#include "RepresentationProblem.h"
#include "DictionaryLearningProblem.h"
#include "OMP.h"

namespace elsa
{
    /**
     * @brief Class representing the K-Singular Value Decomposition
     *
     * @author Jonas Buerger - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * TODO add meaningful description
     */
    template <typename data_t = real_t>
    class KSVD
    {
    public:
        /**
         * @brief Constructor for KSVD, accepting a dictionary representation problem and,
         * optionally, a value for epsilon
         *
         * @param[in] problem the representation problem that is supposed to be solved
         * @param[in] epsilon affects the stopping condition
         */
        KSVD(const DictionaryLearningProblem<data_t>& problem,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        KSVD(const KSVD<data_t>&) = delete;

        /// default destructor
        //~KSVD() override = default;

    private:
        /// variable affecting the stopping condition
        data_t _epsilon;

        /// lift the base class variable _problem
        DictionaryLearningProblem<data_t> problem;

        /**
         * @brief Solve the representation problem, i.e. apply iterations number of iterations of
         * matching pursuit
         *
         * @param[in] iterations number of iterations to execute. As OMP is a greedy algorithm, this
         * corresponds to the desired sparsity level
         *
         * @returns a reference to the current solution
         */
        DataContainer<data_t>& solveImpl(index_t iterations);

        void calculateSVD(DataContainer<data_t> data);
        /// implement the polymorphic clone operation
        // OMP<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        // bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
