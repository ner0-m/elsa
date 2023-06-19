#pragma once

#include <memory>

#include "DataContainer.h"
#include "LinearOperator.h"
#include "Solver.h"
#include "elsaDefines.h"
#include "Functional.h"

namespace elsa
{
    /**
     * @brief Conjugate Gradient solver for Non-Linear Problems
     *
     * @author Shen Hu - initial code
     *
     * @tparam data_t data type for the domain of the problem to be solved, defaulting to
     * real_t
     *
     * @author Shen Hu - initial code
     *
     * Newton-Raphson method is integrated for line-search and beta (step-size for updating
     * direction d) is calculated in the Fletcher-Reeves way
     *
     * Reference: https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
     * section 14.1
     */
    template <typename data_t = real_t>
    class CGNL : public Solver<data_t>
    {
    public:

        /**
         * @brief constructor for the CGNL solver
         *
         * @param[in] prob the actual problem to be solved
         * @param[in] eps_CG stopping criteria for early stopping of CG iterations
         * @param[in] eps_NR stopping criteria for early stopping of line-search iterations (Newton-Raphson)
         * @param[in] iterations_NR max iterations for each line search (Newton-Raphson)
         * @param[in] restart hard reset iteration count for reinitializing line search direction
         */
        CGNL(const Functional<data_t>& func, data_t eps_CG = 1e-4,
             data_t eps_NR = 1e-4, index_t iterations_NR = 2, index_t restart = 5);

        /// make copy constructor deletion explicit
        CGNL(const CGNL<data_t>&) = delete;

        /// default destructor
        ~CGNL() override = default;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of CGLS
         *
         * @param[in] iterations number of iterations to execute
         * @param[in] x0 optional initial solution, initial solution set to zero if not present
         *
         * @returns the approximated solution
         */
        DataContainer<data_t>
            solve(index_t iterations_CG,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

    private:
        /// implement the polymorphic clone operation
        CGNL<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;

        /// smart ptr to the actual problem to be solved
        std::unique_ptr<Functional<data_t>> func_;

        /// stopping criteria for early stopping of CG iterations
        data_t eps_CG_;

        /// stopping criteria for early stopping of line-search iterations (Newton-Raphson)
        data_t eps_NR_;

        /// max iterations for each line search (Newton-Raphson)
        index_t iterations_NR_;

        /// hard reset iteration count for reinitializing line search direction
        index_t restart_;
    };
} // namespace elsa
