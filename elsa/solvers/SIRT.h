#pragma once
#include "WLSProblem.h"
#include "Solver.h"
#include "LinearOperator.h"

namespace elsa
{
    /**
     * \brief Class representing a general SIRT solver.
     *
     * \author Maryna Shcherbak - initial code
     *
     * \tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class implements a SIRT solver with the update expression \f$ x^{(k+1)} = x^k +
     * CA^T(b-Ax^k)) \f$. Here \f$ C \f$ is a diagonal matrix of inverse column sum of the system
     * matrix.
     *
     * References:
     * https://doi.org/10.1109/tmi.2008.923696
     */
    template <typename data_t = real_t>
    class SIRT : public Solver<data_t>
    {
    public:
        /**
         * \brief Constructor for SIRT, accepting a problem.
         * \param[in] problem the problem that is supposed to be solved
         */
        SIRT(const WLSProblem<data_t>& problem);

        /// make copy constructor deletion explicit
        SIRT(const SIRT<data_t>&) = delete;

        /// default destructor
        ~SIRT() override = default;

        /// lift the base class method getCurrentSolution
        using Solver<data_t>::getCurrentSolution;

    private:
        /// the default number of iterations
        const index_t _defaultIterations{100};

        /// lift the base class variable _problem
        using Solver<data_t>::_problem;
        /**
         * \brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * SIRT
         *
         * \param[in] iterations number of iterations to execute (the default 0 value executes
         * _defaultIterations of iterations)
         *
         * \returns a reference to the current solution
         */
        DataContainer<data_t>& solveImpl(index_t iterations) override;

        /// implement the polymorphic clone operation
        SIRT<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };

} // namespace elsa
