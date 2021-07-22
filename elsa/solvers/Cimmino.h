#pragma once
#include "WLSProblem.h"
#include "Solver.h"
#include "LinearOperator.h"

namespace elsa
{
    /**
     * @brief Class representing a  Cimmino solver.
     *
     * @author Maryna Shcherbak - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class implements a Cimmino solver with the update expression \f$ x^{(k+1)} = x^k +
     * \lambda_kATD(b-Ax^k)) \f$. Here \f$ D = \frac{1}{m}diag(\frac{1}{\norm{a^1}_2^2},
     * \frac{1}{\norm{a^2}_2^2}, ... , \frac{1}{\norm{a^m}_2^2} ) \f$ is a diagonal matrix of
     * inverse column sum of the system matrix.
     *
     * References:
     * https://doi.org/10.1016/S0167-8191(00)00100-9
     */
    template <typename data_t = real_t>
    class Cimmino : public Solver<data_t>
    {
    public:
        /**
         * @brief Constructor for Cimmino, accepting a problem and, optionally, a relaxation
         * parameter.
         * @param[in] problem the problem that is supposed to be solved
         * @param[in]
         * relaxation parameter, default is 1
         */
        Cimmino(const WLSProblem<data_t>& problem, data_t relaxationParam = 1.0);

        /// make copy constructor deletion explicit
        Cimmino(const Cimmino<data_t>&) = delete;

        /// default destructor
        ~Cimmino() override = default;
        /// lift the base class method getCurrentSolution
        using Solver<data_t>::getCurrentSolution;

    private:
        /// the step size
        data_t _relaxationParam;

        /// the default number of iterations
        const index_t _defaultIterations{100};

        /// lift the base class variable _problem
        using Solver<data_t>::_problem;
        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations Cimmino
         *
         * @param[in] iterations number of iterations to execute (the default 0 value executes
         * _defaultIterations of iterations)
         *
         * @returns a reference to the current solution
         */
        DataContainer<data_t>& solveImpl(index_t iterations) override;

        /// implement the polymorphic clone operation
        Cimmino<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa