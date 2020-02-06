#pragma once

#include "WLSProblem.h"

namespace elsa
{
    /**
     * \brief Class representing a Tikhonov regularized weighted least squares problem
     *
     * \author Nikola Dinev
     *
     * \tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents a Tikhonov regularized weighted least squares problem.
     * Some common examples are:
     * - \f$ \argmin_x \frac{1}{2} \| Ax - b \|_2^2 + \lambda \| x \|_2^2 \f$
     * - \f$ \argmin_x \frac{1}{2} \| Ax - b \|_2^2 + \lambda \| x - x^* \|_2^2 \f$
     * - \f$ \argmin_x \frac{1}{2} \| Ax - b \|_2^2 + \lambda \| Lx \|_2^2 \f$
     * - \f$ \argmin_x \frac{1}{2} \| Ax - b \|_2^2 + \lambda \| L(x - x^*) \|_2^2 \f$,
     * where \f$ A \f$ is a linear operator and \f$ b \f$ and \f$ x^* \f$ are data vectors,
     * \f$ \lambda \f$ is the regularization weight, and \f$ L \f$ is a discretized differential
     * operator.
     *
     * This class supports a wider range of problems - any problem of the form
     * \f$ \argmin_x \frac{1}{2} \| Ax - b \|_{W,2}^2 + \sum_{i=1}^n \lambda_i \| B_ix - x^*_i
     * \|_{V_i,2}^2 \f$ is considered a Tikhonov problem. Here \f$ A \f$ and \f$ B_i \f$ are linear
     * operators, \f$ b \f$ and \f$ x^*_i \f$ are data vectors, \f$ \lambda_i \f$ are the
     * regularization weights, and \f$ W \f$ and \f$ V_i \f$ are scaling operators.
     */
    template <typename data_t = real_t>
    class TikhonovProblem : public Problem<data_t>
    {
    public:
        /**
         * \brief Constructor for a Tikhonov problem
         *
         * \param[in] wlsProblem a wls problem specifying the data term and the initial solution
         * \param[in] regTerms the regularization terms, all should be of type L2NormPow2 or
         * WeightedL2NormPow2
         */
        TikhonovProblem(const WLSProblem<data_t>& wlsProblem,
                        const std::vector<RegularizationTerm<data_t>>& regTerms);

        /**
         * \brief Constructor for a Tikhonov problem
         *
         * \param[in] wlsProblem a wls problem specifying the data term and the initial solution
         * \param[in] regTerm the regularization term, should be of type L2NormPow2 or
         * WeightedL2NormPow2
         */
        TikhonovProblem(const WLSProblem<data_t>& wlsProblem,
                        const RegularizationTerm<data_t>& regTerm);

        /// default destructor
        ~TikhonovProblem() override = default;

    protected:
        /// default copy constructor, hidden from non-derived classes to prevent potential slicing
        TikhonovProblem(const TikhonovProblem<data_t>&) = default;

        /// implement the polymorphic clone operation
        TikhonovProblem<data_t>* cloneImpl() const override;

    private:
        /// lift from base class
        using Problem<data_t>::_regTerms;
    };
} // namespace elsa
