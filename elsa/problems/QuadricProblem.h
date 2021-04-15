#pragma once

#include "LinearOperator.h"
#include "DataContainer.h"
#include "WLSProblem.h"
#include "Quadric.h"

namespace elsa
{

    /**
     * @brief Class representing a quadric problem
     *
     * @author Nikola Dinev
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents a quadric problem, i.e.
     * \f$ \argmin_x \frac{1}{2} x^tAx - x^tb \f$
     * where \f$ A \f$ is a symmetric positive-definite operator and \f$ b \f$ is a data vector.
     */
    template <typename data_t = real_t>
    class QuadricProblem : public Problem<data_t>
    {
    public:
        /**
         * @brief Constructor for the quadric problem accepting A, b, and an initial guess x0
         *
         * @param[in] A linear operator
         * @param[in] b data vector
         * @param[in] x0 initial value for the current estimated solution
         * @param[in] spdA flag whether A is symmetric positive-definite
         *
         * Sets up the quadric problem
         * \f$ \argmin_x \frac{1}{2} x^tAx - x^tb \f$ if \f$ A \f$ is spd, and
         * \f$ \argmin_x \frac{1}{2} x^tA^tAx - x^tA^tb \f$ if \f$ A \f$ is not spd
         *
         * @warning A must be nonsingular even if it is not spd.
         *
         * Please note: For a general complex operator \f$ A \f$, it does not necessarily
         * hold that \f$ A^*A \f$ is spd. Therefore, QuadricProblem is restricted to the
         * non-complex variants.
         */
        QuadricProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                       const DataContainer<data_t>& x0, bool spdA);

        /**
         * @brief Constructor for the quadric problem accepting A and b
         *
         * @param[in] A linear operator
         * @param[in] b data vector
         * @param[in] spdA flag whether A is symmetric positive-definite
         *
         * Sets up the quadric problem
         * \f$ \argmin_x \frac{1}{2} x^tAx - x^tb \f$ if \f$ A \f$ is spd, and
         * \f$ \argmin_x \frac{1}{2} x^tA^tAx - x^tA^tb \f$ if \f$ A \f$ is not spd
         *
         * @warning A must be nonsingular even if it is not spd.
         *
         * Please note: For a general complex operator \f$ A \f$, it does not necessarily
         * hold that \f$ A^*A \f$ is spd. Therefore, QuadricProblem is restricted to the
         * non-complex variants.
         */
        QuadricProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b, bool spdA);

        /**
         * @brief Constructor for the quadric problem accepting a quadric and an initial guess x0
         *
         * @param[in] quadric a Quadric containing the entire problem formulation
         * @param[in] x0 initial value for the current estimated solution
         */
        QuadricProblem(const Quadric<data_t>& quadric, const DataContainer<data_t>& x0);

        /**
         * @brief Constructor for the quadric problem accepting a quadric
         *
         * @param[in] quadric a Quadric containing the entire problem formulation
         */
        explicit QuadricProblem(const Quadric<data_t>& quadric);

        /**
         * @brief Constructor for converting a general optimization problem to a quadric one
         *
         * @param[in] problem the problem to be converted
         *
         * Only problems that consist exclusively of Quadric and (Weighted)L2NormPow2 terms
         * can be converted. If (Weighted)L2NormPow2 terms are present, they should be acting
         * on a LinearResidual.
         *
         * Acts as a copy constructor if the supplied optimization problem is a quadric problem.
         */
        explicit QuadricProblem(const Problem<data_t>& problem);

    protected:
        /// implement the polymorphic clone operation
        QuadricProblem<data_t>* cloneImpl() const override;

    private:
        /// lift from base class
        using Problem<data_t>::_dataTerm;

        /// returns an expression for calculating the gradient of the regularization term
        static LinearResidual<data_t>
            getGradientExpression(const RegularizationTerm<data_t>& regTerm);

        /// converts an optimization problem to a quadric
        static std::unique_ptr<Quadric<data_t>> quadricFromProblem(const Problem<data_t>& problem);
    };
} // namespace elsa