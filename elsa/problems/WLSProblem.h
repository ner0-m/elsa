#pragma once

#include "Problem.h"
#include "Scaling.h"
#include "LinearResidual.h"

namespace elsa
{
    /**
     * \brief Class representing a weighted least squares problem.
     *
     * \author Jakob Vogel - initial code
     * \author Matthias Wieczorek - rewrite
     * \author Tobias Lasser - another rewrite, modernization
     * \author Nikola Dinev - added conversion constructor
     *
     * \tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents a weighted least squares optimization problem, i.e.
     * \f$ \underset{x}{\operatorname{argmin}} \frac{1}{2} \| Ax - b \|_{W,2}^2 \f$, where \f$ W \f$ is a weighting (scaling)
     * operator, \f$ A \f$ is a linear operator and \f$ b \f$ is a data vector.
     */
    template <typename data_t = real_t>
    class WLSProblem : public Problem<data_t>
    {
    public:
        /**
         * \brief Constructor for the wls problem, accepting W, A, b, and an initial guess x0
         *
         * \param[in] W scaling operator for weighting
         * \param[in] A linear operator
         * \param[in] b data vector
         * \param[in] x0 initial value for the current estimated solution
         */
        WLSProblem(const Scaling<data_t>& W, const LinearOperator<data_t>& A,
                   const DataContainer<data_t>& b, const DataContainer<data_t>& x0);

        /**
         * \brief Constructor for the wls problem, accepting W, A, and b
         *
         * \param[in] W scaling operator for weighting
         * \param[in] A linear operator
         * \param[in] b data vector
         */
        WLSProblem(const Scaling<data_t>& W, const LinearOperator<data_t>& A,
                   const DataContainer<data_t>& b);

        /**
         * \brief Constructor for the (w)ls problem, accepting A, b, and an initial guess x0 (no
         * weights)
         *
         * \param[in] A linear operator
         * \param[in] b data vector
         * \param[in] x0 initial value for the current estimated solution
         */
        WLSProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                   const DataContainer<data_t>& x0);

        /**
         * \brief Constructor for the (w)ls problem, accepting A and b (no weights)
         *
         * \param[in] A linear operator
         * \param[in] b data vector
         */
        WLSProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b);

        /**
         * \brief Constructor for converting a general optimization problem to a WLS problem
         *
         * \param[in] problem the problem to be converted
         *
         * Only problems that consist exclusively of (Weighted)L2NormPow2 terms can be converted.
         * The (Weighted)L2NormPow2 should be acting on a LinearResidual.
         *
         * Acts as a copy constructor if the supplied optimization problem is a quadric problem.
         */
        explicit WLSProblem(const Problem<data_t>& problem);

        /// default destructor
        ~WLSProblem() override = default;

    protected:
        /// implement the polymorphic clone operation
        WLSProblem<data_t>* cloneImpl() const override;

    private:
        /// converts an optimization problem to a (Weighted)L2NormPow2 functional
        static std::unique_ptr<Functional<data_t>> wlsFromProblem(const Problem<data_t>& problem);
    };
} // namespace elsa
