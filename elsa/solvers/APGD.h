
#include <optional>

#include "Solver.h"
#include "LinearOperator.h"
#include "StrongTypes.h"
#include "MaybeUninitialized.hpp"
#include "ProximalOperator.h"

namespace elsa
{
    /**
     * @brief Accelerated Proximal Gradient Descent (APGD)
     *
     * APGD minimizes function of the the same for as PGD. See the documentation there.
     *
     * This class represents a APGD solver with the following steps:
     *
     *  - @f$ x_{k} = prox_h(y_k - \mu * A^T (Ay_k - b)) @f$
     *  - @f$ t_{k+1} = \frac{1 + \sqrt{1 + 4 * t_{k}^2}}{2} @f$
     *  - @f$ y_{k+1} = x_{k} + (\frac{t_{k} - 1}{t_{k+1}}) * (x_{k} - x_{k - 1}) @f$
     *
     * APGD has a worst-case complexity result of @f$ O(1/k^2) @f$.
     *
     * References:
     * http://www.cs.cmu.edu/afs/cs/Web/People/airg/readings/2012_02_21_a_fast_iterative_shrinkage-thresholding.pdf
     * https://arxiv.org/pdf/2008.02683.pdf
     *
     * @see For a more detailed discussion of the type of problem for this solver,
     * see PGD.
     *
     * @author
     * Andi Braimllari - initial code
     * David Frank - generalization to APGD
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     */
    template <typename data_t = real_t>
    class APGD : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        APGD(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
             ProximalOperator<data_t> prox, geometry::Threshold<data_t> mu,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        APGD(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
             ProximalOperator<data_t> prox,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        APGD(const APGD<data_t>&) = delete;

        /// default destructor
        ~APGD() override = default;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * APGD
         *
         * @param[in] iterations number of iterations to execute
         *
         * @returns the approximated solution
         */
        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

    protected:
        /// implement the polymorphic clone operation
        auto cloneImpl() const -> APGD<data_t>* override;

        /// implement the polymorphic comparison operation
        auto isEqual(const Solver<data_t>& other) const -> bool override;

    private:
        /// The LASSO optimization problem
        std::unique_ptr<LinearOperator<data_t>> A_;

        DataContainer<data_t> b_;

        ProximalOperator<data_t> prox_;

        data_t lambda_{1};

        /// the step size
        MaybeUninitialized<data_t> mu_;

        /// variable affecting the stopping condition
        data_t epsilon_;
    };
} // namespace elsa
