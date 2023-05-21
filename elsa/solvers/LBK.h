#pragma once

#include <limits>
#include <optional>

#include "DataContainer.h"
#include "LinearOperator.h"
#include "MaybeUninitialized.hpp"
#include "Solver.h"
#include "StrongTypes.h"
#include "ProximalOperator.h"

namespace elsa
{
    /**
     * Linearized Bregman with Kicking is fast version of Linearized Bregman that solves problems
     * of form:
     * \min_{x} \frac{\mu}{2} || A x - b ||_2^2 + ||x||_1
     *
     * by iteratively solving:
     * \begin{aligned}
     * v^{k+1} & =v^k+A^T\left(b-A x^k\right) \\
     * x^{k+1} & =\beta * \operatorname{shrink}\left(v^{k+1}, 1 / \mu\right)
     * \end{aligned}
     *
     * and during stagnation period x^{k+1} \approx x^{k+2} \approx \cdots \approx x^{k+m} \approx
     * x^* iteration is the following: \left\{\begin{array}{l}
     * x^{k+s} \equiv x^{k+1} \\
     * v^{k+s}=v^k+s \cdot A^{\top}\left(b-A x^{k+1}\right)
     * \end{array}\right.
     *
     * where:
     * \begin{gathered}
     * s_i=\left\lceil\frac{\mu \cdot \operatorname{sign}\left(\left(A^{\top}\left(b-A
     * x^{k+1}\right)\right)_i\right)-v_i^{k+1}}{\left(A^{\top}\left(b-A
     * x^{k+1}\right)\right)_i}\right\rceil \forall i \in I_0 \\ s=\min _{i \in
     * I_0}\left\{s_i\right\} \end{gathered}
     *
     * and I_0 - set of indices of the zero elements of x
     *
     * References:
     * - https://arxiv.org/pdf/1104.0262.pdf
     * - https://web.math.ucsb.edu/~cgarcia/UGProjects/BregmanAlgorithms_JacquelineBush.pdf
     *
     * @tparam data_t
     */

    template <typename data_t = real_t>
    class LBK : public Solver<data_t>
    {
    public:
        /// Construct Linearized Bregman with Kicking
        LBK(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
            ProximalOperator<data_t> prox, data_t mu = 1, std::optional<data_t> beta = std::nullopt,
            data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicitv
        LBK(const LBK<data_t>&) = delete;

        /// default destructor
        ~LBK() override = default;

        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

    protected:
        /// implement the polymorphic clone operation
        auto cloneImpl() const -> LBK<data_t>* override;

        /// implement the polymorphic comparison operation
        auto isEqual(const Solver<data_t>& other) const -> bool override;

    private:
        /// The LASSO optimization problem
        std::unique_ptr<LinearOperator<data_t>> A_;

        DataContainer<data_t> b_;

        ProximalOperator<data_t> prox_;

        /// the step size
        data_t mu_;

        /// parameter
        data_t beta_;

        /// variable affecting the stopping condition
        data_t epsilon_;
    };
} // namespace elsa
