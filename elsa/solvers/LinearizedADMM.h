#pragma once

#include "Solver.h"
#include "LinearOperator.h"
#include "ProximalOperator.h"
#include "DataContainer.h"

namespace elsa
{
    /**
     * @brief Class representing the linearized version of ADMM.
     *
     * The general form of ADMM solves the following optimization problem:
     * \f[
     * \min f(x) + g(z) \\
     * \text{s.t. } Ax + Bz = c
     * \f]
     * with \f$x \in \mathbb{R}^n\f$, \f$z \in \mathbb{R}^m\f$, \f$A \in \mathbb{R}^{p\times n}\f$,
     * \f$B \in \mathbb{R}^{p\times m}\f$ and \f$c \in \mathbb{R}^p\f$
     *
     * The linearized version assumes \f$A = K\f$ to be any linear operator, \f$B = - Id\f$
     * and \f$c = 0\f$. Hence, the optimization problem reduces to:
     * \f[
     * \min f(x) + g(Ax)
     * \f]
     *
     * The main benefit of this is, that in many applications both \f$f\f$ and \f$g\f$ can be
     * _simple_ functionals, for which proximal operators are easy to evaluate.
     * I.e. The usual least square formulation without regularization is achieved
     * with: \f$f(x) = 0\f$, and \f$g(z) = || z - b ||_2^2 \f$, with \f$ z = Kx\f$.
     * For both the proximal operator is analytically known.
     *
     * Many other problems can be converted to this form in a similar fashion by "stacking" the
     * operator \f$K\f$. I.e. \f$L_1\f$ regularization can as: \f$f(x) = 0\f$, and \f$g(z) = || z_1
     * - b ||_2^2 + || z_2 ||_1\f$, with \f$ K = \begin{bmatrix} A \\ Id \end{bmatrix}\f$,
     * and \f$z = \begin{bmatrix} z_1 \\ z_2 \end{bmatrix}\f$. Further, constraints can be added
     * easily via the function \f$f\f$, by setting it to some indicator function (i.e.
     * non-negativity or similar).
     *
     * References:
     * - Distributed Optimization and Statistical Learning via the Alternating Direction Method of
     * Multipliers, by Boyd et al.
     * - Chapter 5.3 of "An introduction to continuous optimization for imaging", by Chambolle and
     * Pock
     *
     * TODO:
     * - Once implemented, take functionals which know their respective proximal instead of handing
     * proximal operators directly
     */
    template <class data_t>
    class LinearizedADMM final : public Solver<data_t>
    {
    public:
        /// Construct the linearized version of ADMM given two proximal
        /// operators for \f$f\f$ and \f$g\f$ respectively, the potentially
        /// stacked operator \f$K\f$ and the two step length parameters
        /// \f$\sigma\f$ and \f$\tau\f$.
        ///
        /// To ensure convergence it is checked if \f$0 < \tau < \frac{\sigma}{||K||_2^2}\f$.
        /// Here \f$||K||_2^2\f$ is the operator norm, which is approximated using
        /// a couple if iterations of the power method, to apprxoximate the largest
        /// eigenvalue of the operator. If `computeKNorm` is `false`, the
        /// computation is not performed, and it is assumed the above inequality
        /// holds.
        LinearizedADMM(const LinearOperator<data_t>& K, ProximalOperator<data_t> proxf,
                       ProximalOperator<data_t> proxg, SelfType_t<data_t> sigma,
                       SelfType_t<data_t> tau, bool computeKNorm = true);

        /// default destructor
        ~LinearizedADMM() override = default;

        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

        LinearizedADMM<data_t>* cloneImpl() const override;

        bool isEqual(const Solver<data_t>& other) const override;

    private:
        /// The stacked linear operator \f$K\f$
        std::unique_ptr<LinearOperator<data_t>> K_;

        /// The proximal operator for \f$f\f$
        ProximalOperator<data_t> proxf_;

        /// The proximal operator for \f$g\f$
        ProximalOperator<data_t> proxg_;

        /// Step length \f$\sigma\f$
        data_t sigma_;

        /// Step length \f$\tau\f$
        data_t tau_;
    };
} // namespace elsa
