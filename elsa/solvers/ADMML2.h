#pragma once

#include <memory>
#include <type_traits>
#include <optional>

#include "Solver.h"
#include "DataContainer.h"
#include "LinearOperator.h"
#include "ProximalOperator.h"
#include "TypeTraits.hpp"
#include "elsaDefines.h"

namespace elsa
{
    /**
     * @brief Class representing an Alternating Direction Method of Multipliers solver
     * for a specific subset of constraints
     *
     * The general form of ADMM solves the following optimization problem:
     * \f[
     * \min f(x) + g(z) \\
     * \text{s.t. } Ax + Bz = c
     * \f]
     * with \f$x \in \mathbb{R}^n\f$, \f$z \in \mathbb{R}^m\f$, \f$A \in \mathbb{R}^{p\times n}\f$,
     * \f$B \in \mathbb{R}^{p\times m}\f$ and \f$c \in \mathbb{R}^p\f$
     *
     * This specific version solves the problem of the form:
     * \f[
     * \min \frac{1}{2} || Op x - b ||_2^2 + g(z) \\
     * \text{s.t. } Ax = z
     * \f]
     * with \f$B = Id\f$ and \f$c = 0\f$. Further: \f$f(x) = || Op x - b ||_2^2\f$.
     *
     * This version of ADMM is useful, as the proximal operator is not known for
     * the least squares functional, and this specifically implements and optimization of the
     * first update step of ADMM. In this implementation, this is done via CGLS.
     *
     * The update steps for ADMM are:
     * \f[
     * x_{k+1} = \argmin_{x} \frac{1}{2}||Op x - b||_2^2 + \frac{1}{2\tau} ||Ax - z_k + u_k||_2^2 \\
     * z_{k+1} = \prox_{\tau g}(Ax_{k+1} + u_{k}) \\
     * u_{k+1} = u_k + Ax_{k+1} - z_{k+1}
     * \f]
     *
     * This is further useful to solve problems such as TV, by setting the \f$A = \nabla\f$.
     * And \f$ g = || \dot ||_1\f$
     *
     * References:
     * - Distributed Optimization and Statistical Learning via the Alternating Direction Method of
     * Multipliers, by Boyd et al.
     * - Chapter 5.3 of "An introduction to continuous optimization for imaging", by Chambolle and
     * Pock
     */
    template <typename data_t = real_t>
    class ADMML2 : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        ADMML2(const LinearOperator<data_t>& op, const DataContainer<data_t>& b,
               const LinearOperator<data_t>& A, const ProximalOperator<data_t>& proxg,
               std::optional<data_t> tau, index_t ninneriters = 5);

        /// default destructor
        ~ADMML2() override = default;

        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

    protected:
        /// implement the polymorphic clone operation
        ADMML2<data_t>* cloneImpl() const override;

        /// implement the polymorphic equality operation
        bool isEqual(const Solver<data_t>& other) const override;

    private:
        std::unique_ptr<LinearOperator<data_t>> op_;

        DataContainer<data_t> b_;

        std::unique_ptr<LinearOperator<data_t>> A_;

        ProximalOperator<data_t> proxg_;

        /// @f$ \tau @f$ from the problem definition
        data_t tau_{1};

        index_t ninneriters_;
    };
} // namespace elsa
