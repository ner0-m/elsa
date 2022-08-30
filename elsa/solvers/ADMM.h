#pragma once

#include <memory>

#include "Solver.h"
#include "ProximityOperator.h"
#include "SplittingProblem.h"
#include "L0PseudoNorm.h"
#include "L1Norm.h"
#include "L2NormPow2.h"
#include "LinearResidual.h"
#include "Logger.h"

namespace elsa
{
    /**
     * @brief Class representing an Alternating Direction Method of Multipliers solver
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     * @tparam XSolver Solver type handling the x update
     * @tparam ZSolver ProximityOperator type handling the z update
     *
     * ADMM solves minimization splitting problems of the form
     * @f$ x \mapsto f(x) + g(z) @f$ such that @f$ Ax + Bz = c @f$.
     * Commonly regularized optimization problems can be rewritten in such a form by using variable
     * splitting.
     *
     * ADMM can be expressed in the following scaled form
     *
     *  - @f$ x_{k+1} = argmin_{x}(f(x) + (\rho/2) ·\| Ax + Bz_{k} - c + u_{k}\|^2_2) @f$
     *  - @f$ z_{k+1} = argmin_{z}(g(z) + (\rho/2) ·\| Ax_{k+1} + Bz - c + u_{k}\|^2_2) @f$
     *  - @f$ u_{k+1} = u_{k} + Ax_{k+1} + Bz_{k+1} - c @f$
     *
     * References:
     * https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
     */
    template <template <typename> class XSolver, template <typename> class ZSolver,
              typename data_t = real_t>
    class ADMM : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        ADMM(const SplittingProblem<data_t>& splittingProblem)
            : Solver<data_t>(),
              _problem(static_cast<SplittingProblem<data_t>*>(splittingProblem.clone().release()))
        {
            static_assert(std::is_base_of<Solver<data_t>, XSolver<data_t>>::value,
                          "ADMM: XSolver must extend Solver");

            static_assert(std::is_base_of<ProximityOperator<data_t>, ZSolver<data_t>>::value,
                          "ADMM: ZSolver must extend ProximityOperator");
        }

        ADMM(const SplittingProblem<data_t>& splittingProblem, index_t defaultXSolverIterations)
            : Solver<data_t>(),
              _problem(static_cast<SplittingProblem<data_t>*>(splittingProblem.clone().release())),
              _defaultXSolverIterations{defaultXSolverIterations}
        {
            static_assert(std::is_base_of<Solver<data_t>, XSolver<data_t>>::value,
                          "ADMM: XSolver must extend Solver");

            static_assert(std::is_base_of<ProximityOperator<data_t>, ZSolver<data_t>>::value,
                          "ADMM: ZSolver must extend ProximityOperator");
        }

        ADMM(const SplittingProblem<data_t>& splittingProblem, index_t defaultXSolverIterations,
             data_t epsilonAbs, data_t epsilonRel)
            : Solver<data_t>(),
              _problem(splittingProblem),
              _defaultXSolverIterations{defaultXSolverIterations},
              _epsilonAbs{epsilonAbs},
              _epsilonRel{epsilonRel}
        {
            static_assert(std::is_base_of<Solver<data_t>, XSolver<data_t>>::value,
                          "ADMM: XSolver must extend Solver");

            static_assert(std::is_base_of<ProximityOperator<data_t>, ZSolver<data_t>>::value,
                          "ADMM: ZSolver must extend ProximityOperator");
        }

        /// default destructor
        ~ADMM() override = default;

        auto solveImpl(index_t iterations) -> DataContainer<data_t>& override
        {
            const auto& f = _problem->getF();
            const auto& g = _problem->getG();

            const auto& dataTerm = f;

            if (!is<L2NormPow2<data_t>>(dataTerm)) {
                throw std::invalid_argument(
                    "ADMM::solveImpl: supported data term only of type L2NormPow2");
            }

            // Safe as long as only LinearResidual exits
            const auto& dataTermResidual = downcast<LinearResidual<data_t>>(f.getResidual());

            if (g.size() != 1) {
                throw std::invalid_argument(
                    "ADMM::solveImpl: supported number of regularization terms is 1");
            }

            data_t regWeight = g[0].getWeight();
            Functional<data_t>& regularizationTerm = g[0].getFunctional();

            if (!is<L0PseudoNorm<data_t>>(regularizationTerm)
                && !is<L1Norm<data_t>>(regularizationTerm)) {
                throw std::invalid_argument("ADMM::solveImpl: supported regularization terms are "
                                            "of type L0PseudoNorm or L1Norm");
            }

            const auto& constraint = _problem->getConstraint();
            const auto& A = constraint.getOperatorA();
            const auto& B = constraint.getOperatorB();
            const auto& c = constraint.getDataVectorC();

            DataContainer<data_t> x(A.getRangeDescriptor());
            x = 0;

            DataContainer<data_t> z(B.getRangeDescriptor());
            z = 0;

            DataContainer<data_t> u(c.getDataDescriptor());
            u = 0;

            Logger::get("ADMM")->info("{:*^20}|{:*^20}|{:*^20}|{:*^20}|{:*^20}|{:*^20}",
                                      "iteration", "xL2NormSq", "zL2NormSq", "uL2NormSq",
                                      "rkL2Norm", "skL2Norm");

            for (index_t iter = 0; iter < iterations; ++iter) {
                LinearResidual<data_t> xLinearResidual(A, c - B.apply(z) - u);
                RegularizationTerm xRegTerm(_rho / 2, L2NormPow2<data_t>(xLinearResidual));
                Problem<data_t> xUpdateProblem(dataTerm, xRegTerm, x);

                XSolver<data_t> xSolver(xUpdateProblem);
                x = xSolver.solve(_defaultXSolverIterations);

                DataContainer<data_t> rk = x;
                DataContainer<data_t> zPrev = z;
                data_t Axnorm = x.l2Norm();

                /// For future reference, below is listed the problem to be solved by the z update
                /// solver. Refer to the documentation of ADMM for further details.
                // LinearResidual<data_t> zLinearResidual(B, c - A.apply(x) - u);
                // RegularizationTerm zRegTerm(_rho / 2, L2NormPow2<data_t>(zLinearResidual));
                // Problem<data_t> zUpdateProblem(regularizationTerm, zRegTerm, z);

                ZSolver<data_t> zProxOp(A.getRangeDescriptor());
                z = zProxOp.apply(x + u, geometry::Threshold{regWeight / _rho});

                rk -= z;
                DataContainer<data_t> sk = zPrev - z;
                sk *= _rho;

                u += A.apply(x) + B.apply(z) - c;

                DataContainer<data_t> Atu = u;
                Atu *= _rho;
                data_t rkL2Norm = rk.l2Norm();
                data_t skL2Norm = sk.l2Norm();

                Logger::get("ADMM")->info("{:<19}| {:<19}| {:<19}| {:<19}| {:<19}| {:<19}", iter,
                                          x.squaredL2Norm(), z.squaredL2Norm(), u.squaredL2Norm(),
                                          rkL2Norm, skL2Norm);

                /// variables for the stopping criteria
                const data_t cL2Norm = !dataTermResidual.hasDataVector()
                                           ? static_cast<data_t>(0.0)
                                           : dataTermResidual.getDataVector().l2Norm();
                const data_t epsRelMax =
                    _epsilonRel * std::max(std::max(Axnorm, z.l2Norm()), cL2Norm);
                const auto epsilonPri = (std::sqrt(rk.getSize()) * _epsilonAbs) + epsRelMax;

                const data_t epsRelL2Norm = _epsilonRel * Atu.l2Norm();
                const auto epsilonDual = (std::sqrt(sk.getSize()) * _epsilonAbs) + epsRelL2Norm;

                if (rkL2Norm <= epsilonPri && skL2Norm <= epsilonDual) {
                    Logger::get("ADMM")->info("SUCCESS: Reached convergence at {}/{} iterations ",
                                              iter, iterations);

                    _problem->getCurrentSolution() = x;
                    return _problem->getCurrentSolution();
                }

                /// varying penalty parameter
                if (std::abs(_tauIncr - static_cast<data_t>(1.0))
                        > std::numeric_limits<data_t>::epsilon()
                    || std::abs(_tauDecr - static_cast<data_t>(1.0))
                           > std::numeric_limits<data_t>::epsilon()) {
                    if (rkL2Norm > _mu * skL2Norm) {
                        _rho *= _tauIncr;
                        u /= _tauIncr;
                    } else if (skL2Norm > _mu * rkL2Norm) {
                        _rho /= _tauDecr;
                        u *= _tauDecr;
                    }
                }
            }

            Logger::get("ADMM")->warn("Failed to reach convergence at {} iterations", iterations);

            _problem->getCurrentSolution() = x;
            return _problem->getCurrentSolution();
        }

    protected:
        /// implement the polymorphic clone operation
        auto cloneImpl() const -> ADMM<XSolver, ZSolver, data_t>* override
        {
            return new ADMM<XSolver, ZSolver, data_t>(
                downcast<SplittingProblem<data_t>>(*_problem));
        }

        bool isEqual(const Solver<data_t>& other) const
        {
            auto otherADMM = downcast_safe<ADMM>(&other);
            if (!otherADMM)
                return false;

            if (_problem != otherADMM->_problem)
                return false;

            return _rho == otherADMM->_rho && _mu == otherADMM->_mu
                   && _tauIncr == otherADMM->_tauIncr && _tauDecr == otherADMM->_tauDecr;
        }

    private:
        /// Splitting problem to solve
        /// TODO: Remove requirement of unique_ptr
        std::unique_ptr<SplittingProblem<data_t>> _problem;

        /// the default number of iterations for the XSolver
        index_t _defaultXSolverIterations{5};

        /// @f$ \rho @f$ from the problem definition
        data_t _rho{1};

        /// variables for varying penalty parameter @f$ \rho @f$
        data_t _mu{10};
        data_t _tauIncr{2};
        data_t _tauDecr{2};

        /// variables for the stopping criteria
        data_t _epsilonAbs{1e-5f};
        data_t _epsilonRel{1e-5f};
    };
} // namespace elsa
