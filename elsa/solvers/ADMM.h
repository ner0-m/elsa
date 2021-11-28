#pragma once

#include "Solver.h"
#include "BlockLinearOperator.h"
#include "VolumeDescriptor.h"
#include "PartitionDescriptor.h"
#include "Identity.h"
#include "L2NormPow2.h"
#include "LinearResidual.h"
#include "ShearletTransform.h"
#include "SoftThresholding.h"
#include "SplittingProblem.h"
#include "Logger.h"
#include "WeightedL1Norm.h"

namespace elsa
{
    /**
     * @brief Class representing an Alternating Direction Method of Multipliers solver
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
     * If the positiveSolutionsOnly flag is set to true, then it will aim to solve the following
     * problem only for positive solutions,
     *
     *  - @f$ argmin_{x>=0}(\frac{1}{2}·\|R_{\phi}x - y\|^2_{2} + \lambda·\|SH(x)\|_{1,w}) @f$
     *
     * which is essentially a LASSO problem but with @f$ \|SH(x)\|_{1,w} @f$ instead of the usual
     * @f$ \|x\|_{1} @f$, as well as x restricted to >= 0.
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     * @tparam XSolver Solver type handling the x update
     * @tparam ZSolver ProximityOperator type handling the z update
     *
     * References:
     * https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
     * https://arxiv.org/pdf/1811.04602.pdf
     */
    template <template <typename> class XSolver, template <typename> class ZSolver,
              typename data_t = real_t>
    class ADMM : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        ADMM(const SplittingProblem<data_t>& splittingProblem, bool positiveSolutionsOnly = false)
            : Solver<data_t>(splittingProblem), _positiveSolutionsOnly{positiveSolutionsOnly}
        {
            static_assert(std::is_base_of<Solver<data_t>, XSolver<data_t>>::value,
                          "ADMM: XSolver must extend Solver");
            static_assert(std::is_base_of<ProximityOperator<data_t>, ZSolver<data_t>>::value,
                          "ADMM: ZSolver must extend ProximityOperator");
        }

        ADMM(const SplittingProblem<data_t>& splittingProblem, index_t defaultXSolverIterations,
             bool positiveSolutionsOnly = false)
            : Solver<data_t>(splittingProblem),
              _positiveSolutionsOnly{positiveSolutionsOnly},
              _defaultXSolverIterations{defaultXSolverIterations}
        {
            static_assert(std::is_base_of<Solver<data_t>, XSolver<data_t>>::value,
                          "ADMM: XSolver must extend Solver");
            static_assert(std::is_base_of<ProximityOperator<data_t>, ZSolver<data_t>>::value,
                          "ADMM: ZSolver must extend ProximityOperator");
        }

        ADMM(const SplittingProblem<data_t>& splittingProblem, index_t defaultXSolverIterations,
             data_t epsilonAbs, data_t epsilonRel, bool positiveSolutionsOnly = false)
            : Solver<data_t>(splittingProblem),
              _positiveSolutionsOnly{positiveSolutionsOnly},
              _defaultXSolverIterations{defaultXSolverIterations},
              _epsilonAbs{epsilonAbs},
              _epsilonRel{epsilonRel}
        {
            static_assert(std::is_base_of<Solver<data_t>, XSolver<data_t>>::value,
                          "ADMM: XSolver must extend Solver");
            static_assert(std::is_base_of<ProximityOperator<data_t>, ZSolver<data_t>>::value,
                          "ADMM: ZSolver must extend ProximityOperator");
        }

        ADMM(const SplittingProblem<data_t>& splittingProblem, index_t defaultXSolverIterations,
             data_t epsilonAbs, data_t epsilonRel, bool varyingPenaltyParameter,
             bool positiveSolutionsOnly = false)
            : Solver<data_t>(splittingProblem),
              _positiveSolutionsOnly{positiveSolutionsOnly},
              _defaultXSolverIterations{defaultXSolverIterations},
              _epsilonAbs{epsilonAbs},
              _epsilonRel{epsilonRel},
              _varyingPenaltyParameter{varyingPenaltyParameter}
        {
            static_assert(std::is_base_of<Solver<data_t>, XSolver<data_t>>::value,
                          "ADMM: XSolver must extend Solver");
            static_assert(std::is_base_of<ProximityOperator<data_t>, ZSolver<data_t>>::value,
                          "ADMM: ZSolver must extend ProximityOperator");
        }

        /// default destructor
        ~ADMM() override = default;

        // TODO can this be put somewhere better? can this make it to master?
        DataContainer<data_t> maxWithZero(DataContainer<data_t> dC)
        {
            DataContainer<data_t> zeroes(dC.getDataDescriptor());
            zeroes = 0;
            return cwiseMax(dC, zeroes);
        }

        /// slice a 3D container based on the specified range (both inclusive)
        // TODO ideally this ought to be implemented somewhere else, perhaps in a more general
        //  manner, but that might take quite some time, can this make it to master in the meantime?
        DataContainer<data_t> sliceByRange(index_t from, index_t to, DataContainer<data_t> dc)
        {
            index_t range = to - from + 1;
            DataContainer<data_t> res(VolumeDescriptor{
                {dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0],
                 dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1], range}});

            index_t j = 0;
            for (index_t i = 0; i < dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[2];
                 ++i) {
                if (i >= from && i <= to) {
                    res.slice(j++) = dc.slice(i);
                }
            }

            return res;
        }

        /// bare-bones implementation of one scenario of torch.unsqueeze, adds a one dimension in
        /// the end, e.g. from (n, n) to (n, n, 1)
        // TODO ideally this ought to be implemented somewhere else, perhaps in a more general
        //  manner, but that might take quite some time, can this make it to master in the meantime?
        DataContainer<data_t> unsqueezeLastDimension(DataContainer<data_t> dc)
        {
            const DataDescriptor& dataDescriptor = dc.getDataDescriptor();
            index_t dims = dataDescriptor.getNumberOfDimensions();
            IndexVector_t coeffsPerDim = dataDescriptor.getNumberOfCoefficientsPerDimension();

            IndexVector_t expandedCoeffsPerDim(dims + 1);

            for (index_t index = 0; index < dims; ++index) {
                expandedCoeffsPerDim[index] = coeffsPerDim[index];
            }
            expandedCoeffsPerDim[dims] = 1;

            DataContainer<data_t> newDC(VolumeDescriptor{expandedCoeffsPerDim});

            for (index_t i = 0; i < dc.getSize(); ++i) { // TODO improve me
                newDC[i] = dc[i];
            }

            return newDC;
        }

        auto solveImpl(index_t iterations) -> DataContainer<data_t>& override
        {
            if (iterations == 0)
                iterations = _defaultIterations;

            auto& splittingProblem = downcast<SplittingProblem<data_t>>(*_problem);

            const auto& f = splittingProblem.getF();
            const auto& g = splittingProblem.getG();

            if (!is<L2NormPow2<data_t>>(f)) {
                throw InvalidArgumentError(
                    "ADMM::solveImpl: supported data term only of type L2NormPow2");
            }

            const auto& dataTerm = f;

            // safe as long as only LinearResidual exits
            const auto& dataTermResidual = downcast<LinearResidual<data_t>>(f.getResidual());

            std::unique_ptr<data_t> regWeight;
            /// weighting operator of the WeightedL1Norm
            std::unique_ptr<DataContainer<data_t>> w;
            std::unique_ptr<std::vector<geometry::Threshold<data_t>>> thresholds;

            if (g.size() == 1) {
                regWeight = std::make_unique<data_t>(g[0].getWeight());
                Functional<data_t>& regularizationTerm = g[0].getFunctional();

                if (!is<L0PseudoNorm<data_t>>(regularizationTerm)
                    && !is<L1Norm<data_t>>(regularizationTerm)
                    && !is<WeightedL1Norm<data_t>>(regularizationTerm)) {
                    throw InvalidArgumentError(
                        "ADMM::solveImpl: supported regularization terms are "
                        "of type L0PseudoNorm or L1Norm or WeightedL1Norm");
                }

                if (is<WeightedL1Norm<data_t>>(regularizationTerm)) {
                    auto wL1NormRegTerm = downcast<WeightedL1Norm<data_t>>(&g[0].getFunctional());

                    w = std::make_unique<DataContainer<data_t>>(
                        static_cast<DataContainer<data_t>>(wL1NormRegTerm->getWeightingOperator()));
                    thresholds = std::make_unique<std::vector<geometry::Threshold<data_t>>>(
                        ProximityOperator<data_t>::valuesToThresholds(_rho0 * *w / _rho1));
                }
            } else {
                throw InvalidArgumentError(
                    "ADMM::solveImpl: supported number of regularization terms is 1");
            }

            const auto& constraint = splittingProblem.getConstraint();
            const LinearOperator<data_t>* A = &constraint.getOperatorA();
            const LinearOperator<data_t>* B = &constraint.getOperatorB();
            const DataContainer<data_t>* c = &constraint.getDataVectorC();

            std::unique_ptr<LinearOperator<data_t>> tA;
            std::unique_ptr<LinearOperator<data_t>> tB;

            if (_positiveSolutionsOnly) {
                const DataDescriptor& domainDescriptor = dataTerm.getDomainDescriptor();

                ShearletTransform<data_t, data_t> shearletTransform =
                    downcast<ShearletTransform<data_t, data_t>>(
                        downcast<LinearResidual<data_t>>(g[0].getFunctional().getResidual())
                            .getOperator());

                index_t n = domainDescriptor.getNumberOfCoefficientsPerDimension()[0];
                index_t layers = shearletTransform.getNumOfLayers();

                VolumeDescriptor layersPlusOneDescriptor{{n, n, layers + 1}};

                IndexVector_t slicesInBlock(2);
                slicesInBlock << layers, 1;

                /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
                std::vector<std::unique_ptr<LinearOperator<data_t>>> opsOfA(0);
                Identity<data_t> identity(domainDescriptor);
                opsOfA.push_back((_rho1 * shearletTransform).clone());
                opsOfA.push_back((_rho2 * identity).clone());
                BlockLinearOperator<data_t> tempA(
                    domainDescriptor, PartitionDescriptor{layersPlusOneDescriptor, slicesInBlock},
                    opsOfA, BlockLinearOperator<data_t>::BlockType::ROW);

                tA = tempA.clone();
                A = &(*tA);

                /// B = diag(−ρ_1*1_Ln^2, −ρ_2*1_n^2) ∈ R ^ (L+1)n^2 × (L+1)n^2
                DataContainer<data_t> factorsOfB(layersPlusOneDescriptor);
                for (int ind = 0; ind < factorsOfB.getSize(); ++ind) {
                    if (ind < (n * n * layers)) {
                        factorsOfB[ind] = -1 * _rho1;
                    } else {
                        factorsOfB[ind] = -1 * _rho2;
                    }
                }
                Scaling<data_t> tempB(layersPlusOneDescriptor, factorsOfB);

                tB = tempB.clone();
                B = &(*tB);

                DataContainer<data_t> tempC(layersPlusOneDescriptor);
                tempC = 0;

                if (*c != tempC) {
                    throw InvalidArgumentError("ADMM: the vector c of the constraint should be 0");
                }
            }

            DataContainer<data_t> x(A->getDomainDescriptor());
            x = 0;

            DataContainer<data_t> z(B->getDomainDescriptor());
            z = 0;

            DataContainer<data_t> u(c->getDataDescriptor());
            u = 0;

            Logger::get("ADMM")->info("{:*^20}|{:*^20}|{:*^20}|{:*^20}|{:*^20}|{:*^20}",
                                      "iteration", "xL2NormSq", "zL2NormSq", "uL2NormSq",
                                      "rkL2Norm", "skL2Norm");
            for (index_t iter = 0; iter < iterations; ++iter) {
                LinearResidual<data_t> xLinearResidual(*A, *c - B->apply(z) - u);
                RegularizationTerm xRegTerm(_rho, L2NormPow2<data_t>(xLinearResidual));
                Problem<data_t> xUpdateProblem(dataTerm, xRegTerm, x);

                XSolver<data_t> xSolver(xUpdateProblem);
                x = xSolver.solve(_defaultXSolverIterations);

                DataContainer<data_t> zPrev = z;

                if (!_positiveSolutionsOnly) {
                    ZSolver<data_t> zProxOp(A->getRangeDescriptor());
                    z = zProxOp.apply(x + u, geometry::Threshold{*regWeight / _rho});
                } else {
                    ZSolver<data_t> zProxOp(w->getDataDescriptor());

                    ShearletTransform<data_t, data_t> shearletTransform =
                        downcast<ShearletTransform<data_t, data_t>>(
                            downcast<LinearResidual<data_t>>(g[0].getFunctional().getResidual())
                                .getOperator());

                    index_t layers = shearletTransform.getNumOfLayers();
                    DataContainer<data_t> P1u = sliceByRange(0, layers - 1, u);
                    DataContainer<data_t> P1z =
                        zProxOp.apply(shearletTransform.apply(x) + P1u, *thresholds);

                    DataContainer<data_t> P2u = u.slice(layers);
                    DataContainer<data_t> P2z = maxWithZero(unsqueezeLastDimension(x) + P2u);

                    z = concatenate(P1z, P2z);
                }

                u += A->apply(x) + B->apply(z) - *c;

                /// primal residual at iteration k
                DataContainer<data_t> rk = A->apply(x) + B->apply(z) - *c;
                /// dual residual at iteration k
                DataContainer<data_t> sk = _rho * A->applyAdjoint(B->apply(z - zPrev));

                data_t rkL2Norm = rk.l2Norm();
                data_t skL2Norm = sk.l2Norm();

                Logger::get("ADMM")->info("{:<19}| {:<19}| {:<19}| {:<19}| {:<19}| {:<19}", iter,
                                          x.squaredL2Norm(), z.squaredL2Norm(), u.squaredL2Norm(),
                                          rkL2Norm, skL2Norm);

                /// variables for the stopping criteria
                data_t Axnorm = A->apply(x).l2Norm();
                data_t Bznorm = B->apply(z).l2Norm();
                const data_t cL2Norm = !dataTermResidual.hasDataVector()
                                           ? static_cast<data_t>(0.0)
                                           : dataTermResidual.getDataVector().l2Norm();

                const data_t epsRelMax = _epsilonRel * std::max(std::max(Axnorm, Bznorm), cL2Norm);
                const auto epsilonPri = (std::sqrt(rk.getSize()) * _epsilonAbs) + epsRelMax;

                data_t Atunorm = A->applyAdjoint(_rho * u).l2Norm();
                const data_t epsRelL2Norm = _epsilonRel * Atunorm;
                const auto epsilonDual = (std::sqrt(sk.getSize()) * _epsilonAbs) + epsRelL2Norm;

                if (rkL2Norm <= epsilonPri && skL2Norm <= epsilonDual) {
                    Logger::get("ADMM")->info("SUCCESS: Reached convergence at {}/{} iterations ",
                                              iter, iterations);

                    getCurrentSolution() = x;
                    return getCurrentSolution();
                }

                /// varying penalty parameter
                if (_varyingPenaltyParameter) {
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
            }

            Logger::get("ADMM")->warn("Failed to reach convergence at {} iterations", iterations);

            getCurrentSolution() = x;
            return getCurrentSolution();
        }

    protected:
        /// lift the base class method getCurrentSolution
        using Solver<data_t>::getCurrentSolution;

        /// implement the polymorphic clone operation
        auto cloneImpl() const -> ADMM<XSolver, ZSolver, data_t>* override
        {
            return new ADMM<XSolver, ZSolver, data_t>(
                downcast<SplittingProblem<data_t>>(*_problem));
        }

    private:
        /// lift the base class variable _problem
        using Solver<data_t>::_problem;

        /// flag to indicate whether to solve for positive solutions or for any solution
        bool _positiveSolutionsOnly{false};

        /// the default number of iterations for ADMM
        index_t _defaultIterations{20};

        /// the default number of iterations for the XSolver
        index_t _defaultXSolverIterations{5};

        /// @f$ \rho @f$ values from the problem definition
        data_t _rho{1};
        data_t _rho0{1.0 / 2}; // consider as hyper-parameters
        /// values specific to the problem statement in T. A. Bubba et al.
        data_t _rho1{1.0 / 2}; // consider as hyper-parameters
        data_t _rho2{1};       // just set it to 1, at least initially

        /// flag to indicate whether to utilize the Varying Penalty Parameter extension, refer
        /// to the section 3.4.1 (Boyd) for further details
        bool _varyingPenaltyParameter{false};

        /// variables for varying penalty parameter @f$ \rho @f$
        data_t _mu{10};
        data_t _tauIncr{2};
        data_t _tauDecr{2};

        /// variables for the stopping criteria
        data_t _epsilonAbs{1e-5f};
        data_t _epsilonRel{1e-5f};
    };
} // namespace elsa
