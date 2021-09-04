#pragma once

#include "Solver.h"
#include "VolumeDescriptor.h"
#include "SoftThresholding.h"
#include "SplittingProblem.h"
#include "BlockLinearOperator.h"
#include "WeightedL1Norm.h"
#include "Indicator.h"
#include "L2NormPow2.h"
#include "LinearResidual.h"
#include "Logger.h"
#include "ShearletTransform.h"

namespace elsa
{
    /**
     * @brief Class representing an Alternating Direction Method of Multipliers solver for
     *
     *  - @f$ argmin_{f>=0}(\frac{1}{2}·\|R_{\phi}f - y\|^2_{2} + \lambda·\|SH(f)\|_{1,w}) @f$
     *
     * which is essentially a LASSO problem but with @f$ \|SH(x)\|_{1,w} @f$ instead of the usual
     * @f$ \|x\|_{1} @f$, as well as f restricted to >= 0.
     *
     * TODO note that here we also require for f >= 0, unlike in the regular ADMM
     *
     * TODO potential initial setup
     *  f^0 := R_φTy
     *  z^0 := 0
     *  u^0 := 0
     *  ρ_2 = 1
     *  ρ_0, ρ_1 as hyperparameters
     *  Note that the algorithm converges independently of the choice of these parameters
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     * @tparam FSolver Solver type handling the f update
     *
     * TODO this class should EVENTUALLY be merged to the existing ADMM solver
     *
     * References:
     * https://arxiv.org/pdf/1811.04602.pdf
     */
    template <template <typename> class FSolver, typename data_t = real_t>
    class SHADMM : public Solver<data_t>
    {
    public:
        SHADMM(const SplittingProblem<data_t>& splittingProblem) : Solver<data_t>(splittingProblem)
        {
            static_assert(std::is_base_of<Solver<data_t>, FSolver<data_t>>::value,
                          "ADMM: FSolver must extend Solver");
        }

        /// default destructor
        ~SHADMM() override = default;

        DataContainer<data_t> maxWithZero(DataContainer<data_t> dC)
        {
            DataContainer<data_t> zeroes(dC.getDataDescriptor());
            zeroes = 0;
            return cwiseMax(dC, zeroes);
        }

        DataContainer<data_t> getFirstNElements(index_t n, DataContainer<data_t> dc)
        {
            DataContainer<data_t> res(dc.getDataDescriptor());
            for (index_t i = 0; i < dc.getSize(); i++) {
                if (i >= n) {
                    res[i] = 0;
                }
                res[i] = dc[i];
            }
            return res;
        }

        DataContainer<data_t> getLastNElements(index_t n, DataContainer<data_t> dc)
        {
            DataContainer<data_t> res(dc.getDataDescriptor());
            for (index_t i = 0; i < dc.getSize(); i++) {
                if (i >= n) {
                    res[i] = dc[i];
                }
                res[i] = 0;
            }
            return res;
        }

        auto solveImpl(index_t iterations) -> DataContainer<data_t>& override
        {
            if (iterations == 0)
                iterations = _defaultIterations;

            auto& splittingProblem = downcast<SplittingProblem<data_t>>(*_problem);

            const auto& F = splittingProblem.getF();
            const auto& G = splittingProblem.getG();

            const auto& dataTerm = F;

            if (!is<L2NormPow2<data_t>>(dataTerm)) {
                throw std::invalid_argument(
                    "SHADMM::solveImpl: supported data term only of type L2NormPow2");
            }

            const auto& dataTermResidual = downcast<LinearResidual<data_t>>(F.getResidual());

            /// TODO now we have two here? the WL1Norm and Indicator, they don't seem to be used (we
            ///  currently only get the weight from WL1Norm), would be used if SoftThresholding
            ///  accepted them in the constructor
            if (G.size() != 2) {
                throw std::invalid_argument(
                    "SHADMM::solveImpl: supported number of regularization terms is 2");
            }

            if (G[0].getWeight() != G[1].getWeight()) {
                throw std::invalid_argument(
                    "SHADMM::solveImpl: regularization weights should match");
            }

            if (!is<WeightedL1Norm<data_t>>(&G[0].getFunctional())
                || !is<Indicator<data_t>>(&G[1].getFunctional())) {
                throw std::invalid_argument("SHADMM::solveImpl: supported regularization terms are "
                                            "of type WeightedL1Norm and Indicator, respectively");
            }

            auto wL1NormRegTerm = downcast<WeightedL1Norm<data_t>>(&G[0].getFunctional());

            auto w = static_cast<DataContainer<data_t>>(wL1NormRegTerm->getWeightingOperator());

            const auto& constraint = splittingProblem.getConstraint();
            /// should come as // AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
            const BlockLinearOperator<data_t>& A =
                downcast<BlockLinearOperator<data_t>>(constraint.getOperatorA());
            /// should come as // B = diag(−ρ_1*1_Ln^2 , −ρ_2*1_n^2) ∈ R ^ (L+1)n^2 × (L+1)n^2
            const BlockLinearOperator<data_t>& B =
                downcast<BlockLinearOperator<data_t>>(constraint.getOperatorB());
            /// should come as the zero vector
            const DataContainer<data_t>& c = constraint.getDataVectorC();

            auto shearletTransform = downcast<ShearletTransform<data_t>>(A.getIthOperator(0));

            if (shearletTransform.getWidth() != shearletTransform.getHeight()) {
                throw std::invalid_argument(
                    "SHADMM::solveImpl: currently only solving square-shaped signals");
            }

            index_t L = shearletTransform.getL();
            index_t n = shearletTransform.getWidth();

            VolumeDescriptor volDescrOfn2({n * n});
            VolumeDescriptor volDescrOfLn2({L * n * n});
            VolumeDescriptor volDescrOfLp1n2({(L + 1) * n * n});

            /// f ∈ R ^ n^2, f is the same as x in the regular ADMM
            DataContainer<data_t> f(volDescrOfn2);
            /// set me to R_φTy, try 0 start as well
            f = 0; // dataTermResidual.getOperator().applyAdjoint(dataTermResidual.getDataVector());

            /// this means z ∈ R ^ (L+1)n^2
            DataContainer<data_t> z(volDescrOfLp1n2);
            z = 0;

            /// this means u ∈ R ^ (L+1)n^2
            DataContainer<data_t> u(volDescrOfLp1n2);
            u = 0;

            /// this means P1z ∈ R ^ Ln^2
            DataContainer<data_t> P1z(volDescrOfLn2);
            P1z = 0;

            /// this means P2z ∈ R ^ n^2
            DataContainer<data_t> P2z(volDescrOfn2);
            P2z = 0;

            /// this means P1u ∈ R ^ Ln^2
            DataContainer<data_t> P1u(volDescrOfLn2);
            P1u = 0;

            /// this means P2u ∈ R ^ n^2
            DataContainer<data_t> P2u(volDescrOfn2);
            P2u = 0;

            Logger::get("SHADMM")->info("{:*^20}|{:*^20}|{:*^20}|{:*^20}|{:*^20}|{:*^20}",
                                        "iteration", "fL2NormSq", "zL2NormSq", "uL2NormSq",
                                        "rkL2Norm", "skL2Norm");
            for (index_t iter = 0; iter < iterations; ++iter) {
                LinearResidual<data_t> fLinearResidual(A, c - B.apply(z) - u); // c is 0 in SHADMM
                RegularizationTerm fRegTerm(_rho / 2, L2NormPow2<data_t>(fLinearResidual));
                Problem<data_t> fUpdateProblem(dataTerm, fRegTerm, f);

                // TODO for SHADMM we can use CG here
                FSolver<data_t> fSolver(fUpdateProblem);
                f = fSolver.solve(_defaultFSolverIterations);

                DataContainer<data_t> rk = f;
                DataContainer<data_t> zPrev = z;
                data_t Afnorm = f.l2Norm();

                // ===== here starts very SHADMM specific code =====
                // TODO add vector of thresholds to SoftThresholding? use ProjectionOperator?

                /// first Ln^2 for P1, last n^2 for P2
                // TODO should f be called x in the merged ADMM? probably yes

                /// this means shrink ∈ R ^ Ln^2
                SoftThresholding<data_t> shrink(volDescrOfLn2);
                // w is pulled from the WeightedL1Norm
                // TODO w[i] instead of w[0]
                P1z =
                    shrink.apply(shearletTransform.apply(f.viewAs(VolumeDescriptor{{n, n}})) + P1u,
                                 geometry::Threshold(_rho0 * w[0] / _rho1));
                // TODO element-wise max?
                P2z = maxWithZero(f + P2u); // TODO is this ReLU-like functionality present already?
                // z is concatenating P1z and P2z? if not, what is it then?
                // TODO consider using BlockLinearOperator for the final ADMM

                /// P1u = P1u + SH.apply(f) - P1z; // not accounting for reshaping f
                P1u = P1u + shearletTransform.apply(f.viewAs(VolumeDescriptor{{n, n}})) - P1z;
                P2u = P2u + f - P2z;
                // u is concatenating P1u and P2u? if not, what is it then?
                // TODO consider using BlockLinearOperator for the final ADMM

                u = concatenate(P1u, P2u);
                z = concatenate(P1z, P2z);
                // ===== here ends very SHADMM specific code =====

                rk -= z;
                DataContainer<data_t> sk = zPrev - z;
                sk *= _rho;

                // TODO this might just work out of the box instead of the sliced parts as done
                //  above
                // u += A.apply(f) + B.apply(z) - c;

                DataContainer<data_t> Atu = u;
                Atu *= _rho;
                data_t rkL2Norm = rk.l2Norm();
                data_t skL2Norm = sk.l2Norm();

                Logger::get("SHADMM")->info("{:<19}| {:<19}| {:<19}| {:<19}| {:<19}| {:<19}", iter,
                                            f.squaredL2Norm(), z.squaredL2Norm(), u.squaredL2Norm(),
                                            rkL2Norm, skL2Norm);

                /// variables for the stopping criteria
                const data_t cL2Norm = !dataTermResidual.hasDataVector()
                                           ? static_cast<data_t>(0.0)
                                           : dataTermResidual.getDataVector().l2Norm();
                const data_t epsRelMax =
                    _epsilonRel * std::max(std::max(Afnorm, z.l2Norm()), cL2Norm);
                const auto epsilonPri = (std::sqrt(rk.getSize()) * _epsilonAbs) + epsRelMax;

                const data_t epsRelL2Norm = _epsilonRel * Atu.l2Norm();
                const auto epsilonDual = (std::sqrt(sk.getSize()) * _epsilonAbs) + epsRelL2Norm;

                if (rkL2Norm <= epsilonPri && skL2Norm <= epsilonDual) {
                    Logger::get("SHADMM")->info("SUCCESS: Reached convergence at {}/{} iterations ",
                                                iter, iterations);

                    getCurrentSolution() = f;
                    return getCurrentSolution();
                }

                // TODO varying penalty parameter was here, add again after development and check
            }

            Logger::get("SHADMM")->warn("Failed to reach convergence at {} iterations", iterations);

            getCurrentSolution() = f;
            return getCurrentSolution();
        }

    protected:
        /// lift the base class method getCurrentSolution
        using Solver<data_t>::getCurrentSolution;

        /// implement the polymorphic clone operation
        auto cloneImpl() const -> SHADMM<FSolver, data_t>* override
        {
            return new SHADMM<FSolver, data_t>(downcast<SplittingProblem<data_t>>(*_problem));
        }

    private:
        /// lift the base class variable _problem
        using Solver<data_t>::_problem;

        /// the default number of iterations for SHADMM
        bool _positiveSolutions{false};

        /// the default number of iterations for SHADMM
        index_t _defaultIterations{100};

        /// the default number of iterations for the FSolver
        index_t _defaultFSolverIterations{5};

        /// @f$ \rho @f$ values from the problem definition
        data_t _rho{1};
        data_t _rho0{1}; // consider as hyper-parameters
        data_t _rho1{1}; // consider as hyper-parameters
        data_t _rho2{1}; // just set it to 1, at least initially

        /// variables for the stopping criteria
        data_t _epsilonAbs{1e-5f};
        data_t _epsilonRel{1e-5f};
    };
} // namespace elsa
