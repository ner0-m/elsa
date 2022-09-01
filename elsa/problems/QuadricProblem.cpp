#include "QuadricProblem.h"
#include "L2NormPow2.h"
#include "WeightedL2NormPow2.h"
#include "LinearOperator.h"
#include "Identity.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    QuadricProblem<data_t>::QuadricProblem(const LinearOperator<data_t>& A,
                                           const DataContainer<data_t>& b, bool spdA)
        : QuadricProblem<data_t>{spdA ? Quadric{A, b} : Quadric{adjoint(A) * A, A.applyAdjoint(b)}}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    QuadricProblem<data_t>::QuadricProblem(const Quadric<data_t>& quadric)
        : Problem<data_t>{quadric}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    QuadricProblem<data_t>::QuadricProblem(const Problem<data_t>& problem)
        : Problem<data_t>{*quadricFromProblem(problem)}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    QuadricProblem<data_t>* QuadricProblem<data_t>::cloneImpl() const
    {
        return new QuadricProblem(*this);
    }

    template <typename data_t>
    LinearResidual<data_t>
        QuadricProblem<data_t>::getGradientExpression(const RegularizationTerm<data_t>& regTerm)
    {
        const auto lambda = regTerm.getWeight();
        const Scaling<data_t> lambdaOp{regTerm.getFunctional().getDomainDescriptor(), lambda};

        if (is<L2NormPow2<data_t>>(regTerm.getFunctional())) {
            const auto& regFunc = downcast<L2NormPow2<data_t>>(regTerm.getFunctional());

            if (!is<LinearResidual<data_t>>(regFunc.getResidual())) {
                throw LogicError("QuadricProblem: cannot convert a non-linear regularization "
                                 "term to quadric form");
            }
            const auto& regTermResidual = downcast<LinearResidual<data_t>>(regFunc.getResidual());

            if (regTermResidual.hasOperator()) {
                const auto& regTermOp = regTermResidual.getOperator();
                LinearOperator<data_t> A = lambdaOp * adjoint(regTermOp) * regTermOp;

                if (regTermResidual.hasDataVector()) {
                    DataContainer<data_t> b =
                        lambda * regTermOp.applyAdjoint(regTermResidual.getDataVector());
                    return LinearResidual<data_t>{A, b};
                } else {
                    return LinearResidual<data_t>{A};
                }
            } else {
                if (regTermResidual.hasDataVector()) {
                    DataContainer<data_t> b = lambda * regTermResidual.getDataVector();
                    return LinearResidual<data_t>{lambdaOp, b};
                } else {
                    return LinearResidual<data_t>{lambdaOp};
                }
            }
        } else if (is<WeightedL2NormPow2<data_t>>(regTerm.getFunctional())) {
            const auto& regFunc = downcast<WeightedL2NormPow2<data_t>>(regTerm.getFunctional());

            if (!is<LinearResidual<data_t>>(regFunc.getResidual()))
                throw LogicError("QuadricProblem: cannot convert a non-linear regularization "
                                 "term to quadric form");

            const auto& regTermResidual = downcast<LinearResidual<data_t>>(regFunc.getResidual());

            const auto& W = regFunc.getWeightingOperator();
            std::unique_ptr<Scaling<data_t>> lambdaWPtr;

            if (W.isIsotropic())
                lambdaWPtr = std::make_unique<Scaling<data_t>>(W.getDomainDescriptor(),
                                                               lambda * W.getScaleFactor());
            else
                lambdaWPtr = std::make_unique<Scaling<data_t>>(W.getDomainDescriptor(),
                                                               lambda * W.getScaleFactors());

            const auto& lambdaW = *lambdaWPtr;

            if (regTermResidual.hasOperator()) {
                auto& regTermOp = regTermResidual.getOperator();
                LinearOperator<data_t> A = adjoint(regTermOp) * lambdaW * regTermOp;

                if (regTermResidual.hasDataVector()) {
                    DataContainer<data_t> b =
                        regTermOp.applyAdjoint(lambdaW.apply(regTermResidual.getDataVector()));
                    return LinearResidual<data_t>{A, b};
                } else {
                    return LinearResidual<data_t>{A};
                }
            } else {
                if (regTermResidual.hasDataVector()) {
                    DataContainer<data_t> b = lambdaW.apply(regTermResidual.getDataVector());
                    return LinearResidual<data_t>{lambdaW, b};
                } else {
                    return LinearResidual<data_t>{lambdaW};
                }
            }
        } else if (is<Quadric<data_t>>(regTerm.getFunctional())) {
            const auto& regFunc = downcast<Quadric<data_t>>(regTerm.getFunctional());
            const auto& quadricResidual = regFunc.getGradientExpression();
            if (quadricResidual.hasOperator()) {
                LinearOperator<data_t> A = lambdaOp * quadricResidual.getOperator();

                if (quadricResidual.hasDataVector()) {
                    const DataContainer<data_t>& b = quadricResidual.getDataVector();
                    return LinearResidual<data_t>{A, lambda * b};
                } else {
                    return LinearResidual<data_t>{A};
                }
            } else {
                if (quadricResidual.hasDataVector()) {
                    const DataContainer<data_t>& b = quadricResidual.getDataVector();
                    return LinearResidual<data_t>{lambdaOp, lambda * b};
                } else {
                    return LinearResidual<data_t>{lambdaOp};
                }
            }
        } else {
            throw InvalidArgumentError("QuadricProblem: regularization terms should be of type "
                                       "(Weighted)L2NormPow2 or Quadric");
        }
    }

    template <typename data_t>
    std::unique_ptr<Quadric<data_t>>
        QuadricProblem<data_t>::quadricFromProblem(const Problem<data_t>& problem)
    {
        const auto& functional = problem.getDataTerm();
        if (is<Quadric<data_t>>(functional) && problem.getRegularizationTerms().empty()) {
            return downcast<Quadric<data_t>>(functional.clone());
        } else {
            std::unique_ptr<LinearOperator<data_t>> dataTermOp;
            std::unique_ptr<DataContainer<data_t>> quadricVec;

            // convert data term
            if (is<Quadric<data_t>>(functional)) {
                const auto& trueFunctional = downcast<Quadric<data_t>>(functional);
                const LinearResidual<data_t>& residual = trueFunctional.getGradientExpression();

                if (residual.hasOperator()) {
                    dataTermOp = residual.getOperator().clone();
                } else {
                    dataTermOp = std::make_unique<Identity<data_t>>(residual.getDomainDescriptor());
                }

                if (residual.hasDataVector()) {
                    quadricVec = std::make_unique<DataContainer<data_t>>(residual.getDataVector());
                }
            } else if (is<L2NormPow2<data_t>>(functional)) {
                const auto& trueFunctional = downcast<L2NormPow2<data_t>>(functional);

                if (!is<LinearResidual<data_t>>(trueFunctional.getResidual()))
                    throw LogicError(
                        "QuadricProblem: cannot convert a non-linear term to quadric form");

                const auto& residual =
                    downcast<LinearResidual<data_t>>(trueFunctional.getResidual());

                if (residual.hasOperator()) {
                    const auto& A = residual.getOperator();
                    dataTermOp = std::make_unique<LinearOperator<data_t>>(adjoint(A) * A);

                    if (residual.hasDataVector()) {
                        quadricVec = std::make_unique<DataContainer<data_t>>(
                            A.applyAdjoint(residual.getDataVector()));
                    }
                } else {
                    dataTermOp = std::make_unique<Identity<data_t>>(residual.getDomainDescriptor());

                    if (residual.hasDataVector()) {
                        quadricVec =
                            std::make_unique<DataContainer<data_t>>(residual.getDataVector());
                    }
                }
            } else if (is<WeightedL2NormPow2<data_t>>(functional)) {
                const auto& trueFunctional = downcast<WeightedL2NormPow2<data_t>>(functional);

                if (!is<LinearResidual<data_t>>(trueFunctional.getResidual()))
                    throw LogicError(
                        "QuadricProblem: cannot convert a non-linear term to quadric form");
                const auto& residual =
                    downcast<LinearResidual<data_t>>(trueFunctional.getResidual());

                const auto& W = trueFunctional.getWeightingOperator();

                if (residual.hasOperator()) {
                    const auto& A = residual.getOperator();
                    dataTermOp = std::make_unique<LinearOperator<data_t>>(adjoint(A) * W * A);

                    if (residual.hasDataVector()) {
                        quadricVec = std::make_unique<DataContainer<data_t>>(
                            A.applyAdjoint(W.apply(residual.getDataVector())));
                    }
                } else {
                    dataTermOp = W.clone();

                    if (residual.hasDataVector()) {
                        quadricVec = std::make_unique<DataContainer<data_t>>(
                            W.apply(residual.getDataVector()));
                    }
                }
            } else {
                throw LogicError("QuadricProblem: can only convert functionals of type "
                                 "(Weighted)L2NormPow2 to Quadric");
            }

            if (problem.getRegularizationTerms().empty()) {
                if (!quadricVec) {
                    return std::make_unique<Quadric<data_t>>(*dataTermOp);
                } else {
                    return std::make_unique<Quadric<data_t>>(*dataTermOp, *quadricVec);
                }
            }

            // add regularization terms
            LinearOperator<data_t> quadricOp{dataTermOp->getDomainDescriptor(),
                                             dataTermOp->getRangeDescriptor()};

            for (std::size_t i = 0; i < problem.getRegularizationTerms().size(); i++) {
                const auto& regTerm = problem.getRegularizationTerms()[i];
                LinearResidual<data_t> residual = getGradientExpression(regTerm);

                if (i == 0)
                    quadricOp = (*dataTermOp + residual.getOperator());
                else
                    quadricOp = quadricOp + residual.getOperator();

                if (residual.hasDataVector()) {
                    if (!quadricVec)
                        quadricVec =
                            std::make_unique<DataContainer<data_t>>(residual.getDataVector());
                    else
                        *quadricVec += residual.getDataVector();
                }
            }

            if (!quadricVec) {
                return std::make_unique<Quadric<data_t>>(quadricOp);
            } else {
                return std::make_unique<Quadric<data_t>>(quadricOp, *quadricVec);
            }
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class QuadricProblem<float>;
    template class QuadricProblem<double>;

} // namespace elsa
