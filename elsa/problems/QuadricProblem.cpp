#include "QuadricProblem.h"
#include "L2NormPow2.h"
#include "WeightedL2NormPow2.h"
#include "LinearOperator.h"
#include "Identity.h"

namespace elsa {

    template <typename data_t>
    QuadricProblem<data_t>::QuadricProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                                           const DataContainer<data_t>& x0, bool spdA)
     : QuadricProblem<data_t>{spdA ? Quadric{A,b} : Quadric{adjoint(A)*A, A.applyAdjoint(b)}, x0}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    QuadricProblem<data_t>::QuadricProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                                           bool spdA)
     : QuadricProblem<data_t>{spdA ? Quadric{A,b} : Quadric{adjoint(A)*A, A.applyAdjoint(b)}}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    QuadricProblem<data_t>::QuadricProblem(const Quadric<data_t>& quadric, const DataContainer<data_t>& x0)
     : Problem<data_t>{quadric, x0}
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
    QuadricProblem<data_t>::QuadricProblem(const Problem<data_t>& optimizationProblem)
     : Problem<data_t>{*quadricFromProblem(optimizationProblem), optimizationProblem.getCurrentSolution()}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    QuadricProblem<data_t>* QuadricProblem<data_t>::cloneImpl() const
    {
        return new QuadricProblem(*this);
    }

    template <typename data_t>
    bool QuadricProblem<data_t>::isEqual(const Problem<data_t>& other) const
    {
        if (!Problem<data_t>::isEqual(other))
            return false;

        auto otherQP = dynamic_cast<const QuadricProblem*>(&other);
        if (!otherQP)
            return false;

        return true;
    }

    template <typename data_t>
    LinearResidual<data_t> QuadricProblem<data_t>::getGradientExpression(const RegularizationTerm<data_t>& regTerm)
    {
        const auto lambda = regTerm.getWeight();
        const Scaling<data_t> lambdaOp{regTerm.getFunctional().getDomainDescriptor(), lambda};

        if (auto regFuncPtr = dynamic_cast<const L2NormPow2<data_t>*>(&regTerm.getFunctional())) {
            
            const auto regTermResidualPtr = dynamic_cast<const LinearResidual<data_t>*>(&regFuncPtr->getResidual());
            if(!regTermResidualPtr) 
                throw std::logic_error("QuadricProblem: cannot convert a non-linear regularization term to quadric form");            
            
            if (regTermResidualPtr->hasOperator()) {
                const auto& regTermOp = regTermResidualPtr->getOperator();
                LinearOperator<data_t> A = lambdaOp * adjoint(regTermOp) * regTermOp;

                if (regTermResidualPtr->hasDataVector()) {
                    DataContainer<data_t> b = lambda * regTermOp.applyAdjoint(regTermResidualPtr->getDataVector());
                    return LinearResidual<data_t>{A, b};
                }
                else {
                    return LinearResidual<data_t>{A};
                }
            }
            else {
                if (regTermResidualPtr->hasDataVector()) {
                    DataContainer<data_t> b = lambda * regTermResidualPtr->getDataVector();
                    return LinearResidual<data_t>{lambdaOp, b};
                }
                else {
                    return LinearResidual<data_t>{lambdaOp};
                }
            }
        }
        else if (auto regFuncPtr = dynamic_cast<const WeightedL2NormPow2<data_t>*>(&regTerm.getFunctional())) {

            const auto regTermResidualPtr = dynamic_cast<const LinearResidual<data_t>*>(&regFuncPtr->getResidual());
            if(!regTermResidualPtr) 
                throw std::logic_error("QuadricProblem: cannot convert a non-linear regularization term to quadric form");
            
            const auto& W = regFuncPtr->getWeightingOperator();
            std::unique_ptr<Scaling<data_t>> lambdaWPtr;
            
            if (W.isIsotropic())
                lambdaWPtr = std::make_unique<Scaling<data_t>>(W.getDomainDescriptor(), lambda*W.getScaleFactor());
            else
                lambdaWPtr = std::make_unique<Scaling<data_t>>(W.getDomainDescriptor(), lambda*W.getScaleFactor());
            
            const auto& lambdaW = *lambdaWPtr;

            if (regTermResidualPtr->hasOperator()) {
                auto& regTermOp = regTermResidualPtr->getOperator();
                LinearOperator<data_t> A = adjoint(regTermOp) * lambdaW * regTermOp;

                if (regTermResidualPtr->hasDataVector()) {
                    DataContainer<data_t> b = regTermOp.applyAdjoint(lambdaW.apply(regTermResidualPtr->getDataVector()));
                    return LinearResidual<data_t>{A, b};
                }
                else {
                    return LinearResidual<data_t>{A};
                }
            }
            else {
                if (regTermResidualPtr->hasDataVector()) {
                    DataContainer<data_t> b = lambdaW.apply(regTermResidualPtr->getDataVector());
                    return LinearResidual<data_t>{lambdaW,b};
                }
                else {
                    return LinearResidual<data_t>{lambdaW};
                }
            }             
        }
        else if (auto regFuncPtr = dynamic_cast<const Quadric<data_t>*>(&regTerm.getFunctional())) {
            const auto& quadricResidual = regFuncPtr->getGradientExpression();
            if (quadricResidual.hasOperator()) {
                LinearOperator<data_t> A = lambdaOp*quadricResidual.getOperator();

                if (quadricResidual.hasDataVector()) {
                    const DataContainer<data_t>& b = quadricResidual.getDataVector();
                    return LinearResidual<data_t>{A,lambda*b}; 
                }
                else {
                    return LinearResidual<data_t>{A};
                }
            }
            else {
                if (quadricResidual.hasDataVector()) {
                    const DataContainer<data_t>& b = quadricResidual.getDataVector();
                    return LinearResidual<data_t>{lambdaOp,lambda*b}; 
                }
                else {
                    return LinearResidual<data_t>{lambdaOp};
                }
            }
        }
        else {
            throw std::invalid_argument("QuadricProblem: regularization terms should be of type (Weighted)L2NormPow2 or Quadric");
        }
    }

    template <typename data_t>
    std::unique_ptr<Quadric<data_t>> QuadricProblem<data_t>::quadricFromProblem(const Problem<data_t>& problem)
    {
        const auto& functional = problem.getDataTerm();
        if (const auto trueFunctionalPtr = dynamic_cast<const Quadric<data_t>*>(&functional) && problem.getRegularizationTerms().empty()) {
            return std::unique_ptr<Quadric<data_t>>{static_cast<Quadric<data_t>*>(functional.clone().release())};
        }
        else {
            std::unique_ptr<LinearOperator<data_t>> quadricOp;
            std::unique_ptr<DataContainer<data_t>> quadricVec;

            // convert data term
            if (const auto trueFunctionalPtr = dynamic_cast<const Quadric<data_t>*>(&functional)) {
                const LinearResidual<data_t>& residual = trueFunctionalPtr->getGradientExpression();

                if (residual.hasOperator()) {
                    quadricOp = std::make_unique<LinearOperator<data_t>>(residual.getOperator());
                }
                else {
                    quadricOp = std::make_unique<Identity<data_t>>(residual.getDomainDescriptor());
                }

                if (residual.hasDataVector()) {
                    quadricVec = std::make_unique<DataContainer<data_t>>(residual.getDataVector());
                }
            }

            if (const auto trueFunctionalPtr = dynamic_cast<const L2NormPow2<data_t>*>(&functional)) {
                const auto residualPtr = dynamic_cast<const LinearResidual<data_t>*>(&trueFunctionalPtr->getResidual());
                if(!residualPtr) 
                    throw std::logic_error("QuadricProblem: cannot convert a non-linear term to quadric form");

                if (residualPtr->hasOperator()) {
                    const auto& A = residualPtr->getOperator();
                    quadricOp = std::make_unique<LinearOperator<data_t>>(adjoint(A)*A);

                    if (residualPtr->hasDataVector()) {
                        quadricVec = std::make_unique<DataContainer<data_t>>(A.applyAdjoint(residualPtr->getDataVector()));
                    }
                }
                else {
                    quadricOp = std::make_unique<Identity<data_t>>(residualPtr->getDomainDescriptor());

                    if (residualPtr->hasDataVector()) {
                        quadricVec = std::make_unique<DataContainer<data_t>>(residualPtr->getDataVector());
                    }
                }
            }
            else if (const auto trueFunctionalPtr = dynamic_cast<const WeightedL2NormPow2<data_t>*>(&functional)) {
                const auto residualPtr = dynamic_cast<const LinearResidual<data_t>*>(&trueFunctionalPtr->getResidual());
                if(!residualPtr) 
                    throw std::logic_error("QuadricProblem: cannot convert a non-linear term to quadric form");

                const auto& W = trueFunctionalPtr->getWeightingOperator();

                if (residualPtr->hasOperator()) {
                    const auto& A = residualPtr->getOperator();
                    quadricOp = std::make_unique<LinearOperator<data_t>>(adjoint(A)*W*A);

                    if (residualPtr->hasDataVector()) {
                        quadricVec = std::make_unique<DataContainer<data_t>>(A.applyAdjoint(W.apply(residualPtr->getDataVector())));
                    }
                }
                else {
                    quadricOp = W.clone();

                    if (residualPtr->hasDataVector()) {
                        quadricVec = std::make_unique<DataContainer<data_t>>(W.apply(residualPtr->getDataVector()));
                    }
                }
            }
            else {
                throw std::logic_error("Quadric problem: can only convert functionals of type (Weighted)L2NormPow2 to Quadric"); 
            }

            // add regularization terms
            for (const RegularizationTerm<data_t>& regTerm: problem.getRegularizationTerms()) {
                LinearResidual<data_t> residual = getGradientExpression(regTerm);
                *quadricOp = *quadricOp + residual.getOperator();

                if (residual.hasDataVector()) {
                    if(!quadricVec)
                        quadricVec = std::make_unique<DataContainer<data_t>>(residual.getDataVector());
                    else 
                        *quadricVec += residual.getDataVector();
                }
            }

            if (!quadricVec) {
                return std::make_unique<Quadric<data_t>>(*quadricOp);
            }
            else {
                return std::make_unique<Quadric<data_t>>(*quadricOp,*quadricVec);
            }
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class QuadricProblem<float>;
    template class QuadricProblem<double>;

} //namespace elsa