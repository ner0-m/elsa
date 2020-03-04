#include "WLSProblem.h"
#include "L2NormPow2.h"
#include "WeightedL2NormPow2.h"
#include "RandomBlocksDescriptor.h"
#include "BlockLinearOperator.h"
#include "Identity.h"

namespace elsa
{
    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const Scaling<data_t>& W, const LinearOperator<data_t>& A,
                                   const DataContainer<data_t>& b, const DataContainer<data_t>& x0)
        : Problem<data_t>{WeightedL2NormPow2<data_t>{LinearResidual<data_t>{A, b}, W}, x0}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const Scaling<data_t>& W, const LinearOperator<data_t>& A,
                                   const DataContainer<data_t>& b)
        : Problem<data_t>{WeightedL2NormPow2<data_t>{LinearResidual<data_t>{A, b}, W}}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                                   const DataContainer<data_t>& x0)
        : Problem<data_t>{L2NormPow2<data_t>{LinearResidual<data_t>{A, b}}, x0}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const LinearOperator<data_t>& A, const DataContainer<data_t>& b)
        : Problem<data_t>{L2NormPow2<data_t>{LinearResidual<data_t>{A, b}}}
    {
        // sanity checks are done in the member constructors already
    }

    template <typename data_t>
    WLSProblem<data_t>::WLSProblem(const Problem<data_t>& problem)
        : Problem<data_t>{*wlsFromProblem(problem), problem.getCurrentSolution()}
    {
    }

    template <typename data_t>
    WLSProblem<data_t>* WLSProblem<data_t>::cloneImpl() const
    {
        return new WLSProblem(*this);
    }

    template <typename data_t>
    std::unique_ptr<Functional<data_t>>
        WLSProblem<data_t>::wlsFromProblem(const Problem<data_t>& problem)
    {
        const auto& dataTerm = problem.getDataTerm();
        const auto& regTerms = problem.getRegularizationTerms();

        if (!dynamic_cast<const WeightedL2NormPow2<data_t>*>(&dataTerm)
            && !dynamic_cast<const L2NormPow2<data_t>*>(&dataTerm))
            throw std::logic_error("WLSProblem: conversion failed - data term is not "
                                   "of type (Weighted)L2NormPow2");

        const auto dataTermResidual =
            dynamic_cast<const LinearResidual<data_t>*>(&dataTerm.getResidual());

        if (!dataTermResidual)
            throw std::logic_error("WLSProblem: conversion failed - data term is non-linear");

        // data term is of convertible type

        // no conversion needed if no regTerms
        if (regTerms.empty())
            return dataTerm.clone();

        // else regTerms present, determine data vector descriptor
        std::vector<std::unique_ptr<DataDescriptor>> rangeDescList(0);
        rangeDescList.push_back(dataTermResidual->getRangeDescriptor().clone());
        for (const auto& regTerm : regTerms) {
            if (!dynamic_cast<const WeightedL2NormPow2<data_t>*>(&regTerm.getFunctional())
                && !dynamic_cast<const L2NormPow2<data_t>*>(&regTerm.getFunctional()))
                throw std::logic_error("WLSProblem: conversion failed - regularization term is not "
                                       "of type (Weighted)L2NormPow2");

            const auto regTermResidual =
                dynamic_cast<const LinearResidual<data_t>*>(&regTerm.getFunctional().getResidual());

            if (!regTermResidual)
                throw std::logic_error(
                    "WLSProblem: conversion failed - regularization term is non-linear");

            rangeDescList.push_back(regTermResidual->getRangeDescriptor().clone());
        }

        // problem is convertible, allocate memory, set to zero and start building block op
        RandomBlocksDescriptor dataVecDesc{rangeDescList};
        DataContainer<data_t> dataVec{dataVecDesc};
        dataVec = 0;
        std::vector<std::unique_ptr<LinearOperator<data_t>>> opList(0);

        // add block corresponding to data term
        if (const auto trueFunc = dynamic_cast<const WeightedL2NormPow2<data_t>*>(&dataTerm)) {
            const auto& scaling = trueFunc->getWeightingOperator();
            const auto& desc = scaling.getDomainDescriptor();

            std::unique_ptr<Scaling<data_t>> sqrtW{};
            if (scaling.isIsotropic()) {
                auto fac = scaling.getScaleFactor();

                if constexpr (std::is_floating_point_v<data_t>) {
                    if (fac < 0) {
                        throw std::logic_error("WLSProblem: conversion failed - negative weighting "
                                               "factor in WeightedL2NormPow2 term");
                    }
                }

                sqrtW = std::make_unique<Scaling<data_t>>(desc, std::sqrt(fac));
            } else {
                const auto& fac = scaling.getScaleFactors();

                if constexpr (std::is_floating_point_v<data_t>) {
                    for (const auto& w : fac) {
                        if (w < 0) {
                            throw std::logic_error(
                                "WLSProblem: conversion failed - negative weighting "
                                "factor in WeightedL2NormPow2 term");
                        }
                    }
                }

                sqrtW = std::make_unique<Scaling<data_t>>(desc, sqrt(fac));
            }

            if (dataTermResidual->hasDataVector())
                dataVec.getBlock(0) = sqrtW->apply(dataTermResidual->getDataVector());

            if (dataTermResidual->hasOperator()) {
                const auto composite = *sqrtW * dataTermResidual->getOperator();
                opList.emplace_back(composite.clone());
            } else {
                opList.push_back(std::move(sqrtW));
            }

        } else {
            if (dataTermResidual->hasOperator()) {
                opList.push_back(dataTermResidual->getOperator().clone());
            } else {
                opList.push_back(
                    std::make_unique<Identity<data_t>>(dataTermResidual->getDomainDescriptor()));
            }

            if (dataTermResidual->hasDataVector())
                dataVec.getBlock(0) = dataTermResidual->getDataVector();
        }

        // add blocks corresponding to reg terms
        index_t blockNum = 1;
        for (const auto& regTerm : regTerms) {
            const data_t lambda = regTerm.getWeight();
            const auto& func = regTerm.getFunctional();
            const auto residual = static_cast<const LinearResidual<data_t>*>(&func.getResidual());

            if (const auto trueFunc = dynamic_cast<const WeightedL2NormPow2<data_t>*>(&func)) {
                const auto& scaling = trueFunc->getWeightingOperator();
                const auto& desc = scaling.getDomainDescriptor();

                std::unique_ptr<Scaling<data_t>> sqrtLambdaW{};
                if (scaling.isIsotropic()) {
                    auto fac = scaling.getScaleFactor();

                    if constexpr (std::is_floating_point_v<data_t>) {
                        if (lambda * fac < 0) {
                            throw std::logic_error(
                                "WLSProblem: conversion failed - negative weighting "
                                "factor in WeightedL2NormPow2 term");
                        }
                    }

                    sqrtLambdaW = std::make_unique<Scaling<data_t>>(desc, std::sqrt(lambda * fac));
                } else {
                    const auto& fac = scaling.getScaleFactors();

                    if constexpr (std::is_floating_point_v<data_t>) {
                        for (const auto& w : fac) {
                            if (lambda * w < 0) {
                                throw std::logic_error(
                                    "WLSProblem: conversion failed - negative weighting "
                                    "factor in WeightedL2NormPow2 term");
                            }
                        }
                    }

                    sqrtLambdaW = std::make_unique<Scaling<data_t>>(desc, sqrt(lambda * fac));
                }

                if (residual->hasDataVector())
                    dataVec.getBlock(blockNum) = sqrtLambdaW->apply(residual->getDataVector());

                if (residual->hasOperator()) {
                    const auto composite = *sqrtLambdaW * residual->getOperator();
                    opList.push_back(composite.clone());
                } else {
                    opList.push_back(std::move(sqrtLambdaW));
                }

            } else {
                if constexpr (std::is_floating_point_v<data_t>) {
                    if (lambda < 0) {
                        throw std::logic_error(
                            "WLSProblem: conversion failed - negative regularization term weight");
                    }
                }

                auto sqrtLambdaScaling = std::make_unique<Scaling<data_t>>(
                    residual->getRangeDescriptor(), std::sqrt(lambda));

                if (residual->hasDataVector())
                    dataVec.getBlock(blockNum) = std::sqrt(lambda) * residual->getDataVector();

                if (residual->hasOperator()) {
                    const auto composite = *sqrtLambdaScaling * residual->getOperator();
                    opList.emplace_back(composite.clone());
                } else {
                    opList.push_back(std::move(sqrtLambdaScaling));
                }
            }

            blockNum++;
        }

        BlockLinearOperator<data_t> blockOp{opList, BlockLinearOperator<data_t>::BlockType::ROW};

        return std::make_unique<L2NormPow2<data_t>>(LinearResidual<data_t>{blockOp, dataVec});
    }

    // ------------------------------------------
    // explicit template instantiation
    template class WLSProblem<float>;
    template class WLSProblem<double>;
    template class WLSProblem<std::complex<float>>;
    template class WLSProblem<std::complex<double>>;

} // namespace elsa
