#include "RegularizedInversion.h"
#include "BlockLinearOperator.h"
#include "RandomBlocksDescriptor.h"
#include "CGNE.h"
#include "Solver.h"

#include <variant>
#include <vector>

namespace elsa
{
    template <class data_t>
    DataContainer<data_t>
        reguarlizedInversion(const LinearOperator<data_t>& op, const DataContainer<data_t>& b,
                             const std::vector<std::unique_ptr<LinearOperator<data_t>>>& regOps,
                             const std::vector<DataContainer<data_t>>& regData,
                             std::variant<data_t, std::vector<data_t>> lambda, index_t niters,
                             std::optional<DataContainer<data_t>> x0)
    {
        index_t size = 1 + asSigned(regOps.size());

        auto x = extract_or(x0, op.getDomainDescriptor());

        // Setup a block problem, where K = [Op; regOps..], and w = [b; c - Bz - u]
        std::vector<std::unique_ptr<DataDescriptor>> descs;
        descs.emplace_back(b.getDataDescriptor().clone());
        for (size_t i = 0; i < regData.size(); ++i) {
            descs.emplace_back(regData[i].getDataDescriptor().clone());
        }
        RandomBlocksDescriptor blockDesc(descs);

        std::vector<std::unique_ptr<LinearOperator<data_t>>> opList;
        opList.reserve(size);

        opList.emplace_back(op.clone());

        for (size_t i = 0; i < regOps.size(); ++i) {
            auto& regOp = *regOps[i];

            auto regParam = [&]() {
                if (std::holds_alternative<data_t>(lambda)) {
                    return std::get<data_t>(lambda);
                } else {
                    return std::get<std::vector<data_t>>(lambda)[i];
                }
            }();
            opList.emplace_back((regParam * regOp).clone());
        }

        BlockLinearOperator K(op.getDomainDescriptor(), blockDesc, opList,
                              BlockLinearOperator<data_t>::BlockType::ROW);

        DataContainer<data_t> w(blockDesc);
        w.getBlock(0) = b;

        for (index_t i = 1; i < size; ++i) {
            w.getBlock(i) = regData[i - 1];
        }

        CGNE<data_t> cg(K, w);
        return cg.solve(niters, x);
    }

    template DataContainer<float> reguarlizedInversion<float>(
        const LinearOperator<float>& op, const DataContainer<float>& b,
        const std::vector<std::unique_ptr<LinearOperator<float>>>& regOps,
        const std::vector<DataContainer<float>>& regData,
        std::variant<float, std::vector<float>> lambda, index_t niters,
        std::optional<DataContainer<float>> x0);

    template DataContainer<double>
        reguarlizedInversion(const LinearOperator<double>& op, const DataContainer<double>& b,
                             const std::vector<std::unique_ptr<LinearOperator<double>>>& regOps,
                             const std::vector<DataContainer<double>>& regData,
                             std::variant<double, std::vector<double>> lambda, index_t niters,
                             std::optional<DataContainer<double>> x0);
} // namespace elsa
