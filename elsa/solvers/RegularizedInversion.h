#pragma once

#include "elsaDefines.h"
#include "LinearOperator.h"
#include "DataContainer.h"

#include <optional>
#include <variant>
#include <vector>

namespace elsa
{
    /**
     * @brief Regularized inversion. This solves a system of equations of a
     * stacked operator
     */
    template <class data_t>
    DataContainer<data_t>
        reguarlizedInversion(const LinearOperator<data_t>& op, const DataContainer<data_t>& b,
                             const std::vector<std::unique_ptr<LinearOperator<data_t>>>& regOps,
                             const std::vector<DataContainer<data_t>>& regData,
                             std::variant<data_t, std::vector<data_t>> lambda, index_t niters = 5,
                             std::optional<DataContainer<data_t>> x0 = std::nullopt);

    template <class data_t>
    DataContainer<data_t>
        reguarlizedInversion(const LinearOperator<data_t>& op, const DataContainer<data_t>& b,
                             const LinearOperator<data_t>& regOp,
                             const DataContainer<data_t>& regData, SelfType_t<data_t> lambda,
                             index_t niters = 5,
                             std::optional<DataContainer<data_t>> x0 = std::nullopt)
    {
        std::vector<std::unique_ptr<LinearOperator<data_t>>> regOps;
        regOps.emplace_back(regOp.clone());

        std::vector<DataContainer<data_t>> regDatas;
        regDatas.emplace_back(regData);

        return reguarlizedInversion<data_t>(op, b, regOps, regDatas, lambda, niters, x0);
    }
} // namespace elsa
