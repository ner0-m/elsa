#pragma once

#include "Solver.h"
#include "LinearOperator.h"
#include "ProximalOperator.h"
#include "DataContainer.h"

namespace elsa
{

    template <class data_t>
    class TGV_LADMM final : public Solver<data_t>
    {
    public:

        TGV_LADMM(const LinearOperator<data_t>& A, const DataContainer<data_t>& b);

        /// default destructor
        ~TGV_LADMM() override = default;

        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

        TGV_LADMM<data_t>* cloneImpl() const override;

        bool isEqual(const Solver<data_t>& other) const override;

    private:

        std::unique_ptr<LinearOperator<data_t>> A_;

        DataContainer<data_t> b_;

    };
} // namespace elsa
