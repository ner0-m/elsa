#include "MLEM.h"
#include "OSMLEM.h"

#include "spdlog/stopwatch.h"

#include "DataContainer.h"
#include "Logger.h"

namespace elsa
{
    template <class data_t>
    DataContainer<data_t> MLEM<data_t>::solve(index_t iterations,
                                              std::optional<DataContainer<data_t>> x0)
    {
        spdlog::stopwatch aggregate_time;

        const auto& range = op_->getRangeDescriptor();
        const auto& domain = op_->getDomainDescriptor();

        auto x = DataContainer<data_t>(domain);
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 1;
        }

        auto tmpRange = DataContainer<data_t>(range);
        tmpRange = 1;

        auto sensitivity = maximum(op_->applyAdjoint(tmpRange), eps_);

        auto tmpDomain = DataContainer<data_t>(domain);

        Logger::get("MLEM")->info("| {:^6} | {:^8} | {:^8} |", "iter", "time", "elapsed");
        for (int iter = 0; iter < iterations; ++iter) {
            spdlog::stopwatch iter_time;

            detail::mlemStep(*op_, data_, x, tmpRange, tmpDomain, sensitivity, eps_);

            Logger::get("MLEM")->info("| {:>6} | {:>8.3} | {:>7.3}s |", iter, iter_time,
                                      aggregate_time);
        }

        return x;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class MLEM<float>;
    template class MLEM<double>;
} // namespace elsa
