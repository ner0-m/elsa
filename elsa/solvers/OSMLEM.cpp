#include "OSMLEM.h"

#include "LinearOperator.h"
#include "elsaDefines.h"
#include "spdlog/stopwatch.h"

#include "DataContainer.h"
#include "Logger.h"

namespace elsa
{
    namespace detail
    {
        template <class data_t>
        void mlemStep(const LinearOperator<data_t>& op, const DataContainer<data_t>& data,
                      DataContainer<data_t>& x, DataContainer<data_t>& range,
                      DataContainer<data_t>& domain, const DataContainer<data_t>& sensitivity,
                      SelfType_t<data_t> eps)
        {
            op.apply(x, range);

            range = maximum(range, eps);
            range = data / range;

            op.applyAdjoint(range, domain);
            domain /= sensitivity;

            x *= domain;
        }
    } // namespace detail

    template <class data_t>
    DataContainer<data_t> OSMLEM<data_t>::solve(index_t iterations,
                                                std::optional<DataContainer<data_t>> x0)
    {
        spdlog::stopwatch aggregate_time;

        const auto nSubsets = ops_.getSize();
        const auto& domain = ops_[0].getDomainDescriptor();

        auto x = DataContainer<data_t>(domain);
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 1;
        }

        std::vector<DataContainer<data_t>> tmpRange;
        for (int subset = 0; subset < nSubsets; ++subset) {
            auto ones = DataContainer<data_t>(ops_[subset].getRangeDescriptor());
            ones = 1;
            tmpRange.emplace_back(ones);
        }

        std::vector<DataContainer<data_t>> sensitivities;
        for (int subset = 0; subset < nSubsets; ++subset) {
            auto sensitivity = maximum(ops_[subset].applyAdjoint(tmpRange[subset]), eps_);
            sensitivity = 1;
            sensitivities.emplace_back(std::move(sensitivity));
        }

        auto tmpDomain = DataContainer<data_t>(domain);

        Logger::get("OSMLEM")->info("| {:^6} | {:^8} | {:^8} |", "iter", "time", "elapsed");
        for (int iter = 0; iter < iterations; ++iter) {
            spdlog::stopwatch iter_time;

            for (int subset = 0; subset < ops_.getSize(); ++subset) {
                detail::mlemStep(ops_[subset], data_[subset], x, tmpRange[subset], tmpDomain,
                                 sensitivities[subset], eps_);
            }

            Logger::get("OSMLEM")->info("| {:>6} | {:>8.3} | {:>7.3}s |", iter, iter_time,
                                        aggregate_time);
        }

        return x;
    }

    template <class data_t>
    OSMLEM<data_t>* OSMLEM<data_t>::cloneImpl() const
    {
        return new OSMLEM(ops_, data_, eps_);
    }

    template <class data_t>
    bool OSMLEM<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherMLEM = downcast_safe<OSMLEM>(&other);
        if (!otherMLEM)
            return false;

        if (!std::equal(ops_.begin(), ops_.end(), otherMLEM->ops_.begin()))
            return false;

        if (!std::equal(data_.begin(), data_.end(), otherMLEM->data_.begin()))
            return false;

        if (eps_ != otherMLEM->eps_)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class OSMLEM<float>;
    template class OSMLEM<double>;

    namespace detail
    {
        template void mlemStep(const LinearOperator<float>& op, const DataContainer<float>& data,
                               DataContainer<float>& x, DataContainer<float>& range,
                               DataContainer<float>& domain,
                               const DataContainer<float>& sensitivity, SelfType_t<float> eps);

        template void mlemStep(const LinearOperator<double>& op, const DataContainer<double>& data,
                               DataContainer<double>& x, DataContainer<double>& range,
                               DataContainer<double>& domain,
                               const DataContainer<double>& sensitivity, SelfType_t<double> eps);
    } // namespace detail
} // namespace elsa
