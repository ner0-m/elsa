#include "CGLS.h"
#include "DataContainer.h"
#include "Error.h"
#include "LinearOperator.h"
#include "LinearResidual.h"
#include "TypeCasts.hpp"
#include "spdlog/stopwatch.h"
#include "Logger.h"

namespace elsa
{
    template <class data_t>
    CGLS<data_t>::CGLS(const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                       SelfType_t<data_t> eps, SelfType_t<data_t> tol)
        : A_(A.clone()), b_(b), damp_(eps * eps), tol_(tol)
    {
    }

    template <class data_t>
    DataContainer<data_t> CGLS<data_t>::solve(index_t iterations,
                                              std::optional<DataContainer<data_t>> x0)
    {
        spdlog::stopwatch aggregate_time;

        auto x = DataContainer<data_t>(A_->getDomainDescriptor());
        auto r = DataContainer<data_t>(A_->getDomainDescriptor());
        auto s = DataContainer<data_t>(A_->getRangeDescriptor());

        if (x0.has_value()) {
            x = *x0;

            // s = b_ - A_->applx(x), but without temporary allocating memory
            A_->apply(x, s);
            s *= -1;
            s += b_;

            A_->applyAdjoint(s, r);
            r -= damp_ * x;
        } else {
            x = 0;
            s = b_;
            A_->applyAdjoint(s, r);
        }

        auto c = r;
        auto q = DataContainer<data_t>(b_.getDataDescriptor());

        auto k = r.squaredL2Norm();
        auto kold = k;

        Logger::get("CGLS")->info("{:^5} | {:^15} |", "Iters", "|| s ||_2");
        for (int iter = 0; iter < iterations; ++iter) {
            if (kold < tol_) {
                Logger::get("CGLS")->info("SUCCESS: Reached convergence at {}/{} iteration", iter,
                                          iterations);
                return x;
            }

            A_->apply(c, q);

            auto delta = q.squaredL2Norm();
            auto alpha = [&]() {
                if (damp_ == 0) {
                    return kold / delta;
                } else {
                    return kold / (delta + damp_ * c.squaredL2Norm());
                }
            }();

            x += alpha * c;
            s -= alpha * q;

            A_->applyAdjoint(s, r);
            if (damp_ != 0.0) {
                r -= damp_ * x;
            }

            k = r.squaredL2Norm();
            auto beta = k / kold;

            c = r + beta * c;
            kold = k;

            Logger::get("CGLS")->info("{:>5} | {:>15.10} |", iter, s.l2Norm());
        }

        Logger::get("CGLS")->warn("Failed to reach convergence at {} iterations", iterations);

        return x;
    }
    template <class data_t>
    CGLS<data_t>* CGLS<data_t>::cloneImpl() const
    {
        return new CGLS(*A_, b_, std::sqrt(damp_));
    }

    template <class data_t>
    bool CGLS<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherCGLS = downcast_safe<CGLS>(&other);

        return otherCGLS && *otherCGLS->A_ == *A_ && otherCGLS->b_ == b_
               && otherCGLS->damp_ == damp_;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class CGLS<float>;
    template class CGLS<double>;

} // namespace elsa
