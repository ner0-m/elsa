#include "CGNL.h"
#include "DataContainer.h"
#include "LinearOperator.h"
#include "TypeCasts.hpp"
#include "spdlog/stopwatch.h"
#include "Logger.h"

namespace elsa
{
    template <class data_t>
    CGNL<data_t>::CGNL(const Functional<data_t>& func, data_t eps_CG,
                       data_t eps_NR, index_t iterations_NR, index_t restart)
        : func_(func.clone()), eps_CG_(eps_CG), eps_NR_(eps_NR), iterations_NR_(iterations_NR),
          restart_(restart)
    {
    }

    template <class data_t>
    DataContainer<data_t> CGNL<data_t>::solve(index_t iterations_CG,
                                              std::optional<DataContainer<data_t>> x0)
    {
        spdlog::stopwatch aggregate_time;

        auto x = DataContainer<data_t>(func_->getDomainDescriptor());
        auto r = DataContainer<data_t>(func_->getDomainDescriptor());
        auto d = DataContainer<data_t>(func_->getDomainDescriptor());

        if (x0.has_value()) {
            x = *x0;
        }
        else {
            x = 0;
        }

        func_->getGradient(x, r);
        r *= static_cast<data_t>(-1.0);
        d = r;

        data_t delta_new = r.dot(r);
        data_t delta_zero = delta_new;

        Logger::get("CGNL")->info("{:^5} | {:^15} | {:^15} |", "Iters", "delta_new", "delta_zero");
        for (index_t iter_CG = 0; iter_CG < iterations_CG; ++iter_CG) {
            if (delta_new < eps_CG_ * eps_CG_ * delta_zero) {
                Logger::get("CGNL")->info("SUCCESS: Reached convergence at {}/{} iteration", iter_CG, iterations_CG);
                return x;
            }

            data_t delta_d = d.dot(d);
            for (index_t iter_NR = 0; iter_NR < iterations_NR_; ++iter_NR) {
                data_t alpha_numerator = static_cast<data_t>(-1.0) * func_->getGradient(x).dot(d);

                auto hessian = func_->getHessian(x);
                data_t alpha_denominator = hessian.apply(d).dot(d);

                auto alpha = alpha_numerator / alpha_denominator;

                x += alpha * d;

//                Logger::get("CGNL")->info("NR: alpha {}", alpha);
//                Logger::get("CGNL")->info("NR: alpha:nume {}", alpha_numerator);
//                Logger::get("CGNL")->info("NR: alpha:deno {}", alpha_denominator);
//                Logger::get("CGNL")->info("NR: x {}", x.sum());

                if (alpha * alpha * delta_d < eps_NR_ * eps_NR_) {
                    break;
                }
            }

            r = func_->getGradient(x);
            r *= static_cast<data_t>(-1.0);

            data_t delta_old = delta_new;
            delta_new = r.dot(r);
            auto beta = delta_new / delta_old;
            d = r + (beta * d);
            if (r.dot(d) <= 0 || (iter_CG + 1) % restart_ == 0) {
                d = r;
            }

//            Logger::get("CGNL")->info("beta {}", beta);
//            Logger::get("CGNL")->info("delta_new {}", delta_new);

            Logger::get("CGNL")->info("{:>5} | {:>15.10} | {:>15.10} |", iter_CG, std::sqrt(delta_new), std::sqrt(delta_zero));
        }

        Logger::get("CGNL")->warn("Failed to reach convergence at {} iterations", iterations_CG);
        return x;
    }

    template <class data_t>
    CGNL<data_t>* CGNL<data_t>::cloneImpl() const
    {
        return new CGNL(*func_, eps_CG_, eps_NR_, iterations_NR_, restart_);
    }

    template <class data_t>
    bool CGNL<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherCGNL = downcast_safe<CGNL>(&other);

        return otherCGNL && *(otherCGNL->func_) == *func_
               && otherCGNL->eps_CG_ == eps_CG_
               && otherCGNL->eps_NR_ == eps_NR_
               && otherCGNL->iterations_NR_ == iterations_NR_
               && otherCGNL->restart_ == restart_;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class CGNL<float>;
    template class CGNL<double>;

} // namespace elsa
