#include "CGNL.h"
#include "Logger.h"
#include "TypeCasts.hpp"
#include "spdlog/stopwatch.h"

namespace elsa
{

    template <typename data_t>
    CGNL<data_t>::CGNL(const Functional<data_t>& functional, data_t epsilon,
                       index_t line_search_iterations,
                       const LineSearchFunction& line_search_function,
                       const BetaFunction& beta_function)
        : Solver<data_t>(),
          functional_{functional.clone()},
          epsilon_{epsilon},
          line_search_iterations_{line_search_iterations},
          line_search_function_{line_search_function},
          beta_function_{beta_function}
    {
    }

    template <typename data_t>
    CGNL<data_t>::CGNL(const Functional<data_t>& functional, data_t epsilon,
                       index_t line_search_iterations)
        : CGNL<data_t>(functional, epsilon, line_search_iterations, lineSearchNewtonRaphson,
                       betaPolakRibiere)
    {
    }

    template <typename data_t>
    CGNL<data_t>::CGNL(const Functional<data_t>& functional, data_t epsilon)
        : CGNL<data_t>(functional, epsilon, 1, lineSearchConstantStepSize, betaPolakRibiere)
    {
    }

    template <typename data_t>
    DataContainer<data_t> CGNL<data_t>::solve(index_t iterations,
                                              std::optional<DataContainer<data_t>> xStart)
    {
        spdlog::stopwatch aggregate_time;
        Logger::get("CGNL")->info("Start preparations...");

        // Restart Nonlinear CG every n iterations, with n being the number of dimensions
        index_t nIterationMax = functional_->getDomainDescriptor().getNumberOfDimensions();

        // use xStart as start point if provided, use 0 otherwise
        DataContainer<data_t> xVector{functional_->getDomainDescriptor()};
        if (xStart.has_value()) {
            xVector = *xStart;
        } else {
            xVector = 0;
        }

        // kIndex <= 0
        index_t kIndex = 0;
        // rVector <= -f'(xVector)
        auto rVector = static_cast<data_t>(-1.0) * functional_->getGradient(xVector);
        // dVector <= rVector
        auto dVector = rVector;
        // deltaNew <= rVector^T * dVector
        auto deltaNew = rVector.dot(dVector);
        // deltaZero <= deltaNew
        auto deltaZero = deltaNew;

        Logger::get("CGNL")->info("Preparations done, tooke {}s", aggregate_time);
        Logger::get("CGNL")->info("epsilon: {}", epsilon_);
        Logger::get("CGNL")->info("delta zero: {}", deltaZero);

        // log history legend
        Logger::get("CGNL")->info("{:^6}|{:*^16}|{:*^16}|{:*^10}|{:*^10}|{:*^16}|", "iter",
                                  "sqrtDeltaNew", "sqrtDeltaZero", "time", "elapsed", "alpha");

        for (index_t it = 0; it != iterations; ++it) {
            spdlog::stopwatch iter_time;

            // Check if convergence is reached
            if (deltaNew <= epsilon_ * epsilon_ * deltaZero) {
                Logger::get("CGNL")->info("SUCCESS: Reached convergence at {}/{} iteration", it + 1,
                                          iterations);
                return xVector;
            }

            // deltaD = dVector^T * dVector
            auto deltaD = dVector.dot(dVector);

            // line search
            xVector = line_search_function_(*functional_, xVector, dVector, deltaD, epsilon_);

            // beta function
            // r <= -f'(x)
            rVector = static_cast<data_t>(-1.0) * functional_->getGradient(xVector);

            data_t beta;
            std::tie(beta, deltaNew) = beta_function_(dVector, rVector, deltaNew);

            Logger::get("CGNL")->info("{:>5} |{:>15} |{:>15} | {:>6.3} |{:>6.3}s |", it,
                                      std::sqrt(deltaNew), std::sqrt(deltaZero), iter_time,
                                      aggregate_time);

            kIndex = kIndex + 1;
            // restart Nonlinear CG  whenever the Polak-Ribière parameter beta is negative
            if (kIndex >= nIterationMax || beta <= 0.0) {
                dVector = rVector;
                kIndex = 0;
            } else {
                dVector = rVector + beta * dVector;
            }
        }

        Logger::get("CGNL")->warn("Failed to reach convergence at {} iterations", iterations);

        return xVector;
    }

    template <typename data_t>
    CGNL<data_t>* CGNL<data_t>::cloneImpl() const
    {
        return new CGNL(*functional_, epsilon_);
    }

    template <typename data_t>
    bool CGNL<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherCG = downcast_safe<CGNL>(&other);
        if (!otherCG)
            return false;

        if (epsilon_ != otherCG->epsilon_)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class CGNL<float>;

    template class CGNL<double>;

} // namespace elsa