#include "CGNL.h"
#include "Logger.h"
#include "TypeCasts.hpp"
#include "spdlog/stopwatch.h"

namespace elsa
{

    template <typename data_t>
    CGNL<data_t>::CGNL(const Problem<data_t>& problem, data_t epsilon,
                       index_t line_search_iterations,
                       const LineSearchFunction& line_search_function,
                       const BetaFunction& beta_function)
        : Solver<data_t>(),
          problem_{problem.clone()},
          epsilon_{epsilon},
          line_search_iterations_{line_search_iterations},
          line_search_function_{line_search_function},
          beta_function_{beta_function}
    {
    }

    template <typename data_t>
    CGNL<data_t>::CGNL(const Problem<data_t>& problem, data_t epsilon,
                       index_t line_search_iterations)
        : CGNL<data_t>(problem, epsilon, line_search_iterations, lineSearchNewtonRaphson,
                       betaPolakRibiere)
    {
    }

    template <typename data_t>
    CGNL<data_t>::CGNL(const Problem<data_t>& problem, data_t epsilon)
        : CGNL<data_t>(problem, epsilon, 1, lineSearchConstantStepSize, betaPolakRibiere)
    {
    }

    template <typename data_t>
    DataContainer<data_t> CGNL<data_t>::solve(index_t iterations,
                                              std::optional<DataContainer<data_t>> xStart)
    {
        spdlog::stopwatch aggregate_time;
        Logger::get("CGNL")->info("Start preparations...");

        // Restart Nonlinear CG every n iterations, with n being the number of dimensions
        index_t nIterationMax =
            problem_->getDataTerm().getDomainDescriptor().getNumberOfDimensions();

        // use xStart as start point if provided, use 0 otherwise
        DataContainer<data_t> xVector{problem_->getDataTerm().getDomainDescriptor()};
        if (xStart.has_value()) {
            xVector = *xStart;
        } else {
            xVector = 0;
        }

        // kIndex <= 0
        index_t kIndex = 0;
        // rVector <= -f'(xVector)
        auto rVector = static_cast<data_t>(-1.0) * problem_->getGradient(xVector);
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

            // Newton-Raphson iterations
            for (index_t j = 0; j < line_search_iterations_; j++) {
                spdlog::stopwatch iter_time_inner;
                bool converged;
                std::tie(converged, xVector) =
                    line_search_function_(problem_, xVector, dVector, deltaD, epsilon_);

                Logger::get("CGNL")->info("{:>5} |{:>15} |{:>15} | {:>6.3} |{:>6.3}s |{:>15} |", it,
                                          j, "line search", iter_time_inner, aggregate_time, 0);
                if (converged) {
                    break;
                }
            }

            // Polak-Ribière parameters
            // r <= -f'(x)
            rVector = static_cast<data_t>(-1.0) * problem_->getGradient(xVector);

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
        return new CGNL(*problem_, epsilon_);
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