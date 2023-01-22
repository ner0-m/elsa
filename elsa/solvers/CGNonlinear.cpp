#pragma clang diagnostic push

#include "CGNonlinear.h"
#include "Logger.h"
#include "TypeCasts.hpp"
#include "spdlog/stopwatch.h"

namespace elsa
{

    template <typename data_t>
    CGNonlinear<data_t>::CGNonlinear(const Problem<data_t>& problem, data_t epsilon)
        : Solver<data_t>(), _problem{problem.clone()}, _epsilon{epsilon}
    {
    }

    template <typename data_t>
    DataContainer<data_t> CGNonlinear<data_t>::solve(index_t iterations,
                                                     std::optional<DataContainer<data_t>> xStart)
    {
        spdlog::stopwatch aggregate_time;
        Logger::get("CGNonlinear")->info("Start preparations...");

        // Amount of inner iterations for Newton-Raphson
        index_t jIterationMax = 10;
        // Restart Nonlinear CG every n iterations, with n being the number of dimensions
        index_t nIterationMax =
            _problem->getDataTerm().getDomainDescriptor().getNumberOfDimensions();

        // use xStart as start point if provided, use 0 otherwise
        DataContainer<data_t> xVector{_problem->getDataTerm().getDomainDescriptor()};
        if (xStart.has_value()) {
            xVector = *xStart;
        } else {
            xVector = 0;
        }

        // kIndex <= 0
        index_t kIndex = 0;
        // rVector <= -f'(xVector)
        auto rVector = static_cast<data_t>(-1.0) * _problem->getGradient(xVector);
        // dVector <= rVector
        auto dVector = rVector;
        // deltaNew <= rVector^T * dVector
        auto deltaNew = rVector.dot(dVector);
        // deltaZero <= deltaNew
        auto deltaZero = deltaNew;

        Logger::get("CGNonlinear")->info("Preparations done, tooke {}s", aggregate_time);
        Logger::get("CGNonlinear")->info("epsilon: {}", _epsilon);
        Logger::get("CGNonlinear")->info("delta zero: {}", deltaZero);

        // log history legend
        Logger::get("CGNonlinear")
            ->info("{:^6}|{:*^16}|{:*^16}|{:*^8}|{:*^8}|{:*^16}|", "iter", "sqrtDeltaNew",
                   "sqrtDeltaZero", "time", "elapsed", "alpha");

        for (index_t it = 0; it != iterations; ++it) {
            spdlog::stopwatch iter_time;

            // Check if convergence is reached
            if (deltaNew <= _epsilon * _epsilon * deltaZero) {
                Logger::get("CGNonlinear")
                    ->info("SUCCESS: Reached convergence at {}/{} iteration", it + 1, iterations);
                return xVector;
            }

            // deltaD = dVector^T * dVector
            auto deltaD = dVector.dot(dVector);

            // Newton-Raphson iterations
            for (index_t j = 0; j < jIterationMax; j++) {
                spdlog::stopwatch iter_time_inner;
                // alpha <= -([f'(xVector)]^T * d) / (d^T * f''(xVector) * d)
                auto alpha = static_cast<data_t>(-1.0) * _problem->getGradient(xVector).dot(dVector)
                             / dVector.dot(_problem->getHessian(xVector).apply(dVector));
                // xVector <= xVector + alpha * d
                xVector = xVector + alpha * dVector;

                Logger::get("CGNonlinear")
                    ->info("{:>5} |{:>15} |{:>15} | {:>6.3} |{:>6.3}s |{:>15} |", it, j, "-",
                           iter_time_inner, aggregate_time, alpha);

                // break if converged
                if (alpha * alpha * deltaD < _epsilon * _epsilon) {
                    break;
                }
            }

            // Polak-Ribière parameters
            // r <= -f'(x)
            rVector = static_cast<data_t>(-1.0) * _problem->getGradient(xVector);
            // deltaOld <= deltaNew
            auto deltaOld = deltaNew;
            // deltaMid <= r^T * d
            auto deltaMid = rVector.dot(dVector);
            // deltaNew <= r^T * r
            deltaNew = rVector.dot(rVector);

            // beta <= (deltaNew - deltaMid) / deltaOld
            auto beta = (deltaNew - deltaMid) / deltaOld;

            Logger::get("CGNonlinear")
                ->info("{:>5} |{:>15} |{:>15} | {:>6.3} |{:>6.3}s |", it, std::sqrt(deltaNew),
                       std::sqrt(deltaZero), iter_time, aggregate_time);

            kIndex = kIndex + 1;
            // restart Nonlinear CG  whenever the Polak-Ribière parameter beta is negative
            if (kIndex >= nIterationMax || beta <= 0.0) {
                dVector = rVector;
                kIndex = 0;
            } else {
                dVector = rVector + beta * dVector;
            }
        }

        Logger::get("CGNonlinear")
            ->warn("Failed to reach convergence at {} iterations", iterations);

        return xVector;
    }

    template <typename data_t>
    CGNonlinear<data_t>* CGNonlinear<data_t>::cloneImpl() const
    {
        return new CGNonlinear(*_problem, _epsilon);
    }

    template <typename data_t>
    bool CGNonlinear<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherCG = downcast_safe<CGNonlinear>(&other);
        if (!otherCG)
            return false;

        if (_epsilon != otherCG->_epsilon)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class CGNonlinear<float>;

    template class CGNonlinear<double>;

} // namespace elsa

#pragma clang diagnostic pop