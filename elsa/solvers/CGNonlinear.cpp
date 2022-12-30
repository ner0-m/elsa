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
                                                     std::optional<DataContainer<data_t>> x0)
    {
        spdlog::stopwatch aggregate_time;
        Logger::get("CGNonlinear")->info("Start preparations...");

        // get references to some variables in the Quadric
        auto x = DataContainer<data_t>(_problem->getDataTerm().getDomainDescriptor());
        if ((x0.has_value())) {
            x = *x0;
        } else {
            x = 0;
        }

        index_t k = 0;
        // r <= -f'(x)
        auto r = _problem->getGradient(x);
        r *= static_cast<data_t>(-1.0);

        // s <= M^(-1) * r
        // auto s = _preconditionerInverse ? _preconditionerInverse->apply(r) : r;
        // d <= s
        auto d = r;
        // deltaNew <= r^T * d
        auto deltaNew = r.dot(d);
        // deltaZero <= deltaNew
        auto deltaZero = deltaNew;

        Logger::get("CGNonlinear")->info("Preparations done, tooke {}s", aggregate_time);

        Logger::get("CGNonlinear")->info("epsilon: {}", _epsilon);
        Logger::get("CGNonlinear")->info("delta zero: {}", std::sqrt(deltaZero));

        // log history legend
        Logger::get("CGNonlinear")
            ->info("{:^6}|{:*^16}|{:*^16}|{:*^8}|{:*^8}|", "iter", "deltaNew", "deltaZero", "time",
                   "elapsed");

        for (index_t it = 0; it != iterations; ++it) {
            spdlog::stopwatch iter_time;
            if (deltaNew <= _epsilon * _epsilon * deltaZero) {
                Logger::get("CGNonlinear")
                    ->info("SUCCESS: Reached convergence at {}/{} iteration", it + 1, iterations);
                return x;
            }

            // deltaD = d^T * d
            auto deltaD = d.dot(d);

            for (index_t j = 0; j < 10;) {
                auto alpha = static_cast<data_t>(-1.0) * _problem->getGradient(x).dot(d)
                             / d.dot(_problem->getHessian(x).apply(d));
                x = x + alpha * d;
                j = j + 1;

                if (alpha * alpha * deltaD < _epsilon * _epsilon) {
                    break;
                }
            }

            /*
            // alpha <= -sigmaZero
            auto alpha = static_cast<data_t>(-1.0) * sigmaZero;


            // etaPrev <= [f'(x + sigmaZero * d)]^T * d
            auto etaPrev = _problem.getGradient(x + sigmaZero * d).dot(d);

            index_t j = 0;
            index_t jMax = 1000;
            do {
                // auto eta = _problem.getGradient(x).dot(d);
                // alpha = alpha * eta / (etaPrev - eta);
                alpha = static_cast<data_t>(-1.0)
                        * (_problem.getGradient(x).dot(d) / d.dot(_problem.getHessian(x).apply(d)));
                x = x + alpha * d;
                j = j + 1;
            }
            // while j < jMax and alpha^2 * deltaD > epsilon^2
            while (j < jMax && alpha * alpha * deltaD > _epsilon * _epsilon);
            */

            r = static_cast<data_t>(-1.0) * _problem->getGradient(x);
            auto deltaOld = deltaNew;
            auto deltaMid = r.dot(d);

            // s = 1 / _problem.getHessian(x);
            deltaNew = r.dot(r);

            auto beta = (deltaNew - deltaMid) / deltaOld;

            Logger::get("CGNonlinear")
                ->info("{:>5} |{:>15} |{:>15} | {:>6.3} |{:>6.3}s |", it, std::sqrt(deltaNew), 0,
                       iter_time, aggregate_time);

            auto n = 10;
            k = k + 1;
            if (k >= n || beta <= 0.0) {
                d = r;
                k = 0;
            } else {
                d = r + beta * d;
            }
        }

        Logger::get("CGNonlinear")
            ->warn("Failed to reach convergence at {} iterations", iterations);

        return x;
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