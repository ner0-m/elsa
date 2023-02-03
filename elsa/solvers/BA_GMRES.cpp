#include "BA_GMRES.h"
#include "Logger.h"
#include "TypeCasts.hpp"
#include "spdlog/stopwatch.h"

namespace elsa
{
    template <typename data_t>
    BA_GMRES<data_t>::BA_GMRES(const LinearOperator<data_t>& projector,
                               const DataContainer<data_t>& sinogram, data_t epsilon)
        : Solver<data_t>(),
          _A{projector.clone()},
          _B{adjoint(projector).clone()},
          _b{sinogram},
          _epsilon{epsilon}
    {
    }

    template <typename data_t>
    BA_GMRES<data_t>::BA_GMRES(const LinearOperator<data_t>& projector,
                               const LinearOperator<data_t>& backprojector,
                               const DataContainer<data_t>& sinogram, data_t epsilon)
        : Solver<data_t>(),
          _A{projector.clone()},
          _B{backprojector.clone()},
          _b{sinogram},
          _epsilon{epsilon}
    {
    }

    template <typename data_t>
    DataContainer<data_t> BA_GMRES<data_t>::solveAndRestart(index_t iterations, index_t restarts,
                                                            std::optional<DataContainer<data_t>> x0)
    {
        auto x = DataContainer<data_t>(_A->getDomainDescriptor());
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        for (index_t k = 0; k < restarts; k++) {
            x = solve(iterations, x);
        }

        return x;
    }

    template <typename data_t>
    DataContainer<data_t> BA_GMRES<data_t>::solve(index_t iterations,
                                                  std::optional<DataContainer<data_t>> x0)
    {
        using Mat = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
        using Vec = Eigen::Vector<data_t, Eigen::Dynamic>;

        spdlog::stopwatch aggregate_time;
        Logger::get("BA_GMRES")->info("Start preparations...");

        auto x = DataContainer<data_t>(_A->getDomainDescriptor());
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        // setup DataContainer for Return Value which should be like x
        auto x_k = DataContainer<data_t>(_A->getDomainDescriptor());

        Mat h = Mat::Constant(iterations + 1, iterations, 0);
        Mat w = Mat::Constant(x.getSize(), iterations, 0);
        Vec e = Vec::Constant(iterations + 1, 0);

        // Init Calculations
        auto Bb = _B->apply(_b);
        auto Ax = _A->apply(x);
        auto BAx = _B->apply(Ax);
        auto r0 = Bb - BAx;
        auto beta = r0.l2Norm();

        // Initializing e Vector
        e(0) = beta;

        // Filling Matrix w with the vector r0/beta at the specified column
        auto w_i0 = r0 / beta;
        w.col(0) = Eigen::Map<Vec>(thrust::raw_pointer_cast(w_i0.storage().data()), w_i0.getSize());

        Logger::get("BA_GMRES")->info("Preparations done, took {}s", aggregate_time);
        Logger::get("BA_GMRES")->info("epsilon: {}", _epsilon);
        Logger::get("BA_GMRES")->info("||r0||: {}", beta);

        Logger::get("BA_GMRES")
            ->info("{:^6}|{:*^16}|{:*^8}|{:*^8}|", "iter", "r", "time", "elapsed");

        for (index_t k = 0; k < iterations; k++) {
            spdlog::stopwatch iter_time;

            auto w_k = DataContainer<data_t>(_A->getDomainDescriptor(), w.col(k));
            auto Aw_k = _A->apply(w_k);

            // Entering eigen space for the many following Matrix/Vector operations, as this seems
            // more efficient that a constant casting into elsa datacontainers
            auto temp = _B->apply(Aw_k);
            auto q_k =
                Eigen::Map<Vec>(thrust::raw_pointer_cast(temp.storage().data()), temp.getSize());

            for (index_t i = 0; i < iterations; i++) {
                auto w_i = w.col(i);
                auto h_ik = q_k.dot(w_i);

                h(i, k) = h_ik;
                q_k -= h_ik * w_i;
            }

            h(k + 1, k) = q_k.norm();

            // Source:
            // https://stackoverflow.com/questions/37962271/whats-wrong-with-my-BA_GMRES-implementation
            // This rule exists as we fill k+1 column of w and w matrix only has k columns
            // another way to implement this would be by having a matrix w with k + 1 columns and
            // instead always just getting the slice w0..wk for wy calculation
            if (k != iterations - 1) {
                w.col(k + 1) = q_k / h(k + 1, k);
            }

            Eigen::ColPivHouseholderQR<Mat> qr(h);
            Vec y = qr.solve(e);
            auto wy = DataContainer<data_t>(_A->getDomainDescriptor(), w * y);

            x_k = x + wy;

            // disable r for faster results ?
            auto r = _b - _A->apply(x_k);

            Logger::get("BA_GMRES")
                ->info("{:>5}|{:>15}|{:>6.3}|{:>6.3}s|", k, r.l2Norm(), iter_time, aggregate_time);

            //  Break Condition via relative residual, there could be more interesting approaches
            //  used here like NCP Criterion or discrepancy principle
            if (r.l2Norm() <= _epsilon) {
                Logger::get("BA_GMRES")->info("||rx|| {}", r.l2Norm());
                Logger::get("BA_GMRES")
                    ->info("SUCCESS: Reached convergence at {}/{} iteration", k + 1, iterations);
                return x_k;
            }
        }

        Logger::get("BA_GMRES")->warn("Failed to reach convergence at {} iterations", iterations);
        return x_k;
    }

    template <typename data_t>
    BA_GMRES<data_t>* BA_GMRES<data_t>::cloneImpl() const
    {
        return new BA_GMRES(*_A, *_B, _b, _epsilon);
    }

    template <typename data_t>
    bool BA_GMRES<data_t>::isEqual(const Solver<data_t>& other) const
    {
        // This is basically stolen from CG

        auto otherGMRES = downcast_safe<BA_GMRES>(&other);

        if (!otherGMRES)
            return false;

        if (_epsilon != otherGMRES->_epsilon)
            return false;

        if ((_A && !otherGMRES->_A) || (!_A && otherGMRES->_A))
            return false;

        if (_A && otherGMRES->_A)
            if (*_A != *otherGMRES->_A)
                return false;

        if ((_B && !otherGMRES->_B) || (!_B && otherGMRES->_B))
            return false;

        if (_B && otherGMRES->_B)
            if (*_B != *otherGMRES->_B)
                return false;

        if (_b != otherGMRES->_b)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class BA_GMRES<float>;
    template class BA_GMRES<double>;

} // namespace elsa