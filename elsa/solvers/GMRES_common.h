#include "Logger.h"
#include "TypeCasts.hpp"
#include "spdlog/stopwatch.h"
#include "LinearOperator.h"
#include <memory>
#include <Eigen/Core>

namespace detail
{
    using namespace elsa;

    template <typename data_t>
    DataContainer<data_t>
        gmres(std::string name, std::unique_ptr<LinearOperator<data_t>>& _A,
              std::unique_ptr<LinearOperator<data_t>>& _B, DataContainer<data_t>& _b,
              data_t _epsilon, DataContainer<data_t> x, index_t iterations,
              std::function<DataContainer<data_t>(std::unique_ptr<LinearOperator<data_t>>&,
                                                  std::unique_ptr<LinearOperator<data_t>>&,
                                                  DataContainer<data_t>&, DataContainer<data_t>&)>
                  calculate_r0,
              std::function<DataContainer<data_t>(std::unique_ptr<LinearOperator<data_t>>&,
                                                  std::unique_ptr<LinearOperator<data_t>>&,
                                                  DataContainer<data_t>&)>
                  calculate_q,
              std::function<DataContainer<data_t>(std::unique_ptr<LinearOperator<data_t>>&,
                                                  DataContainer<data_t>&, DataContainer<data_t>&)>
                  calculate_x)
    {
        // GMRES Implementation
        using Mat = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
        using Vec = Eigen::Vector<data_t, Eigen::Dynamic>;

        spdlog::stopwatch aggregate_time;
        Logger::get(name)->info("Start preparations...");

        // setup DataContainer for Return Value which should be like x
        auto x_k = DataContainer<data_t>(_A->getDomainDescriptor());

        // Custom function for AB/BA-GMRES
        auto r0 = calculate_r0(_A, _B, _b, x);

        Mat h = Mat::Constant(iterations + 1, iterations, 0);
        Mat w = Mat::Constant(r0.getSize(), iterations, 0);
        Vec e = Vec::Constant(iterations + 1, 0);

        // Initializing e Vector
        auto beta = r0.l2Norm();
        e(0) = beta;

        // Filling Matrix w with the vector r0/beta at the specified column
        auto w_i0 = r0 / beta;
        w.col(0) = Eigen::Map<Vec>(thrust::raw_pointer_cast(w_i0.storage().data()), w_i0.getSize());

        Logger::get(name)->info("Preparations done, took {}s", aggregate_time);

        Logger::get(name)->info("epsilon: {}", _epsilon);
        Logger::get(name)->info("||r0||: {}", beta);

        Logger::get(name)->info("{:^6}|{:*^16}|{:*^8}|{:*^8}|", "iter", "r", "time", "elapsed");

        for (index_t k = 0; k < iterations; k++) {
            spdlog::stopwatch iter_time;

            auto w_k = DataContainer<data_t>(r0.getDataDescriptor(), w.col(k));

            // Custom function for AB/BA-GMRES
            auto temp = calculate_q(_A, _B, w_k);

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
            // https://stackoverflow.com/questions/37962271/whats-wrong-with-my-AB_GMRES-implementation
            // This rule exists as we fill k+1 column of w and w matrix only has k columns
            // another way to implement this would be by having a matrix w with k + 1 columns and
            // instead always just getting the slice w0..wk for wy calculation
            if (k != iterations - 1) {
                w.col(k + 1) = q_k / h(k + 1, k);
            }

            Eigen::ColPivHouseholderQR<Mat> qr(h);
            Vec y = qr.solve(e);
            auto wy = DataContainer<data_t>(r0.getDataDescriptor(), w * y);

            // Custom function for AB/BA-GMRES
            x_k = calculate_x(_B, x, wy);

            // disable r for faster results ?
            auto r = _b - _A->apply(x_k);

            Logger::get(name)->info("{:>5}|{:>15}|{:>6.3}|{:>6.3}s|", k, r.l2Norm(), iter_time,
                                    aggregate_time);

            //  Break Condition via relative residual, there could be more interesting approaches
            //  used here like NCP Criterion or discrepancy principle
            if (r.l2Norm() <= _epsilon) {
                Logger::get(name)->info("||rx|| {}", r.l2Norm());
                Logger::get(name)->info("SUCCESS: Reached convergence at {}/{} iteration", k + 1,
                                        iterations);
                return x_k;
            }
        }

        Logger::get(name)->warn("Failed to reach convergence at {} iterations", iterations);
        return x_k;
    };
}; // namespace detail
