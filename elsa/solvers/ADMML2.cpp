#include "ADMML2.h"

#include "DataContainer.h"
#include "LinearOperator.h"
#include "Solver.h"
#include "ProximalOperator.h"
#include "TypeCasts.hpp"
#include "elsaDefines.h"
#include "Logger.h"
#include "RegularizedInversion.h"
#include "PowerIterations.h"

#include <cmath>
#include <memory>
#include <optional>
#include <vector>

namespace elsa
{
    template <class data_t>
    ADMML2<data_t>::ADMML2(const LinearOperator<data_t>& op, const DataContainer<data_t>& b,
                           const LinearOperator<data_t>& A, const ProximalOperator<data_t>& proxg,
                           std::optional<data_t> tau, index_t ninneriters)
        : Solver<data_t>(),
          op_(op.clone()),
          b_(b),
          A_(A.clone()),
          proxg_(proxg),
          tau_(0),
          ninneriters_(ninneriters)
    {
        auto eigenval = data_t{1} / powerIterations(adjoint(A) * A);

        if (tau.has_value()) {
            tau_ = *tau;
            if (tau_ < 0 || tau_ > eigenval) {
                Logger::get("ADMML2")->info("tau ({:8.5}), should be between 0 and {:8.5}", tau_,
                                            eigenval);
            }
        } else {
            tau_ = 0.9 * eigenval;
            Logger::get("ADMML2")->info("tau is chosen {}", tau_, eigenval);
        }
    }

    template <class data_t>
    DataContainer<data_t> ADMML2<data_t>::solve(index_t iterations,
                                                std::optional<DataContainer<data_t>> x0)
    {
        const auto& domain = op_->getDomainDescriptor();
        const auto& range = op_->getRangeDescriptor();

        auto x = extract_or(x0, domain);

        auto z = zeros<data_t>(range);
        auto u = zeros<data_t>(range);

        auto Ax = empty<data_t>(range);
        auto tmp = empty<data_t>(range);

        auto sqrttau = data_t{1} / std::sqrt(tau_);

        auto loglevel = Logger::getLevel();
        Logger::get("ADMML2")->info("| {:^4} | {:^12} | {:^12} | {:^12} |", "iter", "f", "z", "u");
        for (index_t iter = 0; iter < iterations; ++iter) {
            Logger::setLevel(Logger::LogLevel::ERR);

            // x_{k+1} = \min_x 0.5 ||Op x - b||_2^2 + \frac{1}{2\tau}||Ax - z_k + u_k||_2^2
            x = reguarlizedInversion<data_t>(*op_, b_, *A_, z - u, sqrttau, ninneriters_, x);

            Logger::setLevel(loglevel);

            A_->apply(x, Ax);

            // Ax_{k+1} + u_k
            lincomb(1, Ax, 1, u, tmp);

            // z_{k+1} = prox_{\tau * g}(Ax_{k+1} + u_k)
            z = proxg_.apply(tmp, tau_);

            // u_{k+1} = u_k + Ax_{k+1} - z_{k+1}
            u += Ax;
            u -= z;

            Logger::get("ADMML2")->info("| {:>4} | {:12.7} | {:12.7} | {:12.7} |", iter,
                                        0.5 * (op_->apply(x) - b_).l2Norm(), z.l2Norm(),
                                        u.l2Norm());
        }

        return x;
    }

    template <class data_t>
    ADMML2<data_t>* ADMML2<data_t>::cloneImpl() const
    {
        return new ADMML2(*op_, b_, *A_, proxg_, tau_, ninneriters_);
    }

    template <class data_t>
    bool ADMML2<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherADMM = downcast_safe<ADMML2>(&other);
        if (!otherADMM)
            return false;

        if (*op_ != *otherADMM->op_)
            return false;

        if (*A_ != *otherADMM->A_)
            return false;

        if (tau_ != otherADMM->tau_)
            return false;

        if (ninneriters_ != otherADMM->ninneriters_)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class ADMML2<float>;
    template class ADMML2<double>;
} // namespace elsa
