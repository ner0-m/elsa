#include "LinearizedADMM.h"
#include "LinearOperator.h"
#include "ProximalOperator.h"
#include "TypeTraits.hpp"
#include "PowerIterations.h"
#include "elsaDefines.h"
#include "DataContainer.h"
#include "Logger.h"

namespace elsa
{

    template <class data_t>
    LinearizedADMM<data_t>::LinearizedADMM(const LinearOperator<data_t>& K,
                                           ProximalOperator<data_t> proxf,
                                           ProximalOperator<data_t> proxg, SelfType_t<data_t> sigma,
                                           SelfType_t<data_t> tau, bool computeKNorm)
        : K_(K.clone()), proxf_(proxf), proxg_(proxg), sigma_(sigma), tau_(tau)
    {
        if (sigma_ <= 0) {
            throw Error("`sigma` must be strictly positive. Got {}", sigma_);
        }

        if (tau_ <= 0) {
            throw Error("`tau` must be strictly positive. Got {}", tau_);
        }

        if (computeKNorm) {
            auto Knorm = powerIterations(adjoint(K) * K);
            Logger::get("LinearizedADMM")
                ->info("Knorm: {}, sigma / || K ||_2^2: {}", Knorm, sigma_ / Knorm);

            if (!(0 < tau_ && tau_ < sigma_ / Knorm)) {
                Logger::get("LinearizedADMM")
                    ->warn("Parameters do not satisfy: 0 < tau < sigma / ||K||_2^2 ({}, {}). Might "
                           "not "
                           "converge.",
                           tau_, sigma_ / Knorm);
            }
        }
    }

    template <class data_t>
    DataContainer<data_t> LinearizedADMM<data_t>::solve(index_t iterations,
                                                        std::optional<DataContainer<data_t>> x0)
    {
        auto& domain = K_->getDomainDescriptor();
        auto& range = K_->getRangeDescriptor();

        auto x = DataContainer<data_t>(domain);
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        auto z = DataContainer<data_t>(range);
        z = 0;

        auto u = DataContainer<data_t>(range);
        u = 0;

        // Temporary for Kx + u - z
        auto tmpRange = K_->apply(x);

        // Temporary for L^T (Lx + u - z)
        auto tmpDomain = DataContainer<data_t>(domain);

        for (index_t iter = 0; iter < iterations; ++iter) {
            Logger::get("LinearizedADMM")->info("Iterations: {:>3} / {:>3}", iter, iterations);
            // tmpRange is equal to Kx here, so we can compute:
            // K^T(Kx + u - z)
            tmpRange += u;
            tmpRange -= z;
            K_->applyAdjoint(tmpRange, tmpDomain);

            // x = x^k - (tau/sigma) K^T(Kx^k + u^k - z^k)
            tmpDomain *= -(tau_ / sigma_);
            x += tmpDomain;

            // x^{k+1} = prox_{tau * f}(x)
            proxf_.apply(x, geometry::Threshold{tau_}, x);

            // tmp_ran = Kx^{k + 1}
            K_->apply(x, tmpRange);

            // z^{k+1} = prox{sigma * g}(Kx^{k+1} + u^k)
            z = proxg_.apply(tmpRange + u, geometry::Threshold{sigma_});

            // u^{k+1} = u^k + Kx^{k+1} - z^{k+1}
            u += tmpRange;
            u -= z;
        }

        return x;
    }

    template <class data_t>
    LinearizedADMM<data_t>* LinearizedADMM<data_t>::cloneImpl() const
    {
        return new LinearizedADMM<data_t>(*K_, proxf_, proxg_, sigma_, tau_, false);
    }

    template <class data_t>
    bool LinearizedADMM<data_t>::isEqual(const Solver<data_t>& other) const
    {
        return false;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class LinearizedADMM<float>;
    template class LinearizedADMM<double>;
} // namespace elsa
