#include "IS_ADMML2.h"

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
    IS_ADMML2<data_t>::IS_ADMML2(const LinearOperator<data_t>& op, const DataContainer<data_t>& b,
                                 const LinearOperator<data_t>& A,
                                 const ProximalOperator<data_t>& proxg, std::optional<data_t> tau,
                                 index_t ninneriters)
        : IterativeSolver<data_t>{A, b},
          op_(op.clone()),
          proxg_(proxg),
          tau_(0),
          sqrttau(0),
          ninneriters_(ninneriters),
          z{IS::A->getRangeDescriptor()},
          u{IS::A->getRangeDescriptor()},
          Ax{IS::A->getRangeDescriptor()}
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
        sqrttau = data_t{1} / std::sqrt(tau_);
    }

    template <typename data_t>
    void IS_ADMML2<data_t>::reset()
    {
        z = 0;
        u = 0;
        Ax = 0;
    }

    template <typename data_t>
    DataContainer<data_t> IS_ADMML2<data_t>::step(DataContainer<data_t> state)
    {
        DataContainer<data_t> x{op_->getDomainDescriptor()};
        DataContainer<data_t> tmp{IS::A->getRangeDescriptor()};

        x = state;

        // x_{k+1} = \min_x 0.5 ||Op x - b||_2^2 + \frac{1}{2\tau}||Ax - z_k + u_k||_2^2
        x = reguarlizedInversion<data_t>(*op_, IS::b, *IS::A, z - u, sqrttau, ninneriters_, x);
        IS::A->apply(x, Ax); // Have to use this to access base member of templated class... other
                             // option would be this->

        // Ax_{k+1} + u_k
        lincomb(1, Ax, 1, u, tmp);

        // z_{k+1} = prox_{\tau * g}(Ax_{k+1} + u_k)
        z = proxg_.apply(tmp, tau_);

        // u_{k+1} = u_k + Ax_{k+1} - z_{k+1}
        u += Ax;
        u -= z;

        return x;
    }

    template <class data_t>
    IS_ADMML2<data_t>* IS_ADMML2<data_t>::cloneImpl() const
    {
        return new IS_ADMML2(*op_, IS::b, *IS::A, proxg_, tau_, ninneriters_);
    }

    template <class data_t>
    bool IS_ADMML2<data_t>::isEqual(const IterativeSolver<data_t>& other) const
    {
        auto otherADMM = downcast_safe<IS_ADMML2>(&other);
        if (!otherADMM)
            return false;

        if (*op_ != *otherADMM->op_)
            return false;

        if (*IS::A != *otherADMM->A)
            return false;

        if (tau_ != otherADMM->tau_)
            return false;

        if (ninneriters_ != otherADMM->ninneriters_)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class IS_ADMML2<float>;
    template class IS_ADMML2<double>;
} // namespace elsa
