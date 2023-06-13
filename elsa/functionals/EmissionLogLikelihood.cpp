#include "EmissionLogLikelihood.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "LinearOperator.h"
#include "Scaling.h"
#include "TypeCasts.hpp"

#include <cmath>
#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    EmissionLogLikelihood<data_t>::EmissionLogLikelihood(const LinearOperator<data_t>& A,
                                                         const DataContainer<data_t>& y,
                                                         const DataContainer<data_t>& r)
        : Functional<data_t>(A.getDomainDescriptor()), A_(A.clone()), y_{y}, r_{r}
    {
        // sanity check
        if (A.getRangeDescriptor() != y.getDataDescriptor()
            || A.getRangeDescriptor() != r.getDataDescriptor())
            throw InvalidArgumentError(
                "EmissionLogLikelihood: residual and y/r not matching in size.");
    }

    template <typename data_t>
    EmissionLogLikelihood<data_t>::EmissionLogLikelihood(const LinearOperator<data_t>& A,
                                                         const DataContainer<data_t>& y)
        : Functional<data_t>(A.getDomainDescriptor()), A_(A.clone()), y_{y}
    {
        // sanity check
        if (A.getRangeDescriptor() != y.getDataDescriptor())
            throw InvalidArgumentError(
                "EmissionLogLikelihood: residual and y not matching in size.");
    }

    template <typename data_t>
    bool EmissionLogLikelihood<data_t>::isDifferentiable() const
    {
        return true;
    }

    template <typename data_t>
    data_t EmissionLogLikelihood<data_t>::evaluateImpl(const DataContainer<data_t>& x)
    {
        if (x.getDataDescriptor() != A_->getDomainDescriptor()) {
            throw InvalidArgumentError("EmissionLogLikelihood: given x is not the correct size");
        }

        auto result = static_cast<data_t>(0.0);

        auto Rx = A_->apply(x);
        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t temp = Rx[i];
            if (r_)
                temp += (*r_)[i];

            result += temp - y_[i] * std::log(temp);
        }

        return result;
    }

    template <typename data_t>
    void EmissionLogLikelihood<data_t>::getGradientImpl(const DataContainer<data_t>& x,
                                                        DataContainer<data_t>& out)
    {
        auto emissionlog = [&](auto in, auto i) {
            data_t temp = in;
            if (r_)
                temp += (*r_)[i];

            return 1 - y_[i] / temp;
        };

        auto Rx = A_->apply(x);
        for (index_t i = 0; i < Rx.getSize(); ++i) {
            Rx[i] = emissionlog(Rx[i], i);
        }
        A_->applyAdjoint(Rx, out);
    }

    template <typename data_t>
    LinearOperator<data_t>
        EmissionLogLikelihood<data_t>::getHessianImpl(const DataContainer<data_t>& x)
    {
        auto scale = [&](auto in) {
            DataContainer<data_t> s(in.getDataDescriptor());
            for (index_t i = 0; i < in.getSize(); ++i) {
                data_t temp = in[i];
                if (r_)
                    temp += (*r_)[i];

                s[i] = y_[i] / (temp * temp);
            }

            return leaf(Scaling<data_t>(s.getDataDescriptor(), s));
        };

        auto Rx = A_->apply(x);

        // Jacobian is the operator, plus chain rule
        return adjoint(*A_) * scale(Rx) * (*A_);
    }

    template <typename data_t>
    EmissionLogLikelihood<data_t>* EmissionLogLikelihood<data_t>::cloneImpl() const
    {
        if (r_.has_value()) {
            return new EmissionLogLikelihood<data_t>(*A_, y_, *r_);
        }
        return new EmissionLogLikelihood<data_t>(*A_, y_);
    }

    template <typename data_t>
    bool EmissionLogLikelihood<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherELL = downcast_safe<EmissionLogLikelihood>(&other);
        if (!otherELL)
            return false;

        if (r_ && otherELL->r_ && *r_ != *otherELL->r_) {
            return false;
        }

        return y_ == otherELL->y_;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class EmissionLogLikelihood<float>;
    template class EmissionLogLikelihood<double>;
    // no complex instantiations, they make no sense

} // namespace elsa
