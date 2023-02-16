#include "TransmissionLogLikelihood.h"
#include "DataContainer.h"
#include "LinearOperator.h"
#include "Scaling.h"
#include "Error.h"
#include "TypeCasts.hpp"

#include <cmath>

namespace elsa
{

    template <typename data_t>
    TransmissionLogLikelihood<data_t>::TransmissionLogLikelihood(
        const DataDescriptor& domainDescriptor, const DataContainer<data_t>& y,
        const DataContainer<data_t>& b, const DataContainer<data_t>& r)
        : Functional<data_t>(domainDescriptor), y_{y}, b_{b}, r_{r}
    {
        // sanity check
        if (domainDescriptor != y.getDataDescriptor() || domainDescriptor != b.getDataDescriptor()
            || domainDescriptor != r.getDataDescriptor())
            throw InvalidArgumentError(
                "TransmissionLogLikelihood: descriptor and y/b/r not matching in size.");
    }

    template <typename data_t>
    TransmissionLogLikelihood<data_t>::TransmissionLogLikelihood(
        const DataDescriptor& domainDescriptor, const DataContainer<data_t>& y,
        const DataContainer<data_t>& b)
        : Functional<data_t>(domainDescriptor), y_{y}, b_{b}
    {
        // sanity check
        if (domainDescriptor != y.getDataDescriptor() || domainDescriptor != b.getDataDescriptor())
            throw InvalidArgumentError(
                "TransmissionLogLikelihood: descriptor and y/b not matching in size.");
    }

    template <typename data_t>
    TransmissionLogLikelihood<data_t>::TransmissionLogLikelihood(const LinearOperator<data_t>& A,
                                                                 const DataContainer<data_t>& y,
                                                                 const DataContainer<data_t>& b,
                                                                 const DataContainer<data_t>& r)
        : Functional<data_t>(A.getDomainDescriptor()), A_(A.clone()), y_{y}, b_{b}, r_{r}
    {
        // sanity check
        if (A.getRangeDescriptor() != y.getDataDescriptor()
            || A.getRangeDescriptor() != b.getDataDescriptor()
            || A.getRangeDescriptor() != r.getDataDescriptor())
            throw InvalidArgumentError(
                "TransmissionLogLikelihood: operator and y/b/r not matching in size.");
    }

    template <typename data_t>
    TransmissionLogLikelihood<data_t>::TransmissionLogLikelihood(const LinearOperator<data_t>& A,
                                                                 const DataContainer<data_t>& y,
                                                                 const DataContainer<data_t>& b)
        : Functional<data_t>(A.getDomainDescriptor()), A_(A.clone()), y_{y}, b_{b}
    {
        // sanity check
        if (A.getRangeDescriptor() != y.getDataDescriptor()
            || A.getRangeDescriptor() != b.getDataDescriptor())
            throw InvalidArgumentError(
                "TransmissionLogLikelihood: operator and y/b not matching in size.");
    }

    template <typename data_t>
    data_t TransmissionLogLikelihood<data_t>::evaluateImpl(const DataContainer<data_t>& x)
    {
        if (A_ && x.getDataDescriptor() != A_->getDomainDescriptor()) {
            throw InvalidArgumentError(
                "TransmissionLogLikelihood: given x is not the correct size");
        }

        if (!A_ && x.getDataDescriptor() != y_.getDataDescriptor()) {
            throw InvalidArgumentError(
                "TransmissionLogLikelihood: given x is not the correct size");
        }

        auto result = static_cast<data_t>(0.0);

        auto Rx = [&]() {
            if (A_) {
                return A_->apply(x);
            } else {
                return x;
            }
        }();

        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t temp = b_[i] * std::exp(-Rx[i]);
            if (r_.has_value())
                temp += (*r_)[i];

            result += temp - y_[i] * std::log(temp);
        }

        return result;
    }

    template <typename data_t>
    void TransmissionLogLikelihood<data_t>::getGradientImpl(const DataContainer<data_t>& x,
                                                            DataContainer<data_t>& out)
    {
        // Actual computation of the functional
        auto translog = [&](auto in, auto i) {
            data_t temp = b_[i] * std::exp(-in);
            in = -temp;

            if (r_.has_value())
                in += y_[i] * temp / (temp + (*r_)[i]);
            else
                in += y_[i];

            return in;
        };

        if (A_) {
            auto Rx = A_->apply(x);
            for (index_t i = 0; i < Rx.getSize(); ++i) {
                Rx[i] = translog(Rx[i], i);
            }
            A_->applyAdjoint(Rx, out);
        } else {
            for (index_t i = 0; i < x.getSize(); ++i) {
                out[i] = translog(x[i], i);
            }
        }
    }

    template <typename data_t>
    LinearOperator<data_t>
        TransmissionLogLikelihood<data_t>::getHessianImpl(const DataContainer<data_t>& x)
    {
        auto scale = [&](auto in) {
            DataContainer<data_t> s(in.getDataDescriptor());
            for (index_t i = 0; i < in.getSize(); ++i) {
                s[i] = b_[i] * std::exp(-in[i]);
                if (r_.has_value()) {
                    data_t tempR = s[i] + (*r_)[i];
                    s[i] += (*r_)[i] * y_[i] * s[i] / (tempR * tempR);
                }
            }

            return leaf(Scaling<data_t>(s.getDataDescriptor(), s));
        };

        if (A_) {
            auto Rx = A_->apply(x);
            auto s = scale(Rx);

            // Jacobian is the operator, plus chain rule
            return adjoint(*A_) * s * (*A_);
        } else {
            // Jacobian is the identity, no need for chain rule
            return scale(x);
        }
    }

    template <typename data_t>
    TransmissionLogLikelihood<data_t>* TransmissionLogLikelihood<data_t>::cloneImpl() const
    {
        if (A_ && r_.has_value()) {
            return new TransmissionLogLikelihood<data_t>(*A_, y_, b_, *r_);
        }
        if (A_ && !r_.has_value()) {
            return new TransmissionLogLikelihood<data_t>(*A_, y_, b_);
        }
        if (!A_ && r_.has_value()) {
            return new TransmissionLogLikelihood<data_t>(this->getDomainDescriptor(), y_, b_, *r_);
        } else { // (!A_ && !r_.has_value())
            return new TransmissionLogLikelihood<data_t>(this->getDomainDescriptor(), y_, b_);
        }
    }

    template <typename data_t>
    bool TransmissionLogLikelihood<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherTLL = downcast_safe<TransmissionLogLikelihood>(&other);
        if (!otherTLL)
            return false;

        if (y_ != otherTLL->y_ || b_ != otherTLL->b_)
            return false;

        if (r_.has_value() && otherTLL->r_.has_value() && *r_ != *otherTLL->r_)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class TransmissionLogLikelihood<float>;
    template class TransmissionLogLikelihood<double>;
    // no complex instantiations, they make no sense

} // namespace elsa
