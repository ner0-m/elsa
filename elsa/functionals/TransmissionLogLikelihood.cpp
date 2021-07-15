#include "TransmissionLogLikelihood.h"
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
        : Functional<data_t>(domainDescriptor),
          _y{std::make_unique<DataContainer<data_t>>(y)},
          _b{std::make_unique<DataContainer<data_t>>(b)},
          _r{std::make_unique<DataContainer<data_t>>(r)}
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
        : Functional<data_t>(domainDescriptor),
          _y{std::make_unique<DataContainer<data_t>>(y)},
          _b{std::make_unique<DataContainer<data_t>>(b)}
    {
        // sanity check
        if (domainDescriptor != y.getDataDescriptor() || domainDescriptor != b.getDataDescriptor())
            throw InvalidArgumentError(
                "TransmissionLogLikelihood: descriptor and y/b not matching in size.");
    }

    template <typename data_t>
    TransmissionLogLikelihood<data_t>::TransmissionLogLikelihood(const Residual<data_t>& residual,
                                                                 const DataContainer<data_t>& y,
                                                                 const DataContainer<data_t>& b,
                                                                 const DataContainer<data_t>& r)
        : Functional<data_t>(residual),
          _y{std::make_unique<DataContainer<data_t>>(y)},
          _b{std::make_unique<DataContainer<data_t>>(b)},
          _r{std::make_unique<DataContainer<data_t>>(r)}
    {
        // sanity check
        if (residual.getRangeDescriptor() != y.getDataDescriptor()
            || residual.getRangeDescriptor() != b.getDataDescriptor()
            || residual.getRangeDescriptor() != r.getDataDescriptor())
            throw InvalidArgumentError(
                "TransmissionLogLikelihood: residual and y/b/r not matching in size.");
    }

    template <typename data_t>
    TransmissionLogLikelihood<data_t>::TransmissionLogLikelihood(const Residual<data_t>& residual,
                                                                 const DataContainer<data_t>& y,
                                                                 const DataContainer<data_t>& b)
        : Functional<data_t>(residual),
          _y{std::make_unique<DataContainer<data_t>>(y)},
          _b{std::make_unique<DataContainer<data_t>>(b)}
    {
        // sanity check
        if (residual.getRangeDescriptor() != y.getDataDescriptor()
            || residual.getRangeDescriptor() != b.getDataDescriptor())
            throw InvalidArgumentError(
                "TransmissionLogLikelihood: residual and y/b not matching in size.");
    }

    template <typename data_t>
    data_t TransmissionLogLikelihood<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        auto result = static_cast<data_t>(0.0);

        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t temp = (*_b)[i] * std::exp(-Rx[i]);
            if (_r)
                temp += (*_r)[i];

            result += temp - (*_y)[i] * std::log(temp);
        }

        return result;
    }

    template <typename data_t>
    void TransmissionLogLikelihood<data_t>::getGradientInPlaceImpl(DataContainer<data_t>& Rx)
    {
        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t temp = (*_b)[i] * std::exp(-Rx[i]);
            Rx[i] = -temp;

            if (_r)
                Rx[i] += (*_y)[i] * temp / (temp + (*_r)[i]);
            else
                Rx[i] += (*_y)[i];
        }
    }

    template <typename data_t>
    LinearOperator<data_t>
        TransmissionLogLikelihood<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        DataContainer<data_t> scaleFactors(Rx.getDataDescriptor());
        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t temp = (*_b)[i] * std::exp(-Rx[i]);
            scaleFactors[i] = temp;
            if (_r) {
                data_t tempR = temp + (*_r)[i];
                scaleFactors[i] += (*_r)[i] * (*_y)[i] * temp / (tempR * tempR);
            }
        }

        return leaf(Scaling<data_t>(Rx.getDataDescriptor(), scaleFactors));
    }

    template <typename data_t>
    TransmissionLogLikelihood<data_t>* TransmissionLogLikelihood<data_t>::cloneImpl() const
    {
        if (_r)
            return new TransmissionLogLikelihood<data_t>(this->getResidual(), *_y, *_b, *_r);
        else
            return new TransmissionLogLikelihood<data_t>(this->getResidual(), *_y, *_b);
    }

    template <typename data_t>
    bool TransmissionLogLikelihood<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherTLL = downcast_safe<TransmissionLogLikelihood>(&other);
        if (!otherTLL)
            return false;

        if (*_y != *otherTLL->_y || *_b != *otherTLL->_b)
            return false;

        if (_r && *_r != *otherTLL->_r)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class TransmissionLogLikelihood<float>;
    template class TransmissionLogLikelihood<double>;
    // no complex instantiations, they make no sense

} // namespace elsa
