#include "EmissionLogLikelihood.h"
#include "Scaling.h"

#include <cmath>
#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    EmissionLogLikelihood<data_t>::EmissionLogLikelihood(const DataDescriptor& domainDescriptor,
                                                         const DataContainer<data_t>& y,
                                                         const DataContainer<data_t>& r)
        : Functional<data_t>(domainDescriptor),
          _y{std::make_unique<DataContainer<data_t>>(y)},
          _r{std::make_unique<DataContainer<data_t>>(r)}
    {
        // sanity check
        if (domainDescriptor != y.getDataDescriptor() || domainDescriptor != r.getDataDescriptor())
            throw std::invalid_argument(
                "EmissionLogLikelihood: descriptor and y/r not matching in size.");
    }

    template <typename data_t>
    EmissionLogLikelihood<data_t>::EmissionLogLikelihood(const DataDescriptor& domainDescriptor,
                                                         const DataContainer<data_t>& y)
        : Functional<data_t>(domainDescriptor), _y{std::make_unique<DataContainer<data_t>>(y)}
    {
        // sanity check
        if (domainDescriptor != y.getDataDescriptor())
            throw std::invalid_argument(
                "EmissionLogLikelihood: descriptor and y not matching in size.");
    }

    template <typename data_t>
    EmissionLogLikelihood<data_t>::EmissionLogLikelihood(const Residual<data_t>& residual,
                                                         const DataContainer<data_t>& y,
                                                         const DataContainer<data_t>& r)
        : Functional<data_t>(residual),
          _y{std::make_unique<DataContainer<data_t>>(y)},
          _r{std::make_unique<DataContainer<data_t>>(r)}
    {
        // sanity check
        if (residual.getRangeDescriptor() != y.getDataDescriptor()
            || residual.getRangeDescriptor() != r.getDataDescriptor())
            throw std::invalid_argument(
                "EmissionLogLikelihood: residual and y/r not matching in size.");
    }

    template <typename data_t>
    EmissionLogLikelihood<data_t>::EmissionLogLikelihood(const Residual<data_t>& residual,
                                                         const DataContainer<data_t>& y)
        : Functional<data_t>(residual), _y{std::make_unique<DataContainer<data_t>>(y)}
    {
        // sanity check
        if (residual.getRangeDescriptor() != y.getDataDescriptor())
            throw std::invalid_argument(
                "EmissionLogLikelihood: residual and y not matching in size.");
    }

    template <typename data_t>
    data_t EmissionLogLikelihood<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        auto result = static_cast<data_t>(0.0);

        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t temp = Rx[i];
            if (_r)
                temp += (*_r)[i];

            result += temp - (*_y)[i] * std::log(temp);
        }

        return result;
    }

    template <typename data_t>
    void EmissionLogLikelihood<data_t>::getGradientInPlaceImpl(DataContainer<data_t>& Rx)
    {
        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t temp = Rx[i];
            if (_r)
                temp += (*_r)[i];

            Rx[i] = 1 - (*_y)[i] / temp;
        }
    }

    template <typename data_t>
    LinearOperator<data_t>
        EmissionLogLikelihood<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        DataContainer<data_t> scaleFactors(Rx.getDataDescriptor());
        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t temp = Rx[i];
            if (_r)
                temp += (*_r)[i];

            scaleFactors[i] = (*_y)[i] / (temp * temp);
        }

        return leaf(Scaling<data_t>(Rx.getDataDescriptor(), scaleFactors));
    }

    template <typename data_t>
    EmissionLogLikelihood<data_t>* EmissionLogLikelihood<data_t>::cloneImpl() const
    {
        if (_r)
            return new EmissionLogLikelihood<data_t>(this->getResidual(), *_y, *_r);
        else
            return new EmissionLogLikelihood<data_t>(this->getResidual(), *_y);
    }

    template <typename data_t>
    bool EmissionLogLikelihood<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherELL = dynamic_cast<const EmissionLogLikelihood*>(&other);
        if (!otherELL)
            return false;

        if (*_y != *otherELL->_y)
            return false;

        if (_r && *_r != *otherELL->_r)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class EmissionLogLikelihood<float>;
    template class EmissionLogLikelihood<double>;
    // no complex instantiations, they make no sense

} // namespace elsa
