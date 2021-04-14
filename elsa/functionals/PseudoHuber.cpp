#include "PseudoHuber.h"
#include "Scaling.h"

#include <cmath>
#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    PseudoHuber<data_t>::PseudoHuber(const DataDescriptor& domainDescriptor, real_t delta)
        : Functional<data_t>(domainDescriptor), _delta{delta}
    {
        // sanity check delta
        if (delta <= static_cast<real_t>(0.0))
            throw InvalidArgumentError("PseudoHuber: delta has to be positive.");
    }

    template <typename data_t>
    PseudoHuber<data_t>::PseudoHuber(const Residual<data_t>& residual, real_t delta)
        : Functional<data_t>(residual), _delta{delta}
    {
        // sanity check delta
        if (delta <= static_cast<real_t>(0.0))
            throw InvalidArgumentError("PseudoHuber: delta has to be positive.");
    }

    template <typename data_t>
    data_t PseudoHuber<data_t>::evaluateImpl(const DataContainer<data_t>& Rx)
    {
        // note: this is currently not a reduction in DataContainer, but implemented here "manually"

        auto result = static_cast<data_t>(0.0);

        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t temp = Rx[i] / _delta;
            result +=
                _delta * _delta
                * (std::sqrt(static_cast<data_t>(1.0) + temp * temp) - static_cast<data_t>(1.0));
        }

        return result;
    }

    template <typename data_t>
    void PseudoHuber<data_t>::getGradientInPlaceImpl(DataContainer<data_t>& Rx)
    {
        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t temp = Rx[i] / _delta;
            Rx[i] = Rx[i] / std::sqrt(static_cast<data_t>(1.0) + temp * temp);
        }
    }

    template <typename data_t>
    LinearOperator<data_t> PseudoHuber<data_t>::getHessianImpl(const DataContainer<data_t>& Rx)
    {
        DataContainer<data_t> scaleFactors(Rx.getDataDescriptor());
        for (index_t i = 0; i < Rx.getSize(); ++i) {
            data_t temp = Rx[i] / _delta;
            data_t tempSq = temp * temp;
            data_t sqrtOnePTempSq = std::sqrt(static_cast<data_t>(1.0) + tempSq);
            scaleFactors[i] =
                (sqrtOnePTempSq - tempSq / sqrtOnePTempSq) / (static_cast<data_t>(1.0) + tempSq);
        }

        return leaf(Scaling<data_t>(Rx.getDataDescriptor(), scaleFactors));
    }

    template <typename data_t>
    PseudoHuber<data_t>* PseudoHuber<data_t>::cloneImpl() const
    {
        return new PseudoHuber(this->getResidual(), _delta);
    }

    template <typename data_t>
    bool PseudoHuber<data_t>::isEqual(const Functional<data_t>& other) const
    {
        if (!Functional<data_t>::isEqual(other))
            return false;

        auto otherPHuber = dynamic_cast<const PseudoHuber*>(&other);
        if (!otherPHuber)
            return false;

        if (_delta != otherPHuber->_delta)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class PseudoHuber<float>;
    template class PseudoHuber<double>;
    // no complex-number instantiations for PseudoHuber! (they would not really be useful)

} // namespace elsa
