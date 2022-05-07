#include "Scaling.h"
#include "Timer.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    Scaling<data_t>::Scaling(const DataDescriptor& descriptor, data_t scaleFactor)
        : LinearOperator<data_t>(descriptor, descriptor),
          _isIsotropic{true},
          _scaleFactor{scaleFactor}
    {
    }

    template <typename data_t>
    Scaling<data_t>::Scaling(const DataDescriptor& domainDescriptor,
                             const DataDescriptor& rangeDescriptor, data_t scaleFactor)
        : LinearOperator<data_t>(domainDescriptor, rangeDescriptor),
          _isIsotropic{true},
          _scaleFactor{scaleFactor}
    {
    }

    template <typename data_t>
    Scaling<data_t>::Scaling(const DataDescriptor& descriptor,
                             const DataContainer<data_t>& scaleFactors)
        : LinearOperator<data_t>(descriptor, descriptor),
          _isIsotropic{false},
          _scaleFactors{std::make_unique<DataContainer<data_t>>(scaleFactors)}
    {
    }

    template <typename data_t>
    bool Scaling<data_t>::isIsotropic() const
    {
        return _isIsotropic;
    }

    template <typename data_t>
    data_t Scaling<data_t>::getScaleFactor() const
    {
        if (!_isIsotropic)
            throw LogicError("Scaling: scaling is not isotropic");

        return _scaleFactor;
    }

    template <typename data_t>
    const DataContainer<data_t>& Scaling<data_t>::getScaleFactors() const
    {
        if (_isIsotropic)
            throw LogicError("Scaling: scaling is isotropic");

        return *_scaleFactors;
    }

    template <typename data_t>
    void Scaling<data_t>::applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const
    {
        Timer timeguard("Scaling", "apply");

        if (_isIsotropic)
            Ax = _scaleFactor * x;
        else
            Ax = *_scaleFactors * x;
    }

    template <typename data_t>
    void Scaling<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                           DataContainer<data_t>& Aty) const
    {
        Timer timeguard("Scaling", "applyAdjoint");

        if (_isIsotropic)
            Aty = _scaleFactor * y;
        else
            Aty = *_scaleFactors * y;
    }

    template <typename data_t>
    Scaling<data_t>* Scaling<data_t>::cloneImpl() const
    {
        if (_isIsotropic)
            return new Scaling(this->getDomainDescriptor(), _scaleFactor);
        else
            return new Scaling(this->getDomainDescriptor(), *_scaleFactors);
    }

    template <typename data_t>
    bool Scaling<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherScaling = downcast_safe<Scaling>(&other);
        if (!otherScaling)
            return false;

        if (_isIsotropic != otherScaling->_isIsotropic)
            return false;

        if (_isIsotropic && _scaleFactor != otherScaling->_scaleFactor)
            return false;

        if (!_isIsotropic && *_scaleFactors != *otherScaling->_scaleFactors)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Scaling<float>;
    template class Scaling<complex<float>>;
    template class Scaling<double>;
    template class Scaling<complex<double>>;

} // namespace elsa
