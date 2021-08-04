#include "FourierTransform.h"

#include "Error.h"
#include "Timer.h"

#include <iostream>

namespace elsa {
    template <typename data_t>
    FourierTransform<data_t>::FourierTransform(const DataDescriptor &domainDescriptor)
            : B(domainDescriptor, domainDescriptor) {
    }

    template <typename data_t>
    void FourierTransform<data_t>::applyImpl(const DataContainer<data_t>& x,
                                             DataContainer<data_t>& Ax) const
    {

        Timer timeguard("FourierTransform", "apply()");

        auto x_values = x.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        auto x_dims = x.getDataDescriptor().getNumberOfDimensions();

        // input container size must match dimensionality of operator setup
        assert(x_values.size() == this->_domainDescriptor->getNumberOfDimensions());

        // copy the input and fouriertransform it
        Ax = x;
        Ax.fft();
    }

    template <typename data_t>
    void FourierTransform<data_t>::applyAdjointImpl(const DataContainer<data_t>& x,
                                                    DataContainer<data_t>& Atx) const
    {
        Timer timeguard("FourierTransform", "applyAdjoint()");

        auto x_values = x.getDataDescriptor().getNumberOfCoefficientsPerDimension();
        auto x_dims = x.getDataDescriptor().getNumberOfDimensions();

        // input container size must match dimensionality of operator setup
        assert(x_values.size() == this->_domainDescriptor->getNumberOfDimensions());

        // copy the input and fouriertransform it
        Atx = x;
        Atx.ifft();
    }

    template <typename data_t>
    FourierTransform<data_t>* FourierTransform<data_t>::cloneImpl() const
    {
        auto& domainDescriptor = static_cast<const DataDescriptor&>(*this->_domainDescriptor);

        return new FourierTransform(domainDescriptor);
    }

    template <typename data_t>
    bool FourierTransform<data_t>::isEqual(const B& other) const
    {
        if (!B::isEqual(other))
            return false;

        auto otherOP = dynamic_cast<const FourierTransform *>(&other);
        if (!otherOP)
            return false;

        // TODO actually check for equality!
        return true;
    }

    template class FourierTransform<std::complex<float>>;
    template class FourierTransform<std::complex<double>>;
} // namespace elsa
