#include "Convolution.h"

#include "Error.h"

#include <complex>
#include "DataContainer.h"
#include "FourierTransform.h"
#include "LinearOperator.h"

namespace elsa
{
    template <typename data_t>
    Convolution<data_t>::Convolution(const DataDescriptor& domainDescriptor,
                                     const ConvolutionFilter<data_t>& kernel, Type method)
        : B(domainDescriptor, domainDescriptor), method{method}
    {
        // TODO create kernel, fft it, then pad it with zeroes, multiply.
    }

    template <typename data_t>
    void Convolution<data_t>::applyImpl(const DataContainer<data_t>& x,
                                        DataContainer<data_t>& Ax) const
    {
        switch (this->method) {
        case Type::FFT: {
            FourierTransform<data_t> fft{this->_domainDescriptor};
            Ax = fft.apply(x);
            Ax = this->kernel->apply(Ax);
            Ax = fft.applyAdjoint(Ax);
        }
        default:
            throw elsa::Error{"internal error"};
        }
    }

    template <typename data_t>
    void Convolution<data_t>::applyAdjointImpl(const DataContainer<data_t>& x,
                                               DataContainer<data_t>& Atx) const
    {
        switch (this->method) {
        case Type::FFT: {
            FourierTransform<data_t> fft{this->_domainDescriptor};
            Atx = fft.apply(x);
            Atx = this->kernel->applyInverse(Atx);
            Atx = fft.apply(Atx);
        }
        default:
            throw elsa::Error{"internal error"};
        }
    }

    template <typename data_t>
    Convolution* Convolution<data_t>::cloneImpl() const
    {
        auto& domainDescriptor = static_cast<const DataDescriptor&>(*this->_domainDescriptor);
        return new Convolution(domainDescriptor, this->kernel, this->method);
    };

    template <typename data_t>
    bool Convolution<data_t>::isEqual(const B& other) const;

    template class Convolution<complex<float>>;
    template class Convolution<complex<double>>;

} // namespace elsa
