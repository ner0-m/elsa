#include "FiniteDifferences.h"
#include "Timer.h"

namespace elsa
{
    template <typename data_t>
    FiniteDifferences<data_t>::FiniteDifferences(const DataDescriptor& domainDescriptor,
                                                 DiffType type)
        : FiniteDifferences(domainDescriptor,
                            BooleanVector_t::Ones(domainDescriptor.getNumberOfDimensions()), type)
    {
    }

    template <typename data_t>
    FiniteDifferences<data_t>::FiniteDifferences(const DataDescriptor& domainDescriptor,
                                                 const BooleanVector_t& activeDims, DiffType type)
        : LinearOperator<data_t>(domainDescriptor,
                                 domainDescriptor), // setting range in body of constructor
          _activeDims{activeDims},
          _type{type}
    {
        // build the range descriptor of appropriate size
        IndexVector_t coefficients(domainDescriptor.getNumberOfDimensions() + 1);
        coefficients << domainDescriptor.getNumberOfCoefficientsPerDimension(),
            activeDims.cast<index_t>().sum();

        RealVector_t spacing(domainDescriptor.getNumberOfDimensions() + 1);
        spacing << domainDescriptor.getSpacingPerDimension(), 1;

        this->_rangeDescriptor = std::make_unique<DataDescriptor>(coefficients, spacing);

        precomputeHelpers();
    }

    template <typename data_t>
    void FiniteDifferences<data_t>::precomputeHelpers()
    {
        IndexVector_t numberOfCoefficients =
            this->_rangeDescriptor->getNumberOfCoefficientsPerDimension();

        index_t deltaTmp = 1;
        int count = -1;
        for (index_t ic = 0; ic < this->getDomainDescriptor().getNumberOfDimensions(); ++ic) {
            _coordDiff.push_back(numberOfCoefficients.head(ic).prod());

            deltaTmp *= numberOfCoefficients[ic];
            _coordDelta.push_back(deltaTmp);

            if (_activeDims[ic])
                ++count;
            _dimCounter.push_back(count);
        }
    }

    template <typename data_t>
    void FiniteDifferences<data_t>::_apply(const DataContainer<data_t>& x,
                                           DataContainer<data_t>& Ax) const
    {
        Timer<> timeguard("FiniteDifferences", "apply");

        switch (_type) {
            case DiffType::FORWARD:
                applyHelper(x, Ax, DiffType::FORWARD);
                break;
            case DiffType::BACKWARD:
                applyHelper(x, Ax, DiffType::BACKWARD);
                break;
            case DiffType::CENTRAL:
                applyHelper(x, Ax, DiffType::CENTRAL);
                break;
            default:
                throw std::logic_error("FiniteDifferences::apply: invalid DiffType");
        }
    }

    template <typename data_t>
    void FiniteDifferences<data_t>::_applyAdjoint(const DataContainer<data_t>& y,
                                                  DataContainer<data_t>& Aty) const
    {
        Timer<> timeguard("FiniteDifferences", "applyAdjoint");

        switch (_type) {
            case DiffType::FORWARD:
                applyAdjointHelper(y, Aty, DiffType::FORWARD);
                break;
            case DiffType::BACKWARD:
                applyAdjointHelper(y, Aty, DiffType::BACKWARD);
                break;
            case DiffType::CENTRAL:
                applyAdjointHelper(y, Aty, DiffType::CENTRAL);
                break;
            default:
                throw std::logic_error("FiniteDifferences::applyAdjoint: invalid DiffType");
        }
    }

    template <typename data_t>
    template <typename FDtype>
    void FiniteDifferences<data_t>::applyHelper(const DataContainer<data_t>& x,
                                                DataContainer<data_t>& Ax, FDtype type) const
    {
        index_t sizeOfDomain = this->getDomainDescriptor().getNumberOfCoefficients();
        index_t numDim = this->getDomainDescriptor().getNumberOfDimensions();

        IndexVector_t numberOfCoefficients =
            this->getRangeDescriptor().getNumberOfCoefficientsPerDimension();
        IndexVector_t decrementedCoefficients =
            numberOfCoefficients
            - IndexVector_t::Ones(this->getRangeDescriptor().getNumberOfDimensions());

#pragma omp parallel
        for (int currDim = 0; currDim < numDim; ++currDim) {
            if (!_activeDims[currDim])
                continue;

            index_t modulus = numberOfCoefficients.head(currDim + 1).prod();
            index_t divisor = numberOfCoefficients.head(currDim).prod();

#pragma omp for nowait
            for (index_t id = 0; id < sizeOfDomain; ++id) {
                index_t icCount = (id % modulus) / divisor; //_domainDescriptor.index(id, ic);
                index_t ir = id + _dimCounter[currDim] * _coordDelta[currDim];

                // store result depending on mode
                switch (type) {
                    case DiffType::FORWARD:
                        Ax[ir] = -x[id];
                        if (icCount < decrementedCoefficients[currDim])
                            Ax[ir] += x[id + _coordDiff[currDim]];
                        break;
                    case DiffType::BACKWARD:
                        Ax[ir] = x[id];
                        if (icCount > 0)
                            Ax[ir] -= x[id - _coordDiff[currDim]];
                        break;
                    case DiffType::CENTRAL:
                        Ax[ir] = static_cast<data_t>(0.0);
                        if (icCount < decrementedCoefficients[currDim])
                            Ax[ir] += static_cast<data_t>(0.5) * x[id + _coordDiff[currDim]];
                        if (icCount > 0)
                            Ax[ir] -= static_cast<data_t>(0.5) * x[id - _coordDiff[currDim]];
                        break;
                }
            }
        }
    }

    template <typename data_t>
    template <typename FDtype>
    void FiniteDifferences<data_t>::applyAdjointHelper(const DataContainer<data_t>& y,
                                                       DataContainer<data_t>& Aty,
                                                       FDtype type) const
    {
        index_t sizeOfDomain = this->getDomainDescriptor().getNumberOfCoefficients();
        index_t numDim = this->getDomainDescriptor().getNumberOfDimensions();

        IndexVector_t numberOfCoefficients =
            this->getDomainDescriptor().getNumberOfCoefficientsPerDimension();
        IndexVector_t decrementedCoefficients = numberOfCoefficients - IndexVector_t::Ones(numDim);

#pragma omp parallel
        for (index_t currDim = 0; currDim < numDim; ++currDim) {
            if (!_activeDims[currDim])
                continue;

            index_t modulus = numberOfCoefficients.head(currDim + 1).prod();
            index_t divisor = numberOfCoefficients.head(currDim).prod();

#pragma omp for nowait
            for (index_t id = 0; id < sizeOfDomain; ++id) {
                index_t icCount = (id % modulus) / divisor;
                index_t ir = id + _dimCounter[currDim] * _coordDelta[currDim];

                switch (type) {
                    case DiffType::FORWARD:
                        if (icCount > 0)
                            Aty[id] += y[ir - _coordDiff[currDim]];

                        Aty[id] -= y[ir];
                        break;
                    case DiffType::BACKWARD:
                        if (icCount < decrementedCoefficients(currDim))
                            Aty[id] -= y[ir + _coordDiff[currDim]];

                        Aty[id] += y[ir];
                        break;
                    case DiffType::CENTRAL:
                        if (icCount > 0)
                            Aty[ir] += static_cast<data_t>(0.5) * y[ir - _coordDiff[currDim]];

                        if (icCount < decrementedCoefficients(currDim))
                            Aty[ir] -= static_cast<data_t>(0.5) * y[ir + _coordDiff[currDim]];
                        break;
                }
            }
        }
    }

    template <typename data_t>
    FiniteDifferences<data_t>* FiniteDifferences<data_t>::cloneImpl() const
    {
        return new FiniteDifferences(this->getDomainDescriptor(), _activeDims, _type);
    }

    template <typename data_t>
    bool FiniteDifferences<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        if (!LinearOperator<data_t>::isEqual(other))
            return false;

        auto otherFD = dynamic_cast<const FiniteDifferences*>(&other);
        if (!otherFD)
            return false;

        if (_type != otherFD->_type || _activeDims != otherFD->_activeDims)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class FiniteDifferences<float>;
    template class FiniteDifferences<double>;
    template class FiniteDifferences<std::complex<float>>;
    template class FiniteDifferences<std::complex<double>>;

} // namespace elsa
