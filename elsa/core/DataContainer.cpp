#include "DataContainer.h"
#include "DataHandlerCPU.h"

#include <stdexcept>
#include <utility>

namespace elsa {

    template<typename data_t>
    DataContainer<data_t>::DataContainer(const DataDescriptor &dataDescriptor, DataHandlerType handlerType)
            : _dataDescriptor{dataDescriptor.clone()},
              _dataHandler{createDataHandler(handlerType, _dataDescriptor->getNumberOfCoefficients())} 
    {}
    
    template <typename data_t>
    DataContainer<data_t>::DataContainer(const DataDescriptor& dataDescriptor, const  Eigen::Matrix<data_t, Eigen::Dynamic, 1>& data,
            DataHandlerType handlerType)
            : _dataDescriptor{dataDescriptor.clone()},
              _dataHandler{createDataHandler(handlerType, _dataDescriptor->getNumberOfCoefficients())}
    {
        if (_dataHandler->getSize() != data.size())
            throw std::invalid_argument("DataContainer: initialization vector has invalid size");
        
        for (index_t i = 0; i < _dataHandler->getSize(); ++i)
            (*_dataHandler)[i] = data[i];
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(const DataContainer<data_t> &other)
            : _dataDescriptor{other._dataDescriptor->clone()},
              _dataHandler{other._dataHandler->clone()} 
    {}

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator=(const DataContainer<data_t>& other)
    {
        if (this != &other) {
            _dataDescriptor = other._dataDescriptor->clone();
            _dataHandler = other._dataHandler->clone();
        }

        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(DataContainer<data_t> &&other)
        : _dataDescriptor{std::move(other._dataDescriptor)},
          _dataHandler{std::move(other._dataHandler)}
    {
        // make sure to leave other in a valid state (since we do not check for empty pointers!)
        IndexVector_t numCoeff(1); numCoeff << 1;
        other._dataDescriptor = std::make_unique<DataDescriptor>(numCoeff);
        other._dataHandler = createDataHandler(DataHandlerType::CPU, other._dataDescriptor->getNumberOfCoefficients());
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator=(DataContainer<data_t>&& other)
    {
        _dataDescriptor = std::move(other._dataDescriptor);
        _dataHandler = std::move(other._dataHandler);

        // make sure to leave other in a valid state (since we do not check for empty pointers!)
        IndexVector_t numCoeff(1); numCoeff << 1;
        other._dataDescriptor = std::make_unique<DataDescriptor>(numCoeff);
        other._dataHandler = createDataHandler(DataHandlerType::CPU, other._dataDescriptor->getNumberOfCoefficients());

        return *this;
    }



    template<typename data_t>
    const DataDescriptor &DataContainer<data_t>::getDataDescriptor() const
    {
        return *_dataDescriptor;
    }

    template<typename data_t>
    index_t DataContainer<data_t>::getSize() const
    {
        return _dataHandler->getSize();
    }


    template<typename data_t>
    data_t &DataContainer<data_t>::operator[](index_t index)
    {
        return (*_dataHandler)[index];
    }

    template<typename data_t>
    const data_t &DataContainer<data_t>::operator[](index_t index) const
    {
        return (*_dataHandler)[index];
    }

    template<typename data_t>
    data_t &DataContainer<data_t>::operator()(IndexVector_t coordinate)
    {
        return (*_dataHandler)[_dataDescriptor->getIndexFromCoordinate(coordinate)];
    }

    template<typename data_t>
    const data_t &DataContainer<data_t>::operator()(IndexVector_t coordinate) const
    {
        return (*_dataHandler)[_dataDescriptor->getIndexFromCoordinate(coordinate)];
    }


    template<typename data_t>
    data_t DataContainer<data_t>::dot(const DataContainer<data_t> &other) const
    {
        return _dataHandler->dot(*other._dataHandler);
    }

    template<typename data_t>
    data_t DataContainer<data_t>::squaredL2Norm() const
    {
        return _dataHandler->squaredL2Norm();
    }

    template <typename data_t>
    data_t DataContainer<data_t>::l1Norm() const
    {
        return _dataHandler->l1Norm();
    }

    template <typename data_t>
    data_t DataContainer<data_t>::lInfNorm() const
    {
        return _dataHandler->lInfNorm();
    }

    template<typename data_t>
    data_t DataContainer<data_t>::sum() const
    {
        return _dataHandler->sum();
    }


    template<typename data_t>
    DataContainer<data_t> DataContainer<data_t>::square() const
    {
        return DataContainer<data_t>(*_dataDescriptor, _dataHandler->square());
    }

    template<typename data_t>
    DataContainer<data_t> DataContainer<data_t>::sqrt() const
    {
        return DataContainer<data_t>(*_dataDescriptor, _dataHandler->sqrt());
    }

    template<typename data_t>
    DataContainer<data_t> DataContainer<data_t>::exp() const
    {
        return DataContainer<data_t>(*_dataDescriptor, _dataHandler->exp());
    }

    template<typename data_t>
    DataContainer<data_t> DataContainer<data_t>::log() const
    {
        return DataContainer<data_t>(*_dataDescriptor, _dataHandler->log());
    }


    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator+=(const DataContainer<data_t>& dc)
    {
        *_dataHandler += *dc._dataHandler;
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator-=(const DataContainer<data_t>& dc)
    {
        *_dataHandler -= *dc._dataHandler;
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator*=(const DataContainer<data_t>& dc)
    {
        *_dataHandler *= *dc._dataHandler;
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator/=(const DataContainer<data_t>& dc)
    {
        *_dataHandler /= *dc._dataHandler;
        return *this;
    }


    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator+=(data_t scalar)
    {
        *_dataHandler += scalar;
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator-=(data_t scalar)
    {
        *_dataHandler -= scalar;
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator*=(data_t scalar)
    {
        *_dataHandler *= scalar;
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator/=(data_t scalar)
    {
        *_dataHandler /= scalar;
        return *this;
    }


    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator=(data_t scalar)
    {
        *_dataHandler = scalar;
        return *this;
    }


    template<typename data_t>
    template<typename ... Args>
    std::unique_ptr<DataHandler<data_t>>
    DataContainer<data_t>::createDataHandler(DataHandlerType handlerType, Args &&... args) {
        switch (handlerType) {
            case DataHandlerType::CPU:
                return std::make_unique<DataHandlerCPU<data_t>>(std::forward<Args>(args)...);
            default:
                throw std::invalid_argument("DataContainer: unknown handler type");
        }
    }


    template<typename data_t>
    DataContainer<data_t>::DataContainer(const DataDescriptor &dataDescriptor,
                                         std::unique_ptr<DataHandler<data_t>> dataHandler)
            : _dataDescriptor{dataDescriptor.clone()},
              _dataHandler{std::move(dataHandler)}
    {
    }


    template <typename data_t>
    bool DataContainer<data_t>::operator==(const DataContainer<data_t>& other) const
    {
        if (*_dataDescriptor != *other._dataDescriptor)
            return false;

        if (*_dataHandler != *other._dataHandler)
            return false;

        return true;
    }

    template <typename data_t>
    bool DataContainer<data_t>::operator!=(const DataContainer<data_t>& other) const
    {
        return !(*this == other);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DataContainer<float>;
    template class DataContainer<std::complex<float>>;
    template class DataContainer<double>;
    template class DataContainer<std::complex<double>>;
    template class DataContainer<index_t>;

} // namespace elsa
