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
              _dataHandler{other._dataHandler}
    {}

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator=(const DataContainer<data_t>& other)
    {
        if (this != &other) {
            _dataDescriptor = other._dataDescriptor->clone();
            _dataHandler = other._dataHandler;
        }

        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(DataContainer<data_t> &&other) noexcept
        : _dataDescriptor{std::move(other._dataDescriptor)},
          _dataHandler{std::move(other._dataHandler)}
    {
        // leave other in a valid state
        other._dataDescriptor = nullptr;
        other._dataHandler = nullptr;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator=(DataContainer<data_t>&& other) noexcept
    {
        _dataDescriptor = std::move(other._dataDescriptor);
        _dataHandler = std::move(other._dataHandler);

        // leave other in a valid state
        other._dataDescriptor = nullptr;
        other._dataHandler = nullptr;

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
        detach();
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
        detach();
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
        detach();
        *_dataHandler += *dc._dataHandler;
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator-=(const DataContainer<data_t>& dc)
    {
        detach();
        *_dataHandler -= *dc._dataHandler;
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator*=(const DataContainer<data_t>& dc)
    {
        detach();
        *_dataHandler *= *dc._dataHandler;
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator/=(const DataContainer<data_t>& dc)
    {
        detach();
        *_dataHandler /= *dc._dataHandler;
        return *this;
    }


    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator+=(data_t scalar)
    {
        detach();
        *_dataHandler += scalar;
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator-=(data_t scalar)
    {
        detach();
        *_dataHandler -= scalar;
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator*=(data_t scalar)
    {
        detach();
        *_dataHandler *= scalar;
        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator/=(data_t scalar)
    {
        detach();
        *_dataHandler /= scalar;
        return *this;
    }


    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator=(data_t scalar)
    {
        detach();
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

    template <typename data_t>
    void DataContainer<data_t>::detach()
    {
        if (_dataHandler.use_count() != 1) {
#pragma omp barrier
#pragma omp single
            _dataHandler = _dataHandler->clone();
        }
        return;
    }

    template <typename data_t>
    typename DataContainer<data_t>::iterator DataContainer<data_t>::begin()
    {
        detach();
        return iterator(&(*this)[0]);
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_iterator DataContainer<data_t>::begin() const
    {
        return cbegin();
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_iterator DataContainer<data_t>::cbegin() const
    {
        return const_iterator(&(*this)[0]);
    }

    template <typename data_t>
    typename DataContainer<data_t>::iterator DataContainer<data_t>::end()
    {
        detach();
        return iterator(&(*this)[0] + getSize());
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_iterator DataContainer<data_t>::end() const
    {
        return cend();
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_iterator DataContainer<data_t>::cend() const
    {
        return const_iterator(&(*this)[0] + getSize());
    }

    template <typename data_t>
    typename DataContainer<data_t>::reverse_iterator DataContainer<data_t>::rbegin()
    {
        detach();
        return reverse_iterator(end());
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_reverse_iterator DataContainer<data_t>::rbegin() const
    {
        return crbegin();
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_reverse_iterator DataContainer<data_t>::crbegin() const
    {
        return const_reverse_iterator(cend());
    }

    template <typename data_t>
    typename DataContainer<data_t>::reverse_iterator DataContainer<data_t>::rend()
    {
        detach();
        return reverse_iterator(begin());
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_reverse_iterator DataContainer<data_t>::rend() const
    {
        return crend();
    }

    template <typename data_t>
    typename DataContainer<data_t>::const_reverse_iterator DataContainer<data_t>::crend() const
    {
        return const_reverse_iterator(cbegin());
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DataContainer<float>;
    template class DataContainer<std::complex<float>>;
    template class DataContainer<double>;
    template class DataContainer<std::complex<double>>;
    template class DataContainer<index_t>;

} // namespace elsa
