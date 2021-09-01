#include "DataHandlerMapCPU.h"
#include "DataHandlerCPU.h"
#include "TypeCasts.hpp"

namespace elsa
{
    template <typename data_t>
    DataHandlerMapCPU<data_t>::DataHandlerMapCPU(DataHandlerCPU<data_t>* dataOwner, data_t* data,
                                                 index_t n)
        : _map(data, n), _dataOwner{dataOwner}
    {
        // sanity checks performed in getBlock()
#pragma omp critical
        {
            // add self to list of Maps referring to the _dataOwner
            _dataOwner->_associatedMaps.push_front(this);
            _handle = _dataOwner->_associatedMaps.begin();
        }
    }

    template <typename data_t>
    DataHandlerMapCPU<data_t>::DataHandlerMapCPU(const DataHandlerMapCPU<data_t>& other)
        : _map{other._map}, _dataOwner{other._dataOwner}
    {
#pragma omp critical
        {
            // add self to list of Maps referring to the _dataOwner
            _dataOwner->_associatedMaps.push_front(this);
            _handle = _dataOwner->_associatedMaps.begin();
        }
    }

    template <typename data_t>
    DataHandlerMapCPU<data_t>::~DataHandlerMapCPU()
    {
        // remove self from list of Maps referring to the _dataOwner
        if (_dataOwner) {
#pragma omp critical
            _dataOwner->_associatedMaps.erase(_handle);
        }
    }

    template <typename data_t>
    index_t DataHandlerMapCPU<data_t>::getSize() const
    {
        return static_cast<index_t>(_map.size());
    }

    template <typename data_t>
    data_t& DataHandlerMapCPU<data_t>::operator[](index_t index)
    {
        _dataOwner->detach();
        return _map[index];
    }

    template <typename data_t>
    const data_t& DataHandlerMapCPU<data_t>::operator[](index_t index) const
    {
        return _map[index];
    }

    template <typename data_t>
    data_t DataHandlerMapCPU<data_t>::dot(const DataHandler<data_t>& v) const
    {
        if (v.getSize() != getSize())
            throw InvalidArgumentError("DataHandlerMapCPU: dot product argument has wrong size");

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = downcast_safe<DataHandlerCPU<data_t>>(&v)) {
            return _map.dot(*otherHandler->_data);
        } else if (auto otherHandler = downcast_safe<DataHandlerMapCPU>(&v)) {
            return _map.dot(otherHandler->_map);
        } else {
            return this->slowDotProduct(v);
        }
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerMapCPU<data_t>::squaredL2Norm() const
    {
        return _map.squaredNorm();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerMapCPU<data_t>::l2Norm() const
    {
        return _map.norm();
    }

    template <typename data_t>
    index_t DataHandlerMapCPU<data_t>::l0PseudoNorm() const
    {
        using FloatType = GetFloatingPointType_t<data_t>;
        return (_map.array().cwiseAbs() >= std::numeric_limits<FloatType>::epsilon()).count();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerMapCPU<data_t>::l1Norm() const
    {
        return _map.array().abs().sum();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerMapCPU<data_t>::lInfNorm() const
    {
        return _map.array().abs().maxCoeff();
    }

    template <typename data_t>
    data_t DataHandlerMapCPU<data_t>::sum() const
    {
        return _map.sum();
    }

    template <typename data_t>
    data_t DataHandlerMapCPU<data_t>::minElement() const
    {
        if constexpr (isComplex<data_t>) {
            throw LogicError("DataHandlerCPU: minElement of complex type not supported");
        } else {
            return _map.minCoeff();
        }
    }

    template <typename data_t>
    data_t DataHandlerMapCPU<data_t>::maxElement() const
    {
        if constexpr (isComplex<data_t>) {
            throw LogicError("DataHandlerCPU: maxElement of complex type not supported");
        } else {
            return _map.maxCoeff();
        }
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapCPU<data_t>::fft(const DataDescriptor& source_desc)
    {
        // detaches internally
        this->_dataOwner->fft(source_desc);
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapCPU<data_t>::ifft(const DataDescriptor& source_desc)
    {
        // detaches internally
        this->_dataOwner->ifft(source_desc);
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapCPU<data_t>::operator+=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw InvalidArgumentError("DataHandler: addition argument has wrong size");

        _dataOwner->detach();

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = downcast_safe<DataHandlerCPU<data_t>>(&v)) {
            _map += *otherHandler->_data;
        } else if (auto otherHandler = downcast_safe<DataHandlerMapCPU>(&v)) {
            _map += otherHandler->_map;
        } else {
            this->slowAddition(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapCPU<data_t>::operator-=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw InvalidArgumentError("DataHandler: subtraction argument has wrong size");

        _dataOwner->detach();

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = downcast_safe<DataHandlerCPU<data_t>>(&v)) {
            _map -= *otherHandler->_data;
        } else if (auto otherHandler = downcast_safe<DataHandlerMapCPU>(&v)) {
            _map -= otherHandler->_map;
        } else {
            this->slowSubtraction(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapCPU<data_t>::operator*=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw InvalidArgumentError("DataHandler: multiplication argument has wrong size");

        _dataOwner->detach();

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = downcast_safe<DataHandlerCPU<data_t>>(&v)) {
            _map.array() *= otherHandler->_data->array();
        } else if (auto otherHandler = downcast_safe<DataHandlerMapCPU>(&v)) {
            _map.array() *= otherHandler->_map.array();
        } else {
            this->slowMultiplication(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapCPU<data_t>::operator/=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw InvalidArgumentError("DataHandler: division argument has wrong size");

        _dataOwner->detach();

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = downcast_safe<DataHandlerCPU<data_t>>(&v)) {
            _map.array() /= otherHandler->_data->array();
        } else if (auto otherHandler = downcast_safe<DataHandlerMapCPU>(&v)) {
            _map.array() /= otherHandler->_map.array();
        } else {
            this->slowDivision(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandlerMapCPU<data_t>&
        DataHandlerMapCPU<data_t>::operator=(const DataHandlerMapCPU<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw InvalidArgumentError("DataHandler: assignment argument has wrong size");

        if (getSize() == _dataOwner->getSize() && v.getSize() == v._dataOwner->getSize()) {
            _dataOwner->attach(v._dataOwner->_data);
        } else {
            _dataOwner->detachWithUninitializedBlock(_map.data() - _dataOwner->_data->data(),
                                                     getSize());
            _map = v._map;
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapCPU<data_t>::operator+=(data_t scalar)
    {
        _dataOwner->detach();

        _map.array() += scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapCPU<data_t>::operator-=(data_t scalar)
    {
        _dataOwner->detach();

        _map.array() -= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapCPU<data_t>::operator*=(data_t scalar)
    {
        _dataOwner->detach();

        _map.array() *= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapCPU<data_t>::operator/=(data_t scalar)
    {
        _dataOwner->detach();

        _map.array() /= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapCPU<data_t>::operator=(data_t scalar)
    {
        _dataOwner->detach();

        _map.setConstant(scalar);
        return *this;
    }

    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>>
        DataHandlerMapCPU<data_t>::getBlock(index_t startIndex, index_t numberOfElements)
    {
        if (startIndex >= getSize() || numberOfElements > getSize() - startIndex)
            throw InvalidArgumentError("DataHandler: requested block out of bounds");

        return std::unique_ptr<DataHandlerMapCPU<data_t>>(
            new DataHandlerMapCPU{_dataOwner, _map.data() + startIndex, numberOfElements});
    }

    template <typename data_t>
    std::unique_ptr<const DataHandler<data_t>>
        DataHandlerMapCPU<data_t>::getBlock(index_t startIndex, index_t numberOfElements) const
    {
        if (startIndex >= getSize() || numberOfElements > getSize() - startIndex)
            throw InvalidArgumentError("DataHandler: requested block out of bounds");

        // using a const_cast here is fine as long as the DataHandlers never expose the internal
        // Eigen objects
        auto mutableData = const_cast<data_t*>(_map.data() + startIndex);
        return std::unique_ptr<const DataHandlerMapCPU<data_t>>(
            new DataHandlerMapCPU{_dataOwner, mutableData, numberOfElements});
    }

    template <typename data_t>
    DataHandlerCPU<data_t>* DataHandlerMapCPU<data_t>::cloneImpl() const
    {
        if (getSize() == _dataOwner->getSize()) {
            return new DataHandlerCPU<data_t>{*_dataOwner};
        } else {
            return new DataHandlerCPU<data_t>{_map};
        }
    }

    template <typename data_t>
    bool DataHandlerMapCPU<data_t>::isEqual(const DataHandler<data_t>& other) const
    {
        if (auto otherHandler = downcast_safe<DataHandlerMapCPU>(&other)) {

            if (_map.size() != otherHandler->_map.size())
                return false;

            if (_map.data() != otherHandler->_map.data() && _map != otherHandler->_map) {
                return false;
            }

            return true;
        } else if (auto otherHandler = downcast_safe<DataHandlerCPU<data_t>>(&other)) {

            if (_map.size() != otherHandler->_data->size())
                return false;

            if (_map.data() != otherHandler->_data->data() && _map != *otherHandler->_data)
                return false;

            return true;
        } else
            return false;
    }

    template <typename data_t>
    void DataHandlerMapCPU<data_t>::assign(const DataHandler<data_t>& other)
    {

        if (auto otherHandler = downcast_safe<DataHandlerMapCPU>(&other)) {
            if (getSize() == _dataOwner->getSize()
                && otherHandler->getSize() == otherHandler->_dataOwner->getSize()) {
                _dataOwner->attach(otherHandler->_dataOwner->_data);
            } else {
                _dataOwner->detachWithUninitializedBlock(_map.data() - _dataOwner->_data->data(),
                                                         getSize());
                _map = otherHandler->_map;
            }
        } else if (auto otherHandler = downcast_safe<DataHandlerCPU<data_t>>(&other)) {
            if (getSize() == _dataOwner->getSize()) {
                _dataOwner->attach(otherHandler->_data);
            } else {
                _dataOwner->detachWithUninitializedBlock(_map.data() - _dataOwner->_data->data(),
                                                         getSize());
                _map = *otherHandler->_data;
            }
        } else
            this->slowAssign(other);
    }

    template <typename data_t>
    void DataHandlerMapCPU<data_t>::assign(DataHandler<data_t>&& other)
    {
        assign(other);
    }

    template <typename data_t>
    typename DataHandlerMapCPU<data_t>::DataMap_t DataHandlerMapCPU<data_t>::accessData()
    {
        _dataOwner->detach();
        return _map;
    }

    template <typename data_t>
    typename DataHandlerMapCPU<data_t>::DataMap_t DataHandlerMapCPU<data_t>::accessData() const
    {
        return _map;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DataHandlerMapCPU<float>;
    template class DataHandlerMapCPU<std::complex<float>>;
    template class DataHandlerMapCPU<double>;
    template class DataHandlerMapCPU<std::complex<double>>;
    template class DataHandlerMapCPU<index_t>;

} // namespace elsa
