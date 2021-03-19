#include "DataHandlerMapGPU.h"
#include "DataHandlerGPU.h"

namespace elsa
{
    template <typename data_t>
    DataHandlerMapGPU<data_t>::DataHandlerMapGPU(DataHandlerGPU<data_t>* dataOwner, data_t* data,
                                                 index_t n)
        : _map(data, static_cast<size_t>(n)), _dataOwner{dataOwner}
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
    DataHandlerMapGPU<data_t>::DataHandlerMapGPU(const DataHandlerMapGPU<data_t>& other)
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
    DataHandlerMapGPU<data_t>::~DataHandlerMapGPU()
    {
        // remove self from list of Maps referring to the _dataOwner
        if (_dataOwner) {
#pragma omp critical
            _dataOwner->_associatedMaps.erase(_handle);
        }
    }

    template <typename data_t>
    index_t DataHandlerMapGPU<data_t>::getSize() const
    {
        return static_cast<index_t>(_map.size());
    }

    template <typename data_t>
    data_t& DataHandlerMapGPU<data_t>::operator[](index_t index)
    {
        _dataOwner->detach();
        return _map[static_cast<size_t>(index)];
    }

    template <typename data_t>
    const data_t& DataHandlerMapGPU<data_t>::operator[](index_t index) const
    {
        return _map[static_cast<size_t>(index)];
    }

    template <typename data_t>
    data_t DataHandlerMapGPU<data_t>::dot(const DataHandler<data_t>& v) const
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandlerMapGPU: dot product argument has wrong size");

        // use quickvec if the other handler is GPU or GPU map, otherwise use the slow fallback
        // version
        if (auto otherHandler = dynamic_cast<const DataHandlerGPU<data_t>*>(&v)) {
            return _map.dot(*otherHandler->_data);
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapGPU*>(&v)) {
            return _map.dot(otherHandler->_map);
        } else {
            return this->slowDotProduct(v);
        }
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerMapGPU<data_t>::squaredL2Norm() const
    {
        return _map.squaredL2Norm();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerMapGPU<data_t>::l2Norm() const
    {
        return _map.l2Norm();
    }

    template <typename data_t>
    index_t DataHandlerMapGPU<data_t>::l0PseudoNorm() const
    {
        return _map.l0PseudoNorm();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerMapGPU<data_t>::l1Norm() const
    {
        return _map.l1Norm();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerMapGPU<data_t>::lInfNorm() const
    {
        return _map.lInfNorm();
    }

    template <typename data_t>
    data_t DataHandlerMapGPU<data_t>::sum() const
    {
        return _map.sum();
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapGPU<data_t>::operator+=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: addition argument has wrong size");

        _dataOwner->detach();

        // use quickvec if the other handler is GPU or GPU map, otherwise use the slow fallback
        // version
        if (auto otherHandler = dynamic_cast<const DataHandlerGPU<data_t>*>(&v)) {
            _map += *otherHandler->_data;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapGPU*>(&v)) {
            _map += otherHandler->_map;
        } else {
            this->slowAddition(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapGPU<data_t>::operator-=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: subtraction argument has wrong size");

        _dataOwner->detach();

        // use quickvec if the other handler is GPU or GPU map, otherwise use the slow fallback
        // version
        if (auto otherHandler = dynamic_cast<const DataHandlerGPU<data_t>*>(&v)) {
            _map -= *otherHandler->_data;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapGPU*>(&v)) {
            _map -= otherHandler->_map;
        } else {
            this->slowSubtraction(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapGPU<data_t>::operator*=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: multiplication argument has wrong size");

        _dataOwner->detach();

        // use quickvec if the other handler is GPU or GPU map, otherwise use the slow fallback
        // version
        if (auto otherHandler = dynamic_cast<const DataHandlerGPU<data_t>*>(&v)) {
            _map *= *otherHandler->_data;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapGPU*>(&v)) {
            _map *= otherHandler->_map;
        } else {
            this->slowMultiplication(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapGPU<data_t>::operator/=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: division argument has wrong size");

        _dataOwner->detach();

        // use quickvec if the other handler is GPU or GPU map, otherwise use the slow fallback
        // version
        if (auto otherHandler = dynamic_cast<const DataHandlerGPU<data_t>*>(&v)) {
            _map /= *otherHandler->_data;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapGPU*>(&v)) {
            _map /= otherHandler->_map;
        } else {
            this->slowDivision(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandlerMapGPU<data_t>&
        DataHandlerMapGPU<data_t>::operator=(const DataHandlerMapGPU<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: assignment argument has wrong size");

        if (getSize() == _dataOwner->getSize() && v.getSize() == v._dataOwner->getSize()) {
            _dataOwner->attach(v._dataOwner->_data);
        } else {
            _dataOwner->detachWithUninitializedBlock(
                _map._data.get() - _dataOwner->_data->_data.get(), getSize());
            _map = v._map;
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapGPU<data_t>::operator+=(data_t scalar)
    {
        _dataOwner->detach();

        _map += scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapGPU<data_t>::operator-=(data_t scalar)
    {
        _dataOwner->detach();

        _map -= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapGPU<data_t>::operator*=(data_t scalar)
    {
        _dataOwner->detach();

        _map *= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapGPU<data_t>::operator/=(data_t scalar)
    {
        _dataOwner->detach();

        _map /= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerMapGPU<data_t>::operator=(data_t scalar)
    {
        _dataOwner->detach();

        _map = scalar;
        return *this;
    }

    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>>
        DataHandlerMapGPU<data_t>::getBlock(index_t startIndex, index_t numberOfElements)
    {
        if (startIndex >= getSize() || numberOfElements > getSize() - startIndex)
            throw std::invalid_argument("DataHandler: requested block out of bounds");

        return std::unique_ptr<DataHandlerMapGPU<data_t>>(
            new DataHandlerMapGPU{_dataOwner, _map._data.get() + startIndex, numberOfElements});
    }

    template <typename data_t>
    std::unique_ptr<const DataHandler<data_t>>
        DataHandlerMapGPU<data_t>::getBlock(index_t startIndex, index_t numberOfElements) const
    {
        if (startIndex >= getSize() || numberOfElements > getSize() - startIndex)
            throw std::invalid_argument("DataHandler: requested block out of bounds");

        // using a const_cast here is fine as long as the DataHandlers never expose the internal
        // Eigen objects
        auto mutableData = const_cast<data_t*>(_map._data.get() + startIndex);
        return std::unique_ptr<const DataHandlerMapGPU<data_t>>(
            new DataHandlerMapGPU{_dataOwner, mutableData, numberOfElements});
    }

    template <typename data_t>
    DataHandlerGPU<data_t>* DataHandlerMapGPU<data_t>::cloneImpl() const
    {
        if (getSize() == _dataOwner->getSize()) {
            return new DataHandlerGPU<data_t>{*_dataOwner};
        } else {
            return new DataHandlerGPU<data_t>{_map};
        }
    }

    template <typename data_t>
    bool DataHandlerMapGPU<data_t>::isEqual(const DataHandler<data_t>& other) const
    {
        if (auto otherHandler = dynamic_cast<const DataHandlerMapGPU*>(&other)) {

            if (_map.size() != otherHandler->_map.size())
                return false;

            if (_map._data.get() != otherHandler->_map._data.get() && _map != otherHandler->_map) {
                return false;
            }

            return true;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerGPU<data_t>*>(&other)) {

            if (_map.size() != otherHandler->_data->size())
                return false;

            if (_map._data.get() != otherHandler->_data->_data.get()
                && _map != *otherHandler->_data)
                return false;

            return true;
        } else
            return false;
    }

    template <typename data_t>
    void DataHandlerMapGPU<data_t>::assign(const DataHandler<data_t>& other)
    {

        if (auto otherHandler = dynamic_cast<const DataHandlerMapGPU*>(&other)) {
            if (getSize() == _dataOwner->getSize()
                && otherHandler->getSize() == otherHandler->_dataOwner->getSize()) {
                _dataOwner->attach(otherHandler->_dataOwner->_data);
            } else {
                _dataOwner->detachWithUninitializedBlock(
                    _map._data.get() - _dataOwner->_data->_data.get(), getSize());
                _map = otherHandler->_map;
            }
        } else if (auto otherHandler = dynamic_cast<const DataHandlerGPU<data_t>*>(&other)) {
            if (getSize() == _dataOwner->getSize()) {
                _dataOwner->attach(otherHandler->_data);
            } else {
                _dataOwner->detachWithUninitializedBlock(
                    _map._data.get() - _dataOwner->_data->_data.get(), getSize());
                _map = *otherHandler->_data;
            }
        } else
            this->slowAssign(other);
    }

    template <typename data_t>
    void DataHandlerMapGPU<data_t>::assign(DataHandler<data_t>&& other)
    {
        assign(other);
    }

    template <typename data_t>
    quickvec::Vector<data_t> DataHandlerMapGPU<data_t>::accessData()
    {
        _dataOwner->detach();
        return _map;
    }

    template <typename data_t>
    quickvec::Vector<data_t> DataHandlerMapGPU<data_t>::accessData() const
    {
        return _map;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DataHandlerMapGPU<float>;
    template class DataHandlerMapGPU<std::complex<float>>;
    template class DataHandlerMapGPU<double>;
    template class DataHandlerMapGPU<std::complex<double>>;
    template class DataHandlerMapGPU<index_t>;

} // namespace elsa
