#include "DataHandlerRawMapCPU.h"
#include "DataHandlerMapCPU.h"
#include "DataHandlerCPU.h"

namespace elsa
{
    template <typename data_t>
    DataHandlerRawMapCPU<data_t>::DataHandlerRawMapCPU(data_t* data, index_t n) : _map(data, n)
    {
    }

    template <typename data_t>
    DataHandlerRawMapCPU<data_t>::DataHandlerRawMapCPU(const DataHandlerRawMapCPU<data_t>& other)
        : _map{other._map}
    {
    }

    template <typename data_t>
    DataHandlerRawMapCPU<data_t>::~DataHandlerRawMapCPU()
    {
    }

    template <typename data_t>
    index_t DataHandlerRawMapCPU<data_t>::getSize() const
    {
        return static_cast<index_t>(_map.size());
    }

    template <typename data_t>
    data_t& DataHandlerRawMapCPU<data_t>::operator[](index_t index)
    {
        return _map[index];
    }

    template <typename data_t>
    const data_t& DataHandlerRawMapCPU<data_t>::operator[](index_t index) const
    {
        return _map[index];
    }

    template <typename data_t>
    data_t DataHandlerRawMapCPU<data_t>::dot(const DataHandler<data_t>& v) const
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument(
                "DataHandlerRawMapCPU: dot product argument has wrong size");

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerCPU<data_t>*>(&v)) {
            return _map.dot(*otherHandler->_data);
        } else if (auto otherHandler = dynamic_cast<const DataHandlerRawMapCPU*>(&v)) {
            return _map.dot(otherHandler->_map);
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&v)) {
            return _map.dot(otherHandler->_map);
        } else {
            return this->slowDotProduct(v);
        }
    } // namespace elsa

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerRawMapCPU<data_t>::squaredL2Norm() const
    {
        return _map.squaredNorm();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerRawMapCPU<data_t>::l1Norm() const
    {
        return _map.array().abs().sum();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerRawMapCPU<data_t>::lInfNorm() const
    {
        return _map.array().abs().maxCoeff();
    }

    template <typename data_t>
    data_t DataHandlerRawMapCPU<data_t>::sum() const
    {
        return _map.sum();
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerRawMapCPU<data_t>::operator+=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: addition argument has wrong size");

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerCPU<data_t>*>(&v)) {
            _map += *otherHandler->_data;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerRawMapCPU*>(&v)) {
            _map += otherHandler->_map;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&v)) {
            _map += otherHandler->_map;
        } else {
            this->slowAddition(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerRawMapCPU<data_t>::operator-=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: subtraction argument has wrong size");

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerCPU<data_t>*>(&v)) {
            _map -= *otherHandler->_data;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerRawMapCPU*>(&v)) {
            _map -= otherHandler->_map;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&v)) {
            _map -= otherHandler->_map;
        } else {
            this->slowSubtraction(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerRawMapCPU<data_t>::operator*=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: multiplication argument has wrong size");

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerCPU<data_t>*>(&v)) {
            _map.array() *= otherHandler->_data->array();
        } else if (auto otherHandler = dynamic_cast<const DataHandlerRawMapCPU*>(&v)) {
            _map.array() *= otherHandler->_map.array();
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&v)) {
            _map.array() *= otherHandler->_map.array();
        } else {
            this->slowMultiplication(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerRawMapCPU<data_t>::operator/=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: division argument has wrong size");

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerCPU<data_t>*>(&v)) {
            _map.array() /= otherHandler->_data->array();
        } else if (auto otherHandler = dynamic_cast<const DataHandlerRawMapCPU*>(&v)) {
            _map.array() /= otherHandler->_map.array();
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&v)) {
            _map.array() /= otherHandler->_map.array();
        } else {
            this->slowDivision(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandlerRawMapCPU<data_t>& DataHandlerRawMapCPU<data_t>::
        operator=(const DataHandlerRawMapCPU<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: assignment argument has wrong size");

        _map = v._map;

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerRawMapCPU<data_t>::operator+=(data_t scalar)
    {
        _map.array() += scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerRawMapCPU<data_t>::operator-=(data_t scalar)
    {
        _map.array() -= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerRawMapCPU<data_t>::operator*=(data_t scalar)
    {
        _map.array() *= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerRawMapCPU<data_t>::operator/=(data_t scalar)
    {
        _map.array() /= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerRawMapCPU<data_t>::operator=(data_t scalar)
    {
        _map.setConstant(scalar);
        return *this;
    }

    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>>
        DataHandlerRawMapCPU<data_t>::getBlock(index_t startIndex, index_t numberOfElements)
    {
        if (startIndex >= getSize() || numberOfElements > getSize() - startIndex)
            throw std::invalid_argument("DataHandler: requested block out of bounds");

        return std::unique_ptr<DataHandlerRawMapCPU<data_t>>(
            new DataHandlerRawMapCPU{_map.data() + startIndex, numberOfElements});
    }

    template <typename data_t>
    std::unique_ptr<const DataHandler<data_t>>
        DataHandlerRawMapCPU<data_t>::getBlock(index_t startIndex, index_t numberOfElements) const
    {
        if (startIndex >= getSize() || numberOfElements > getSize() - startIndex)
            throw std::invalid_argument("DataHandler: requested block out of bounds");

        // using a const_cast here is fine as long as the DataHandlers never expose the internal
        // Eigen objects
        auto mutableData = const_cast<data_t*>(_map.data() + startIndex);
        return std::unique_ptr<const DataHandlerRawMapCPU<data_t>>(
            new DataHandlerRawMapCPU{mutableData, numberOfElements});
    }

    template <typename data_t>
    DataHandlerCPU<data_t>* DataHandlerRawMapCPU<data_t>::cloneImpl() const
    {
        return new DataHandlerCPU<data_t>{_map};
    }

    template <typename data_t>
    bool DataHandlerRawMapCPU<data_t>::isEqual(const DataHandler<data_t>& other) const
    {
        if (auto otherHandler = dynamic_cast<const DataHandlerRawMapCPU*>(&other)) {
            if (_map.size() != otherHandler->_map.size())
                return false;

            if (_map.data() != otherHandler->_map.data() && _map != otherHandler->_map) {
                return false;
            }
            return true;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerCPU<data_t>*>(&other)) {

            if (_map.size() != otherHandler->_data->size())
                return false;

            if (_map.data() != otherHandler->_data->data() && _map != *otherHandler->_data)
                return false;

            return true;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerRawMapCPU*>(&other)) {

            if (_map.size() != otherHandler->_map.size())
                return false;

            if (_map.data() != otherHandler->_map.data() && _map != otherHandler->_map) {
                return false;
            }
            return true;
        } else {
            return false;
        }
    }

    template <typename data_t>
    void DataHandlerRawMapCPU<data_t>::assign(const DataHandler<data_t>& other)
    {

        if (auto otherHandler = dynamic_cast<const DataHandlerRawMapCPU*>(&other)) {
            _map = otherHandler->_map;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerCPU<data_t>*>(&other)) {
            _map = *otherHandler->_data;
        } else {
            this->slowAssign(other);
        }
    }

    template <typename data_t>
    void DataHandlerRawMapCPU<data_t>::assign(DataHandler<data_t>&& other)
    {
        assign(other);
    }

    template <typename data_t>
    typename DataHandlerRawMapCPU<data_t>::DataMap_t DataHandlerRawMapCPU<data_t>::accessData()
    {
        return _map;
    }

    template <typename data_t>
    typename DataHandlerRawMapCPU<data_t>::DataMap_t
        DataHandlerRawMapCPU<data_t>::accessData() const
    {
        return _map;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DataHandlerRawMapCPU<float>;
    template class DataHandlerRawMapCPU<std::complex<float>>;
    template class DataHandlerRawMapCPU<double>;
    template class DataHandlerRawMapCPU<std::complex<double>>;
    template class DataHandlerRawMapCPU<index_t>;

} // namespace elsa
