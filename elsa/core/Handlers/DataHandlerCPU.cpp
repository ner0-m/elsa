#include "DataHandlerCPU.h"
#include "DataHandlerMapCPU.h"

namespace elsa
{

    template <typename data_t>
    DataHandlerCPU<data_t>::DataHandlerCPU(index_t size)
        : _data(std::make_shared<DataVector_t>(size))
    {
    }

    template <typename data_t>
    DataHandlerCPU<data_t>::DataHandlerCPU(DataVector_t const& vector)
        : _data{std::make_shared<DataVector_t>(vector)}
    {
    }

    template <typename data_t>
    DataHandlerCPU<data_t>::DataHandlerCPU(const DataHandlerCPU<data_t>& other)
        : _data{other._data}, _associatedMaps{}
    {
    }

    template <typename data_t>
    DataHandlerCPU<data_t>::DataHandlerCPU(DataHandlerCPU<data_t>&& other)
        : _data{std::move(other._data)}, _associatedMaps{std::move(other._associatedMaps)}
    {
        for (auto& map : _associatedMaps)
            map->_dataOwner = this;
    }

    template <typename data_t>
    DataHandlerCPU<data_t>::~DataHandlerCPU()
    {
        for (auto& map : _associatedMaps)
            map->_dataOwner = nullptr;
    }

    template <typename data_t>
    index_t DataHandlerCPU<data_t>::getSize() const
    {
        return static_cast<index_t>(_data->size());
    }

    template <typename data_t>
    data_t& DataHandlerCPU<data_t>::operator[](index_t index)
    {
        detach();
        return (*_data)[index];
    }

    template <typename data_t>
    const data_t& DataHandlerCPU<data_t>::operator[](index_t index) const
    {
        return (*_data)[index];
    }

    template <typename data_t>
    data_t DataHandlerCPU<data_t>::dot(const DataHandler<data_t>& v) const
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandlerCPU: dot product argument has wrong size");

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&v)) {
            return _data->dot(*otherHandler->_data);
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&v)) {
            return _data->dot(otherHandler->_map);
        } else {
            return this->slowDotProduct(v);
        }
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerCPU<data_t>::squaredL2Norm() const
    {
        return _data->squaredNorm();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerCPU<data_t>::l1Norm() const
    {
        return _data->array().abs().sum();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerCPU<data_t>::lInfNorm() const
    {
        return _data->array().abs().maxCoeff();
    }

    template <typename data_t>
    data_t DataHandlerCPU<data_t>::sum() const
    {
        return _data->sum();
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator+=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: addition argument has wrong size");

        detach();

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&v)) {
            *_data += *otherHandler->_data;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&v)) {
            *_data += otherHandler->_map;
        } else {
            this->slowAddition(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator-=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: subtraction argument has wrong size");

        detach();

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&v)) {
            *_data -= *otherHandler->_data;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&v)) {
            *_data -= otherHandler->_map;
        } else {
            this->slowSubtraction(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator*=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: multiplication argument has wrong size");

        detach();

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&v)) {
            _data->array() *= otherHandler->_data->array();
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&v)) {
            _data->array() *= otherHandler->_map.array();
        } else {
            this->slowMultiplication(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator/=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: division argument has wrong size");

        detach();

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&v)) {
            _data->array() /= otherHandler->_data->array();
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&v)) {
            _data->array() /= otherHandler->_map.array();
        } else {
            this->slowDivision(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandlerCPU<data_t>& DataHandlerCPU<data_t>::operator=(const DataHandlerCPU<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: assignment argument has wrong size");

        attach(v._data);
        return *this;
    }

    template <typename data_t>
    DataHandlerCPU<data_t>& DataHandlerCPU<data_t>::operator=(DataHandlerCPU<data_t>&& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: assignment argument has wrong size");

        attach(std::move(v._data));

        for (auto& map : v._associatedMaps)
            map->_dataOwner = this;

        _associatedMaps.splice(_associatedMaps.end(), std::move(v._associatedMaps));

        // make sure v no longer owns the object
        v._data.reset();
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator+=(data_t scalar)
    {
        detach();
        _data->array() += scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator-=(data_t scalar)
    {
        detach();
        _data->array() -= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator*=(data_t scalar)
    {
        detach();
        _data->array() *= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator/=(data_t scalar)
    {
        detach();
        _data->array() /= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator=(data_t scalar)
    {
        detach();
        _data->setConstant(scalar);
        return *this;
    }

    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> DataHandlerCPU<data_t>::getBlock(index_t startIndex,
                                                                          index_t numberOfElements)
    {
        if (startIndex >= getSize() || numberOfElements > getSize() - startIndex)
            throw std::invalid_argument("DataHandler: requested block out of bounds");

        return std::unique_ptr<DataHandlerMapCPU<data_t>>{
            new DataHandlerMapCPU{this, _data->data() + startIndex, numberOfElements}};
    }

    template <typename data_t>
    std::unique_ptr<const DataHandler<data_t>>
        DataHandlerCPU<data_t>::getBlock(index_t startIndex, index_t numberOfElements) const
    {
        if (startIndex >= getSize() || numberOfElements > getSize() - startIndex)
            throw std::invalid_argument("DataHandler: requested block out of bounds");

        // using a const_cast here is fine as long as the DataHandlers never expose the internal
        // Eigen objects
        auto mutableThis = const_cast<DataHandlerCPU<data_t>*>(this);
        auto mutableData = const_cast<data_t*>(_data->data() + startIndex);
        return std::unique_ptr<const DataHandlerMapCPU<data_t>>{
            new DataHandlerMapCPU{mutableThis, mutableData, numberOfElements}};
    }

    template <typename data_t>
    DataHandlerCPU<data_t>* DataHandlerCPU<data_t>::cloneImpl() const
    {
        return new DataHandlerCPU<data_t>(*this);
    }

    template <typename data_t>
    bool DataHandlerCPU<data_t>::isEqual(const DataHandler<data_t>& other) const
    {
        if (const auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&other)) {

            if (_data->size() != otherHandler->_map.size())
                return false;

            if (_data->data() != otherHandler->_map.data() && *_data != otherHandler->_map)
                return false;

            return true;
        } else if (const auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&other)) {

            if (_data->size() != otherHandler->_data->size())
                return false;

            if (_data->data() != otherHandler->_data->data() && *_data != *otherHandler->_data)
                return false;

            return true;
        } else
            return false;
    }

    template <typename data_t>
    void DataHandlerCPU<data_t>::assign(const DataHandler<data_t>& other)
    {

        if (const auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&other)) {
            if (getSize() == otherHandler->_dataOwner->getSize()) {
                attach(otherHandler->_dataOwner->_data);
            } else {
                detachWithUninitializedBlock(0, getSize());
                *_data = otherHandler->_map;
            }
        } else if (const auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&other)) {
            attach(otherHandler->_data);
        } else
            this->slowAssign(other);
    }

    template <typename data_t>
    void DataHandlerCPU<data_t>::assign(DataHandler<data_t>&& other)
    {
        if (const auto otherHandler = dynamic_cast<const DataHandlerMapCPU<data_t>*>(&other)) {
            if (getSize() == otherHandler->_dataOwner->getSize()) {
                attach(otherHandler->_dataOwner->_data);
            } else {
                detachWithUninitializedBlock(0, getSize());
                *_data = otherHandler->_map;
            }
        } else if (const auto otherHandler = dynamic_cast<DataHandlerCPU*>(&other)) {
            attach(std::move(otherHandler->_data));

            for (auto& map : otherHandler->_associatedMaps)
                map->_dataOwner = this;

            _associatedMaps.splice(_associatedMaps.end(), std::move(otherHandler->_associatedMaps));

            // make sure v no longer owns the object
            otherHandler->_data.reset();
        } else
            this->slowAssign(other);
    }

    template <typename data_t>
    void DataHandlerCPU<data_t>::detach()
    {
        if (_data.use_count() != 1) {
#pragma omp barrier
#pragma omp single
            {
                data_t* oldData = _data->data();

                // create deep copy of vector
                _data = std::make_shared<DataVector_t>(*_data);

                // modify all associated maps
                for (auto map : _associatedMaps)
                    new (&map->_map) Eigen::Map<DataVector_t>(
                        _data->data() + (map->_map.data() - oldData), map->getSize());
            }
        }
    }

    template <typename data_t>
    void DataHandlerCPU<data_t>::detachWithUninitializedBlock(index_t startIndex,
                                                              index_t numberOfElements)
    {
        if (_data.use_count() != 1) {
            // allocate new vector
            auto newData = std::make_shared<DataVector_t>(getSize());

            // copy elements before start of block
            newData->head(startIndex) = _data->head(startIndex);
            newData->tail(getSize() - startIndex - numberOfElements) =
                _data->tail(getSize() - startIndex - numberOfElements);

            // modify all associated maps
            for (auto map : _associatedMaps)
                new (&map->_map) Eigen::Map<DataVector_t>(
                    newData->data() + (map->_map.data() - _data->data()), map->getSize());

            _data = newData;
        }
    }

    template <typename data_t>
    void DataHandlerCPU<data_t>::attach(const std::shared_ptr<DataVector_t>& data)
    {
        data_t* oldData = _data->data();

        // shallow copy
        _data = data;

        // modify all associated maps
        for (auto& map : _associatedMaps)
            new (&map->_map) Eigen::Map<DataVector_t>(_data->data() + (map->_map.data() - oldData),
                                                      map->getSize());
    }

    template <typename data_t>
    void DataHandlerCPU<data_t>::attach(std::shared_ptr<DataVector_t>&& data)
    {
        data_t* oldData = _data->data();

        // shallow copy
        _data = std::move(data);

        // modify all associated maps
        for (auto& map : _associatedMaps)
            new (&map->_map) Eigen::Map<DataVector_t>(_data->data() + (map->_map.data() - oldData),
                                                      map->getSize());
    }

    template <typename data_t>
    typename DataHandlerCPU<data_t>::DataMap_t DataHandlerCPU<data_t>::accessData() const
    {
        return DataMap_t(&(_data->operator[](0)), getSize());
    }

    template <typename data_t>
    typename DataHandlerCPU<data_t>::DataMap_t DataHandlerCPU<data_t>::accessData()
    {
        detach();
        return DataMap_t(&(_data->operator[](0)), getSize());
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DataHandlerCPU<float>;
    template class DataHandlerCPU<std::complex<float>>;
    template class DataHandlerCPU<double>;
    template class DataHandlerCPU<std::complex<double>>;
    template class DataHandlerCPU<index_t>;

} // namespace elsa
