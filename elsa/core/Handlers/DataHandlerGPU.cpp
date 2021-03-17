#include "DataHandlerGPU.h"
#include "DataHandlerMapGPU.h"
#include <cublas_v2.h>

namespace elsa
{
    template <typename data_t>
    DataHandlerGPU<data_t>::DataHandlerGPU(index_t size)
        : _data(std::make_shared<quickvec::Vector<data_t>>(size))
    {
    }

    template <typename data_t>
    DataHandlerGPU<data_t>::DataHandlerGPU(DataVector_t const& vector)
        : _data(std::make_shared<quickvec::Vector<data_t>>(vector))
    {
    }

    template <typename data_t>
    DataHandlerGPU<data_t>::DataHandlerGPU(quickvec::Vector<data_t> const& vector)
        : _data(std::make_shared<quickvec::Vector<data_t>>(vector.clone()))
    {
    }

    template <typename data_t>
    DataHandlerGPU<data_t>::DataHandlerGPU(const DataHandlerGPU<data_t>& other)
        : _data{other._data}, _associatedMaps{}
    {
    }

    template <typename data_t>
    DataHandlerGPU<data_t>::DataHandlerGPU(DataHandlerGPU<data_t>&& other) noexcept
        : _data{std::move(other._data)}, _associatedMaps{std::move(other._associatedMaps)}
    {
        for (auto& map : _associatedMaps)
            map->_dataOwner = this;
    }

    template <typename data_t>
    DataHandlerGPU<data_t>::~DataHandlerGPU()
    {
        for (auto& map : _associatedMaps)
            map->_dataOwner = nullptr;
    }

    template <typename data_t>
    index_t DataHandlerGPU<data_t>::getSize() const
    {
        return static_cast<index_t>(_data->size());
    }

    template <typename data_t>
    data_t& DataHandlerGPU<data_t>::operator[](index_t index)
    {
        detach();
        return (*_data)[static_cast<size_t>(index)];
    }

    template <typename data_t>
    const data_t& DataHandlerGPU<data_t>::operator[](index_t index) const
    {
        return (*_data)[static_cast<size_t>(index)];
    }

    template <typename data_t>
    data_t DataHandlerGPU<data_t>::dot(const DataHandler<data_t>& v) const
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandlerGPU: dot product argument has wrong size");

        // use CUDA if the other handler is GPU, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerGPU*>(&v)) {
            return _data->dot(*otherHandler->_data);
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapGPU<data_t>*>(&v)) {
            return _data->dot(otherHandler->_map);
        } else {
            return this->slowDotProduct(v);
        }
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerGPU<data_t>::squaredL2Norm() const
    {
        return _data->squaredL2Norm();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerGPU<data_t>::l2Norm() const
    {
        return _data->l2Norm();
    }

    template <typename data_t>
    index_t DataHandlerGPU<data_t>::l0PseudoNorm() const
    {
        return _data->l0PseudoNorm();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerGPU<data_t>::l1Norm() const
    {
        return _data->l1Norm();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataHandlerGPU<data_t>::lInfNorm() const
    {
        return _data->lInfNorm();
    }

    template <typename data_t>
    data_t DataHandlerGPU<data_t>::sum() const
    {
        return _data->sum();
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerGPU<data_t>::operator+=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: addition argument has wrong size");

        detach();

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerGPU*>(&v)) {
            *_data += *otherHandler->_data;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapGPU<data_t>*>(&v)) {
            *_data += otherHandler->_map;
        } else {
            this->slowAddition(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerGPU<data_t>::operator-=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: subtraction argument has wrong size");

        detach();

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerGPU*>(&v)) {
            *_data -= *otherHandler->_data;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapGPU<data_t>*>(&v)) {
            *_data -= otherHandler->_map;
        } else {
            this->slowSubtraction(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerGPU<data_t>::operator*=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: multiplication argument has wrong size");

        detach();

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerGPU*>(&v)) {
            *_data *= *otherHandler->_data;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapGPU<data_t>*>(&v)) {
            *_data *= otherHandler->_map;
        } else {
            this->slowMultiplication(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerGPU<data_t>::operator/=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: division argument has wrong size");

        detach();

        // use Eigen if the other handler is CPU or Map, otherwise use the slow fallback version
        if (auto otherHandler = dynamic_cast<const DataHandlerGPU*>(&v)) {
            *_data /= *otherHandler->_data;
        } else if (auto otherHandler = dynamic_cast<const DataHandlerMapGPU<data_t>*>(&v)) {
            *_data /= otherHandler->_map;
        } else {
            this->slowDivision(v);
        }

        return *this;
    }

    template <typename data_t>
    DataHandlerGPU<data_t>& DataHandlerGPU<data_t>::operator=(const DataHandlerGPU<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: assignment argument has wrong size");

        attach(v._data);
        return *this;
    }

    template <typename data_t>
    DataHandlerGPU<data_t>& DataHandlerGPU<data_t>::operator=(DataHandlerGPU<data_t>&& v)
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
    DataHandler<data_t>& DataHandlerGPU<data_t>::operator+=(data_t scalar)
    {
        detach();
        *_data += scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerGPU<data_t>::operator-=(data_t scalar)
    {
        detach();
        *_data -= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerGPU<data_t>::operator*=(data_t scalar)
    {
        detach();
        *_data *= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerGPU<data_t>::operator/=(data_t scalar)
    {
        detach();
        *_data /= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerGPU<data_t>::operator=(data_t scalar)
    {
        detach();
        *_data = scalar;
        return *this;
    }

    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> DataHandlerGPU<data_t>::getBlock(index_t startIndex,
                                                                          index_t numberOfElements)
    {
        if (startIndex >= getSize() || numberOfElements > getSize() - startIndex)
            throw std::invalid_argument("DataHandler: requested block out of bounds");

        return std::make_unique<DataHandlerMapGPU<data_t>>(Badge<DataHandlerGPU<data_t>>{}, this,
                                                           _data->_data.get() + startIndex,
                                                           numberOfElements);
    }

    template <typename data_t>
    std::unique_ptr<const DataHandler<data_t>>
        DataHandlerGPU<data_t>::getBlock(index_t startIndex, index_t numberOfElements) const
    {
        if (startIndex >= getSize() || numberOfElements > getSize() - startIndex)
            throw std::invalid_argument("DataHandler: requested block out of bounds");

        // using a const_cast here is fine as long as the DataHandlers never expose the internal
        // Eigen objects
        auto mutableThis = const_cast<DataHandlerGPU<data_t>*>(this);
        auto mutableData = const_cast<data_t*>(_data->_data.get() + startIndex);

        return std::make_unique<DataHandlerMapGPU<data_t>>(
            Badge<DataHandlerGPU<data_t>>{}, mutableThis, mutableData, numberOfElements);
    }

    template <typename data_t>
    DataHandlerGPU<data_t>* DataHandlerGPU<data_t>::cloneImpl() const
    {
        return new DataHandlerGPU<data_t>(*this);
    }

    template <typename data_t>
    bool DataHandlerGPU<data_t>::isEqual(const DataHandler<data_t>& other) const
    {
        if (const auto otherHandler = dynamic_cast<const DataHandlerMapGPU<data_t>*>(&other)) {

            if (_data->size() != otherHandler->_map.size())
                return false;

            if (_data->_data.get() != otherHandler->_map._data.get()
                && *_data != otherHandler->_map)
                return false;

            return true;
        } else if (const auto otherHandler = dynamic_cast<const DataHandlerGPU*>(&other)) {

            if (_data->size() != otherHandler->_data->size())
                return false;

            if (_data->_data.get() != otherHandler->_data->_data.get()
                && *_data != *otherHandler->_data)
                return false;

            return true;
        } else
            return false;
    }

    template <typename data_t>
    void DataHandlerGPU<data_t>::assign(const DataHandler<data_t>& other)
    {
        if (const auto otherHandler = dynamic_cast<const DataHandlerMapGPU<data_t>*>(&other)) {
            if (getSize() == otherHandler->_dataOwner->getSize()) {
                attach(otherHandler->_dataOwner->_data);
            } else {
                detachWithUninitializedBlock(0, getSize());
                *_data = otherHandler->_map;
            }
        } else if (const auto otherHandler = dynamic_cast<const DataHandlerGPU*>(&other)) {
            attach(otherHandler->_data);
        } else
            this->slowAssign(other);
    }

    template <typename data_t>
    void DataHandlerGPU<data_t>::assign(DataHandler<data_t>&& other)
    {
        if (const auto otherHandler = dynamic_cast<const DataHandlerMapGPU<data_t>*>(&other)) {
            if (getSize() == otherHandler->_dataOwner->getSize()) {
                attach(otherHandler->_dataOwner->_data);
            } else {
                detachWithUninitializedBlock(0, getSize());
                *_data = otherHandler->_map;
            }
        } else if (const auto otherHandler = dynamic_cast<DataHandlerGPU*>(&other)) {
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
    quickvec::Vector<data_t> DataHandlerGPU<data_t>::accessData() const
    {
        return *_data;
    }

    template <typename data_t>
    quickvec::Vector<data_t> DataHandlerGPU<data_t>::accessData()
    {
        detach();
        return *_data;
    }

    template <typename data_t>
    void DataHandlerGPU<data_t>::detach()
    {
        if (_data.use_count() != 1) {
#pragma omp barrier
#pragma omp single
            {
                data_t* oldData = _data->_data.get();

                // create deep copy of vector
                _data = std::make_shared<quickvec::Vector<data_t>>(_data->clone());

                // modify all associated maps
                for (auto map : _associatedMaps)
                    new (&map->_map) quickvec::Vector<data_t>(
                        _data->_data.get() + (map->_map._data.get() - oldData),
                        static_cast<size_t>(map->getSize()));
            }
        }
    }

    template <typename data_t>
    void DataHandlerGPU<data_t>::detachWithUninitializedBlock(index_t startIndex,
                                                              index_t numberOfElements)
    {
        if (_data.use_count() != 1) {
            // allocate new vector
            auto newData = std::make_shared<quickvec::Vector<data_t>>(getSize());

            // copy elements before start of block
            for (index_t i = 0; i < startIndex; ++i) {
                newData->operator[](static_cast<size_t>(i)) =
                    _data->operator[](static_cast<size_t>(i));
            }

            // copy elements after end of block
            for (index_t i = getSize() - startIndex - numberOfElements; i < getSize(); ++i) {
                newData->operator[](static_cast<size_t>(i)) =
                    _data->operator[](static_cast<size_t>(i));
            }

            // modify all associated maps
            for (auto map : _associatedMaps)
                new (&map->_map) quickvec::Vector<data_t>(
                    newData->_data.get() + (map->_map._data.get() - _data->_data.get()),
                    static_cast<size_t>(map->getSize()));

            _data = newData;
        }
    }

    template <typename data_t>
    void DataHandlerGPU<data_t>::attach(const std::shared_ptr<quickvec::Vector<data_t>>& data)
    {
        data_t* oldData = _data->_data.get();

        // shallow copy
        _data = data;

        // modify all associated maps
        for (auto& map : _associatedMaps)
            new (&map->_map)
                quickvec::Vector<data_t>(_data->_data.get() + (map->_map._data.get() - oldData),
                                         static_cast<size_t>(map->getSize()));
    }

    template <typename data_t>
    void DataHandlerGPU<data_t>::attach(std::shared_ptr<quickvec::Vector<data_t>>&& data)
    {
        data_t* oldData = _data->_data.get();

        // shallow copy
        _data = std::move(data);

        // modify all associated maps
        for (auto& map : _associatedMaps)
            new (&map->_map)
                quickvec::Vector<data_t>(_data->_data.get() + (map->_map._data.get() - oldData),
                                         static_cast<size_t>(map->getSize()));
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DataHandlerGPU<float>;
    template class DataHandlerGPU<std::complex<float>>;
    template class DataHandlerGPU<double>;
    template class DataHandlerGPU<std::complex<double>>;
    template class DataHandlerGPU<index_t>;

} // namespace elsa
