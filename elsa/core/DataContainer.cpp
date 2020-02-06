#include "DataContainer.h"
#include "DataHandlerCPU.h"
#include "DataHandlerMapCPU.h"
#include "BlockDescriptor.h"

#include <stdexcept>
#include <utility>

namespace elsa
{

    template <>
    DataContainer<float, 0>::DataContainer(const DataDescriptor& dataDescriptor)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{_dataDescriptor->getNumberOfCoefficients()}
    {
    }

    template <>
    DataContainer<double, 0>::DataContainer(const DataDescriptor& dataDescriptor)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{_dataDescriptor->getNumberOfCoefficients()}
    {
    }

    template <>
    DataContainer<std::complex<float>, 0>::DataContainer(const DataDescriptor& dataDescriptor)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{_dataDescriptor->getNumberOfCoefficients()}
    {
    }

    template <>
    DataContainer<std::complex<double>, 0>::DataContainer(const DataDescriptor& dataDescriptor)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{_dataDescriptor->getNumberOfCoefficients()}
    {
    }

    template <>
    DataContainer<index_t, 0>::DataContainer(const DataDescriptor& dataDescriptor)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{_dataDescriptor->getNumberOfCoefficients()}
    {
    }

    template <>
    DataContainer<float, 0>::DataContainer(const DataDescriptor& dataDescriptor,
                                           const Eigen::Matrix<float, Eigen::Dynamic, 1>& data)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{_dataDescriptor->getNumberOfCoefficients()}
    {
        if (_dataHandler.getSize() != data.size())
            throw std::invalid_argument("DataContainer: initialization vector has invalid size");

        for (index_t i = 0; i < _dataHandler.getSize(); ++i)
            (_dataHandler)[i] = data[i];
    }

    template <>
    DataContainer<double, 0>::DataContainer(const DataDescriptor& dataDescriptor,
                                            const Eigen::Matrix<double, Eigen::Dynamic, 1>& data)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{_dataDescriptor->getNumberOfCoefficients()}
    {
        if (_dataHandler.getSize() != data.size())
            throw std::invalid_argument("DataContainer: initialization vector has invalid size");

        for (index_t i = 0; i < _dataHandler.getSize(); ++i)
            (_dataHandler)[i] = data[i];
    }

    template <>
    DataContainer<std::complex<float>, 0>::DataContainer(
        const DataDescriptor& dataDescriptor,
        const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1>& data)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{_dataDescriptor->getNumberOfCoefficients()}
    {
        if (_dataHandler.getSize() != data.size())
            throw std::invalid_argument("DataContainer: initialization vector has invalid size");

        for (index_t i = 0; i < _dataHandler.getSize(); ++i)
            (_dataHandler)[i] = data[i];
    }

    template <>
    DataContainer<std::complex<double>, 0>::DataContainer(
        const DataDescriptor& dataDescriptor,
        const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>& data)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{_dataDescriptor->getNumberOfCoefficients()}
    {
        if (_dataHandler.getSize() != data.size())
            throw std::invalid_argument("DataContainer: initialization vector has invalid size");

        for (index_t i = 0; i < _dataHandler.getSize(); ++i)
            (_dataHandler)[i] = data[i];
    }

    template <>
    DataContainer<index_t, 0>::DataContainer(const DataDescriptor& dataDescriptor,
                                             const Eigen::Matrix<index_t, Eigen::Dynamic, 1>& data)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{_dataDescriptor->getNumberOfCoefficients()}
    {
        if (_dataHandler.getSize() != data.size())
            throw std::invalid_argument("DataContainer: initialization vector has invalid size");

        for (index_t i = 0; i < _dataHandler.getSize(); ++i)
            (_dataHandler)[i] = data[i];
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>::DataContainer(const DataContainer_t& other)
        : _dataDescriptor{other._dataDescriptor->clone()}, _dataHandler{other._dataHandler}
    {
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>& DataContainer<data_t, handler_t>::
        operator=(const DataContainer<data_t, handler_t>& other)
    {
        if (this != &other) {
            _dataDescriptor = other._dataDescriptor->clone();
            _dataHandler = other._dataHandler;
        }

        return *this;
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>::DataContainer(DataContainer_t&& other) noexcept
        : _dataDescriptor{std::move(other._dataDescriptor)},
          _dataHandler{std::move(other._dataHandler)}
    {
        // leave other in a valid state
        other._dataDescriptor = nullptr;
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>& DataContainer<data_t, handler_t>::
        operator=(DataContainer_t&& other)
    {
        _dataDescriptor = std::move(other._dataDescriptor);
        _dataHandler = std::move(other._dataHandler);

        // leave other in a valid state
        other._dataDescriptor = nullptr;

        return *this;
    }

    template <typename data_t, int handler_t>
    const DataDescriptor& DataContainer<data_t, handler_t>::getDataDescriptor() const
    {
        return *_dataDescriptor;
    }

    template <typename data_t, int handler_t>
    index_t DataContainer<data_t, handler_t>::getSize() const
    {
        return _dataHandler.getSize();
    }

    template <typename data_t, int handler_t>
    data_t& DataContainer<data_t, handler_t>::operator[](index_t index)
    {
        return (_dataHandler)[index];
    }

    template <typename data_t, int handler_t>
    const data_t& DataContainer<data_t, handler_t>::operator[](index_t index) const
    {
        return static_cast<const DataHandler<data_t>&>(_dataHandler)[index];
    }

    template <typename data_t, int handler_t>
    data_t& DataContainer<data_t, handler_t>::operator()(IndexVector_t coordinate)
    {
        return (_dataHandler)[_dataDescriptor->getIndexFromCoordinate(std::move(coordinate))];
    }

    template <typename data_t, int handler_t>
    const data_t& DataContainer<data_t, handler_t>::operator()(IndexVector_t coordinate) const
    {
        return static_cast<const DataHandler<data_t>&>(
            _dataHandler)[_dataDescriptor->getIndexFromCoordinate(std::move(coordinate))];
    }

    template <typename data_t, int handler_t>
    data_t DataContainer<data_t, handler_t>::dot(const DataContainer_t& other) const
    {
        return _dataHandler.dot(other._dataHandler);
    }

    template <typename data_t, int handler_t>
    GetFloatingPointType_t<data_t> DataContainer<data_t, handler_t>::squaredL2Norm() const
    {
        return _dataHandler.squaredL2Norm();
    }

    template <typename data_t, int handler_t>
    GetFloatingPointType_t<data_t> DataContainer<data_t, handler_t>::l1Norm() const
    {
        return _dataHandler.l1Norm();
    }

    template <typename data_t, int handler_t>
    GetFloatingPointType_t<data_t> DataContainer<data_t, handler_t>::lInfNorm() const
    {
        return _dataHandler.lInfNorm();
    }

    template <typename data_t, int handler_t>
    data_t DataContainer<data_t, handler_t>::sum() const
    {
        return _dataHandler.sum();
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>& DataContainer<data_t, handler_t>::
        operator+=(const DataContainer_t& dc)
    {
        _dataHandler += dc._dataHandler;
        return *this;
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>& DataContainer<data_t, handler_t>::
        operator-=(const DataContainer_t& dc)
    {
        _dataHandler -= dc._dataHandler;
        return *this;
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>& DataContainer<data_t, handler_t>::
        operator*=(const DataContainer_t& dc)
    {
        _dataHandler *= dc._dataHandler;
        return *this;
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>& DataContainer<data_t, handler_t>::
        operator/=(const DataContainer_t& dc)
    {
        _dataHandler /= dc._dataHandler;
        return *this;
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>& DataContainer<data_t, handler_t>::operator+=(data_t scalar)
    {
        _dataHandler += scalar;
        return *this;
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>& DataContainer<data_t, handler_t>::operator-=(data_t scalar)
    {
        _dataHandler -= scalar;
        return *this;
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>& DataContainer<data_t, handler_t>::operator*=(data_t scalar)
    {
        _dataHandler *= scalar;
        return *this;
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>& DataContainer<data_t, handler_t>::operator/=(data_t scalar)
    {
        _dataHandler /= scalar;
        return *this;
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, handler_t>& DataContainer<data_t, handler_t>::operator=(data_t scalar)
    {
        _dataHandler = scalar;
        return *this;
    }

    template <>
    DataContainer<float, DataHandlerType::MAP_CPU>::DataContainer(
        const DataDescriptor& dataDescriptor, DataHandlerMapCPU<float> dataHandler)
        : _dataDescriptor{dataDescriptor.clone()}, _dataHandler{std::move(dataHandler)}
    {
    }

    template <>
    DataContainer<std::complex<double>, DataHandlerType::MAP_CPU>::DataContainer(
        const DataDescriptor& dataDescriptor, DataHandlerMapCPU<std::complex<double>> dataHandler)
        : _dataDescriptor{dataDescriptor.clone()}, _dataHandler{std::move(dataHandler)}
    {
    }

    template <>
    DataContainer<std::complex<float>, DataHandlerType::MAP_CPU>::DataContainer(
        const DataDescriptor& dataDescriptor, DataHandlerMapCPU<std::complex<float>> dataHandler)
        : _dataDescriptor{dataDescriptor.clone()}, _dataHandler{std::move(dataHandler)}
    {
    }

    template <>
    DataContainer<index_t, DataHandlerType::MAP_CPU>::DataContainer(
        const DataDescriptor& dataDescriptor, DataHandlerMapCPU<index_t> dataHandler)
        : _dataDescriptor{dataDescriptor.clone()}, _dataHandler{std::move(dataHandler)}
    {
    }

    template <>
    DataContainer<double, DataHandlerType::MAP_CPU>::DataContainer(
        const DataDescriptor& dataDescriptor, DataHandlerMapCPU<double> dataHandler)
        : _dataDescriptor{dataDescriptor.clone()}, _dataHandler{std::move(dataHandler)}
    {
    }

    template <typename data_t, int handler_t>
    bool DataContainer<data_t, handler_t>::operator==(const DataContainer_t& other) const
    {
        if (*_dataDescriptor != *other._dataDescriptor)
            return false;

        if (_dataHandler != other._dataHandler)
            return false;

        return true;
    }

    template <typename data_t, int handler_t>
    bool DataContainer<data_t, handler_t>::operator!=(const DataContainer_t& other) const
    {
        return !(*this == other);
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, DataHandlerType::MAP_CPU>
        DataContainer<data_t, handler_t>::getBlock(index_t i)
    {
        const auto blockDesc = dynamic_cast<const BlockDescriptor*>(_dataDescriptor.get());
        if (!blockDesc)
            throw std::logic_error("DataContainer: cannot get block from not-blocked container");

        if (i >= blockDesc->getNumberOfBlocks() || i < 0)
            throw std::invalid_argument("DataContainer: block index out of bounds");

        index_t startIndex = blockDesc->getOffsetOfBlock(i);
        const auto& ithDesc = blockDesc->getDescriptorOfBlock(i);
        index_t blockSize = ithDesc.getNumberOfCoefficients();

        return DataContainer<data_t, DataHandlerType::MAP_CPU>{
            ithDesc, _dataHandler.getBlock(startIndex, blockSize)};
    }

    template <typename data_t, int handler_t>
    const DataContainer<data_t, DataHandlerType::MAP_CPU>
        DataContainer<data_t, handler_t>::getBlock(index_t i) const
    {
        const auto blockDesc = dynamic_cast<const BlockDescriptor*>(_dataDescriptor.get());
        if (!blockDesc)
            throw std::logic_error("DataContainer: cannot get block from not-blocked container");

        if (i >= blockDesc->getNumberOfBlocks() || i < 0)
            throw std::invalid_argument("DataContainer: block index out of bounds");

        index_t startIndex = blockDesc->getOffsetOfBlock(i);
        const auto& ithDesc = blockDesc->getDescriptorOfBlock(i);
        index_t blockSize = ithDesc.getNumberOfCoefficients();

        // getBlock() returns a pointer to non-const DH, but that's fine as it gets wrapped in a
        // constant container
        return DataContainer<data_t, DataHandlerType::MAP_CPU>{
            ithDesc, _dataHandler.getBlock(startIndex, blockSize)};
    }

    template <typename data_t, int handler_t>
    DataContainer<data_t, DataHandlerType::MAP_CPU>
        DataContainer<data_t, handler_t>::viewAs(const DataDescriptor& dataDescriptor)
    {
        if (dataDescriptor.getNumberOfCoefficients() != getSize())
            throw std::invalid_argument("DataContainer: view must have same size as container");

        return DataContainer<data_t, DataHandlerType::MAP_CPU>{dataDescriptor,
                                                               _dataHandler.getBlock(0, getSize())};
    }

    template <typename data_t, int handler_t>
    const DataContainer<data_t, DataHandlerType::MAP_CPU>
        DataContainer<data_t, handler_t>::viewAs(const DataDescriptor& dataDescriptor) const
    {
        if (dataDescriptor.getNumberOfCoefficients() != getSize())
            throw std::invalid_argument("DataContainer: view must have same size as container");

        // getBlock() returns a pointer to non-const DH, but that's fine as it gets wrapped in a
        // constant container
        return DataContainer<data_t, DataHandlerType::MAP_CPU>{dataDescriptor,
                                                               _dataHandler.getBlock(0, getSize())};
    }

    template <typename data_t, int handler_t>
    typename DataContainer<data_t, handler_t>::iterator DataContainer<data_t, handler_t>::begin()
    {
        return iterator(&(*this)[0]);
    }

    template <typename data_t, int handler_t>
    typename DataContainer<data_t, handler_t>::const_iterator
        DataContainer<data_t, handler_t>::begin() const
    {
        return cbegin();
    }

    template <typename data_t, int handler_t>
    typename DataContainer<data_t, handler_t>::const_iterator
        DataContainer<data_t, handler_t>::cbegin() const
    {
        return const_iterator(&(*this)[0]);
    }

    template <typename data_t, int handler_t>
    typename DataContainer<data_t, handler_t>::iterator DataContainer<data_t, handler_t>::end()
    {
        return iterator(&(*this)[0] + getSize());
    }

    template <typename data_t, int handler_t>
    typename DataContainer<data_t, handler_t>::const_iterator
        DataContainer<data_t, handler_t>::end() const
    {
        return cend();
    }

    template <typename data_t, int handler_t>
    typename DataContainer<data_t, handler_t>::const_iterator
        DataContainer<data_t, handler_t>::cend() const
    {
        return const_iterator(&(*this)[0] + getSize());
    }

    template <typename data_t, int handler_t>
    typename DataContainer<data_t, handler_t>::reverse_iterator
        DataContainer<data_t, handler_t>::rbegin()
    {
        return reverse_iterator(end());
    }

    template <typename data_t, int handler_t>
    typename DataContainer<data_t, handler_t>::const_reverse_iterator
        DataContainer<data_t, handler_t>::rbegin() const
    {
        return crbegin();
    }

    template <typename data_t, int handler_t>
    typename DataContainer<data_t, handler_t>::const_reverse_iterator
        DataContainer<data_t, handler_t>::crbegin() const
    {
        return const_reverse_iterator(cend());
    }

    template <typename data_t, int handler_t>
    typename DataContainer<data_t, handler_t>::reverse_iterator
        DataContainer<data_t, handler_t>::rend()
    {
        return reverse_iterator(begin());
    }

    template <typename data_t, int handler_t>
    typename DataContainer<data_t, handler_t>::const_reverse_iterator
        DataContainer<data_t, handler_t>::rend() const
    {
        return crend();
    }

    template <typename data_t, int handler_t>
    typename DataContainer<data_t, handler_t>::const_reverse_iterator
        DataContainer<data_t, handler_t>::crend() const
    {
        return const_reverse_iterator(cbegin());
    }

    template <typename data_t, int handler_t>
    DataHandlerType DataContainer<data_t, handler_t>::getDataHandlerType() const
    {
        return DataHandlerType(handler_t);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DataContainer<float>;
    template class DataContainer<std::complex<float>>;
    template class DataContainer<double>;
    template class DataContainer<std::complex<double>>;
    template class DataContainer<index_t>;

    template class DataContainer<float, DataHandlerType::MAP_CPU>;
    template class DataContainer<std::complex<float>, DataHandlerType::MAP_CPU>;
    template class DataContainer<double, DataHandlerType::MAP_CPU>;
    template class DataContainer<std::complex<double>, DataHandlerType::MAP_CPU>;
    template class DataContainer<index_t, DataHandlerType::MAP_CPU>;

} // namespace elsa
