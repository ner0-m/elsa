#include "DataContainer.h"
#include "DataContainerFormatter.hpp"
#include "DataHandlerCPU.h"
#include "DataHandlerMapCPU.h"
#include "BlockDescriptor.h"
#include "RandomBlocksDescriptor.h"
#include "PartitionDescriptor.h"
#include "Error.h"
#include "TypeCasts.hpp"

#include <utility>

namespace elsa
{

    template <typename data_t>
    DataContainer<data_t>::DataContainer(const DataDescriptor& dataDescriptor,
                                         DataHandlerType handlerType)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{createDataHandler(handlerType, _dataDescriptor->getNumberOfCoefficients())},
          _dataHandlerType{handlerType}
    {
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(const DataDescriptor& dataDescriptor,
                                         const Eigen::Matrix<data_t, Eigen::Dynamic, 1>& data,
                                         DataHandlerType handlerType)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{createDataHandler(handlerType, _dataDescriptor->getNumberOfCoefficients())},
          _dataHandlerType{handlerType}
    {
        if (_dataHandler->getSize() != data.size())
            throw InvalidArgumentError("DataContainer: initialization vector has invalid size");

        for (index_t i = 0; i < _dataHandler->getSize(); ++i)
            (*_dataHandler)[i] = data[i];
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(const DataContainer<data_t>& other)
        : _dataDescriptor{other._dataDescriptor->clone()},
          _dataHandler{other._dataHandler->clone()},
          _dataHandlerType{other._dataHandlerType}
    {
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator=(const DataContainer<data_t>& other)
    {
        if (this != &other) {
            _dataDescriptor = other._dataDescriptor->clone();

            if (_dataHandler && canAssign(other._dataHandlerType)) {
                *_dataHandler = *other._dataHandler;
            } else {
                _dataHandler = other._dataHandler->clone();
                _dataHandlerType = other._dataHandlerType;
            }
        }

        return *this;
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(DataContainer<data_t>&& other) noexcept
        : _dataDescriptor{std::move(other._dataDescriptor)},
          _dataHandler{std::move(other._dataHandler)},
          _dataHandlerType{std::move(other._dataHandlerType)}
    {
        // leave other in a valid state
        other._dataDescriptor = nullptr;
        other._dataHandler = nullptr;
    }

    template <typename data_t>
    DataContainer<data_t>& DataContainer<data_t>::operator=(DataContainer<data_t>&& other)
    {
        _dataDescriptor = std::move(other._dataDescriptor);

        if (_dataHandler && canAssign(other._dataHandlerType)) {
            *_dataHandler = std::move(*other._dataHandler);
        } else {
            _dataHandler = std::move(other._dataHandler);
            _dataHandlerType = std::move(other._dataHandlerType);
        }

        // leave other in a valid state
        other._dataDescriptor = nullptr;
        other._dataHandler = nullptr;

        return *this;
    }

    template <typename data_t>
    const DataDescriptor& DataContainer<data_t>::getDataDescriptor() const
    {
        return *_dataDescriptor;
    }

    template <typename data_t>
    index_t DataContainer<data_t>::getSize() const
    {
        return _dataHandler->getSize();
    }

    template <typename data_t>
    data_t& DataContainer<data_t>::operator[](index_t index)
    {
        return (*_dataHandler)[index];
    }

    template <typename data_t>
    const data_t& DataContainer<data_t>::operator[](index_t index) const
    {
        return static_cast<const DataHandler<data_t>&>(*_dataHandler)[index];
    }

    template <typename data_t>
    data_t& DataContainer<data_t>::operator()(IndexVector_t coordinate)
    {
        return (*_dataHandler)[_dataDescriptor->getIndexFromCoordinate(std::move(coordinate))];
    }

    template <typename data_t>
    const data_t& DataContainer<data_t>::operator()(IndexVector_t coordinate) const
    {
        return static_cast<const DataHandler<data_t>&>(
            *_dataHandler)[_dataDescriptor->getIndexFromCoordinate(std::move(coordinate))];
    }

    template <typename data_t>
    data_t DataContainer<data_t>::dot(const DataContainer<data_t>& other) const
    {
        return _dataHandler->dot(*other._dataHandler);
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataContainer<data_t>::squaredL2Norm() const
    {
        return _dataHandler->squaredL2Norm();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataContainer<data_t>::l2Norm() const
    {
        return _dataHandler->l2Norm();
    }

    template <typename data_t>
    index_t DataContainer<data_t>::l0PseudoNorm() const
    {
        return _dataHandler->l0PseudoNorm();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataContainer<data_t>::l1Norm() const
    {
        return _dataHandler->l1Norm();
    }

    template <typename data_t>
    GetFloatingPointType_t<data_t> DataContainer<data_t>::lInfNorm() const
    {
        return _dataHandler->lInfNorm();
    }

    template <typename data_t>
    data_t DataContainer<data_t>::sum() const
    {
        return _dataHandler->sum();
    }

    template <typename data_t>
    void DataContainer<data_t>::fft(FFTNorm norm) const
    {
        this->_dataHandler->fft(*this->_dataDescriptor, norm);
    }

    template <typename data_t>
    void DataContainer<data_t>::ifft(FFTNorm norm) const
    {
        this->_dataHandler->ifft(*this->_dataDescriptor, norm);
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

    template <typename data_t>
    template <typename... Args>
    std::unique_ptr<DataHandler<data_t>>
        DataContainer<data_t>::createDataHandler(DataHandlerType handlerType, Args&&... args)
    {
        switch (handlerType) {
            case DataHandlerType::CPU:
                return std::make_unique<DataHandlerCPU<data_t>>(std::forward<Args>(args)...);
            case DataHandlerType::MAP_CPU:
                return std::make_unique<DataHandlerCPU<data_t>>(std::forward<Args>(args)...);
#ifdef ELSA_CUDA_VECTOR
            case DataHandlerType::GPU:
                return std::make_unique<DataHandlerGPU<data_t>>(std::forward<Args>(args)...);
            case DataHandlerType::MAP_GPU:
                return std::make_unique<DataHandlerGPU<data_t>>(std::forward<Args>(args)...);
#endif
            default:
                throw InvalidArgumentError("DataContainer: unknown handler type");
        }
    }

    template <typename data_t>
    DataContainer<data_t>::DataContainer(const DataDescriptor& dataDescriptor,
                                         std::unique_ptr<DataHandler<data_t>> dataHandler,
                                         DataHandlerType dataType)
        : _dataDescriptor{dataDescriptor.clone()},
          _dataHandler{std::move(dataHandler)},
          _dataHandlerType{dataType}
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
    DataContainer<data_t> DataContainer<data_t>::getBlock(index_t i)
    {
        const auto blockDesc = downcast_safe<BlockDescriptor>(_dataDescriptor.get());
        if (!blockDesc)
            throw LogicError("DataContainer: cannot get block from not-blocked container");

        if (i >= blockDesc->getNumberOfBlocks() || i < 0)
            throw InvalidArgumentError("DataContainer: block index out of bounds");

        index_t startIndex = blockDesc->getOffsetOfBlock(i);
        const auto& ithDesc = blockDesc->getDescriptorOfBlock(i);
        index_t blockSize = ithDesc.getNumberOfCoefficients();

        DataHandlerType newHandlerType = (_dataHandlerType == DataHandlerType::CPU
                                          || _dataHandlerType == DataHandlerType::MAP_CPU)
                                             ? DataHandlerType::MAP_CPU
                                             : DataHandlerType::MAP_GPU;

        return DataContainer<data_t>{ithDesc, _dataHandler->getBlock(startIndex, blockSize),
                                     newHandlerType};
    }

    template <typename data_t>
    const DataContainer<data_t> DataContainer<data_t>::getBlock(index_t i) const
    {
        const auto blockDesc = downcast_safe<BlockDescriptor>(_dataDescriptor.get());
        if (!blockDesc)
            throw LogicError("DataContainer: cannot get block from not-blocked container");

        if (i >= blockDesc->getNumberOfBlocks() || i < 0)
            throw InvalidArgumentError("DataContainer: block index out of bounds");

        index_t startIndex = blockDesc->getOffsetOfBlock(i);
        const auto& ithDesc = blockDesc->getDescriptorOfBlock(i);
        index_t blockSize = ithDesc.getNumberOfCoefficients();

        DataHandlerType newHandlerType = (_dataHandlerType == DataHandlerType::CPU
                                          || _dataHandlerType == DataHandlerType::MAP_CPU)
                                             ? DataHandlerType::MAP_CPU
                                             : DataHandlerType::MAP_GPU;

        // getBlock() returns a pointer to non-const DH, but that's fine as it gets wrapped in a
        // constant container
        return DataContainer<data_t>{ithDesc, _dataHandler->getBlock(startIndex, blockSize),
                                     newHandlerType};
    }

    template <typename data_t>
    DataContainer<data_t> DataContainer<data_t>::viewAs(const DataDescriptor& dataDescriptor)
    {
        if (dataDescriptor.getNumberOfCoefficients() != getSize())
            throw InvalidArgumentError("DataContainer: view must have same size as container");

        DataHandlerType newHandlerType = (_dataHandlerType == DataHandlerType::CPU
                                          || _dataHandlerType == DataHandlerType::MAP_CPU)
                                             ? DataHandlerType::MAP_CPU
                                             : DataHandlerType::MAP_GPU;

        return DataContainer<data_t>{dataDescriptor, _dataHandler->getBlock(0, getSize()),
                                     newHandlerType};
    }

    template <typename data_t>
    const DataContainer<data_t>
        DataContainer<data_t>::viewAs(const DataDescriptor& dataDescriptor) const
    {
        if (dataDescriptor.getNumberOfCoefficients() != getSize())
            throw InvalidArgumentError("DataContainer: view must have same size as container");

        DataHandlerType newHandlerType = (_dataHandlerType == DataHandlerType::CPU
                                          || _dataHandlerType == DataHandlerType::MAP_CPU)
                                             ? DataHandlerType::MAP_CPU
                                             : DataHandlerType::MAP_GPU;

        // getBlock() returns a pointer to non-const DH, but that's fine as it gets wrapped in a
        // constant container
        return DataContainer<data_t>{dataDescriptor, _dataHandler->getBlock(0, getSize()),
                                     newHandlerType};
    }

    template <typename data_t>
    const DataContainer<data_t> DataContainer<data_t>::slice(index_t i) const
    {
        auto& desc = getDataDescriptor();
        auto dim = desc.getNumberOfDimensions();
        auto sizeOfLastDim = desc.getNumberOfCoefficientsPerDimension()[dim - 1];

        if (i >= sizeOfLastDim) {
            throw LogicError("Trying to access out of bound slice");
        }

        auto sliceDesc = PartitionDescriptor(desc, sizeOfLastDim);

        // Now set the slice
        return viewAs(sliceDesc).getBlock(i);
    }

    template <typename data_t>
    DataContainer<data_t> DataContainer<data_t>::slice(index_t i)
    {
        auto& desc = getDataDescriptor();
        auto dim = desc.getNumberOfDimensions();
        auto sizeOfLastDim = desc.getNumberOfCoefficientsPerDimension()[dim - 1];

        if (i >= sizeOfLastDim) {
            throw LogicError("Trying to access out of bound slice");
        }

        auto sliceDesc = PartitionDescriptor(desc, sizeOfLastDim);

        // Now set the slice
        return viewAs(sliceDesc).getBlock(i);
    }

    template <typename data_t>
    typename DataContainer<data_t>::iterator DataContainer<data_t>::begin()
    {
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

    template <typename data_t>
    DataHandlerType DataContainer<data_t>::getDataHandlerType() const
    {
        return _dataHandlerType;
    }

    template <typename data_t>
    void DataContainer<data_t>::format(std::ostream& os) const
    {
        DataContainerFormatter<data_t> fmt;
        fmt.format(os, *this);
    }

    template <typename data_t>
    DataContainer<data_t> DataContainer<data_t>::loadToCPU()
    {
        if (_dataHandlerType == DataHandlerType::CPU
            || _dataHandlerType == DataHandlerType::MAP_CPU) {
            throw LogicError(
                "DataContainer: cannot load data to CPU with already CPU based container");
        }

        DataContainer<data_t> dcCPU(*_dataDescriptor, DataHandlerType::CPU);

        for (index_t i = 0; i < getSize(); i++) {
            dcCPU[i] = this->operator[](i);
        }

        return dcCPU;
    }

    template <typename data_t>
    DataContainer<data_t> DataContainer<data_t>::loadToGPU()
    {
        if (_dataHandlerType == DataHandlerType::GPU
            || _dataHandlerType == DataHandlerType::MAP_GPU) {
            throw LogicError(
                "DataContainer: cannot load data to GPU with already GPU based container");
        }

        DataContainer<data_t> dcGPU(*_dataDescriptor, DataHandlerType::GPU);

        for (index_t i = 0; i < getSize(); i++) {
            dcGPU[i] = this->operator[](i);
        }

        return dcGPU;
    }

    template <typename data_t>
    bool DataContainer<data_t>::canAssign(DataHandlerType handlerType)
    {
        if (_dataHandlerType == DataHandlerType::CPU
            || _dataHandlerType == DataHandlerType::MAP_CPU) {
            switch (handlerType) {
                case DataHandlerType::CPU:
                    return true;
                    break;
                case DataHandlerType::MAP_CPU:
                    return true;
                    break;
                default:
                    return false;
            }
        } else {
            switch (handlerType) {
                case DataHandlerType::GPU:
                    return true;
                    break;
                case DataHandlerType::MAP_GPU:
                    return true;
                    break;
                default:
                    return false;
            }
        }
    }

    template <typename data_t>
    DataContainer<data_t> concatenate(const DataContainer<data_t>& dc1,
                                      const DataContainer<data_t>& dc2)
    {
        auto desc1 = dc1.getDataDescriptor().clone();
        auto desc2 = dc2.getDataDescriptor().clone();

        if (desc1->getNumberOfDimensions() != desc2->getNumberOfDimensions()) {
            throw LogicError("Can't concatenate two DataContainers with different dimension");
        }

        std::vector<std::unique_ptr<DataDescriptor>> descriptors;
        descriptors.reserve(2);
        descriptors.push_back(std::move(desc1));
        descriptors.push_back(std::move(desc2));

        auto blockDesc = RandomBlocksDescriptor(descriptors);
        auto concatenated = DataContainer<data_t>(blockDesc);

        concatenated.getBlock(0) = dc1;
        concatenated.getBlock(1) = dc2;
        return concatenated;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class DataContainer<float>;
    template class DataContainer<std::complex<float>>;
    template class DataContainer<double>;
    template class DataContainer<std::complex<double>>;
    template class DataContainer<index_t>;

    template DataContainer<float> concatenate<float>(const DataContainer<float>&,
                                                     const DataContainer<float>&);
    template DataContainer<double> concatenate<double>(const DataContainer<double>&,
                                                       const DataContainer<double>&);
    template DataContainer<std::complex<float>>
        concatenate<std::complex<float>>(const DataContainer<std::complex<float>>&,
                                         const DataContainer<std::complex<float>>&);
    template DataContainer<std::complex<double>>
        concatenate<std::complex<double>>(const DataContainer<std::complex<double>>&,
                                          const DataContainer<std::complex<double>>&);

} // namespace elsa
