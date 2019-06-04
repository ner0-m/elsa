#include "DataHandlerCPU.h"

namespace elsa
{

    template <typename data_t>
    DataHandlerCPU<data_t>::DataHandlerCPU(index_t size, bool initialize)
        : _data(size)
    {
        if (initialize)
            _data.setZero();
    }

    template <typename data_t>
    DataHandlerCPU<data_t>::DataHandlerCPU(DataVector_t vector)
        : _data{vector}
    {
    }


    template <typename data_t>
    index_t DataHandlerCPU<data_t>::getSize() const
    {
        return static_cast<index_t>(_data.size());
    }


    template <typename data_t>
    data_t& DataHandlerCPU<data_t>::operator[](index_t index)
    {
        return _data[index];
    }

    template <typename data_t>
    const data_t& DataHandlerCPU<data_t>::operator[](index_t index) const
    {
        return _data[index];
    }


    template <typename data_t>
    data_t DataHandlerCPU<data_t>::dot(const DataHandler<data_t>& v) const
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandlerCPU: dot product argument has wrong size");

        // if the other handler is not CPU, use the slow element-wise fallback version of dot product
        auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&v);
        if (!otherHandler)
            return this->slowDotProduct(v);

        return _data.dot(otherHandler->_data);
    }

    template <typename data_t>
    data_t DataHandlerCPU<data_t>::squaredNorm() const
    {
        return _data.squaredNorm();
    }

    template <typename data_t>
    data_t DataHandlerCPU<data_t>::sum() const
    {
        return _data.sum();
    }


    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> DataHandlerCPU<data_t>::square() const
    {
        auto result = std::make_unique<DataHandlerCPU<data_t>>(getSize(), false);
        result->_data = _data.array().square();

        return result;
    }

    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> DataHandlerCPU<data_t>::sqrt() const
    {
        auto result = std::make_unique<DataHandlerCPU<data_t>>(getSize(), false);
        result->_data = _data.array().sqrt();

        return result;
    }

    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> DataHandlerCPU<data_t>::exp() const
    {
        auto result = std::make_unique<DataHandlerCPU<data_t>>(getSize(), false);
        result->_data = _data.array().exp();

        return result;
    }

    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> DataHandlerCPU<data_t>::log() const
    {
        auto result = std::make_unique<DataHandlerCPU<data_t>>(getSize(), false);
        result->_data = _data.array().log();

        return result;
    }


    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator+=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: addition argument has wrong size");

        // if the other handler is not CPU, use the slow element-wise fallback version of addition
        auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&v);
        if (!otherHandler) {
            this->slowAddition(v);
            return *this;
        }

        _data += otherHandler->_data;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator-=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: subtraction argument has wrong size");

        // if the other handler is not CPU, use the slow element-wise fallback version of subtraction
        auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&v);
        if (!otherHandler) {
            this->slowSubtraction(v);
            return *this;
        }

        _data -= otherHandler->_data;
        return *this;

    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator*=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: multiplication argument has wrong size");

        // if the other handler is not CPU, use the slow element-wise fallback version of multiplication
        auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&v);
        if (!otherHandler) {
            this->slowMultiplication(v);
            return *this;
        }

        _data.array() *= otherHandler->_data.array();
        return *this;

    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator/=(const DataHandler<data_t>& v)
    {
        if (v.getSize() != getSize())
            throw std::invalid_argument("DataHandler: division argument has wrong size");

        // if the other handler is not CPU, use the slow element-wise fallback version of division
        auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&v);
        if (!otherHandler) {
            this->slowDivision(v);
            return *this;
        }

        _data.array() /= otherHandler->_data.array();
        return *this;

    }


    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator+=(data_t scalar)
    {
        _data.array() += scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator-=(data_t scalar)
    {
        _data.array() -= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator*=(data_t scalar)
    {
        _data.array() *= scalar;
        return *this;
    }

    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator/=(data_t scalar)
    {
        _data.array() /= scalar;
        return *this;
    }


    template <typename data_t>
    DataHandler<data_t>& DataHandlerCPU<data_t>::operator=(data_t scalar)
    {
        _data = DataVector_t::Constant(getSize(), scalar);
        return *this;
    }



    template <typename data_t>
    DataHandlerCPU<data_t>* DataHandlerCPU<data_t>::cloneImpl() const {
        return new DataHandlerCPU<data_t>(*this);
    }

    template <typename data_t>
    bool DataHandlerCPU<data_t>::isEqual(const DataHandler<data_t> &other) const {
        auto otherHandler = dynamic_cast<const DataHandlerCPU*>(&other);
        if (!otherHandler)
            return false;

        if (_data.size() != otherHandler->_data.size())
            return false;

        if (_data != otherHandler->_data)
            return false;

        return true;
    }


    // ------------------------------------------
    // explicit template instantiation
    template class DataHandlerCPU<float>;
    template class DataHandlerCPU<std::complex<float>>;
    template class DataHandlerCPU<double>;
    template class DataHandlerCPU<std::complex<double>>;
    template class DataHandlerCPU<index_t>;

} // namespace elsa
