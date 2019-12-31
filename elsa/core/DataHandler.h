#pragma once

#include "elsaDefines.h"
#include "Cloneable.h"

namespace elsa
{

    /**
     * \brief Base class encapsulating data handling. The data is stored transparently, for example
     * on CPU or GPU.
     *
     * \author David Frank - initial code
     * \author Tobias Lasser - modularization, modernization
     * \author Nikola Dinev - add block support
     *
     * This abstract base class serves as an interface for data handlers, which encapsulate the
     * actual data being stored e.g. in main memory of the CPU or in various memory types of GPUs.
     * The data itself is treated as a vector, i.e. an array of data_t elements (which usually comes
     * from linearized n-dimensional signals).
     *
     * Caveat: If data is not stored in main memory (e.g. on GPUs), then some operations may trigger
     * an automatic synchronization of GPU to main memory. Please see the GPU-based handlers'
     * documentation for details.
     */
    template <typename data_t = real_t>
    class DataHandler : public Cloneable<DataHandler<data_t>>
    {
    public:
        /// return the size of the stored data (i.e. number of elements in linearized data vector)
        virtual index_t getSize() const = 0;

        /// return the index-th element of the data vector (not bounds-checked!)
        virtual data_t& operator[](index_t index) = 0;

        /// return the index-th element of the data vector as read-only (not bounds-checked!)
        virtual const data_t& operator[](index_t index) const = 0;

        /// return the dot product of the data vector with vector v
        virtual data_t dot(const DataHandler<data_t>& v) const = 0;

        /// return the squared l2 norm of the data vector (dot product with itself)
        virtual data_t squaredL2Norm() const = 0;

        /// return the l1 norm of the data vector (sum of absolute values)
        virtual data_t l1Norm() const = 0;

        /// return the linf norm of the data vector (maximum of absolute values)
        virtual data_t lInfNorm() const = 0;

        /// return the sum of all elements of the data vector
        virtual data_t sum() const = 0;

        /// return a new DataHandler with element-wise squared values of this one
        virtual std::unique_ptr<DataHandler<data_t>> square() const = 0;

        /// return a new DataHandler with element-wise square roots of this one
        virtual std::unique_ptr<DataHandler<data_t>> sqrt() const = 0;

        /// return a new DataHandler with element-wise exponentials of this one
        virtual std::unique_ptr<DataHandler<data_t>> exp() const = 0;

        /// return a new DataHandler with element-wise logarithms of this one
        virtual std::unique_ptr<DataHandler<data_t>> log() const = 0;

        /// compute in-place element-wise addition of another vector v
        virtual DataHandler<data_t>& operator+=(const DataHandler<data_t>& v) = 0;

        /// compute in-place element-wise subtraction of another vector v
        virtual DataHandler<data_t>& operator-=(const DataHandler<data_t>& v) = 0;

        /// compute in-place element-wise multiplication by another vector v
        virtual DataHandler<data_t>& operator*=(const DataHandler<data_t>& v) = 0;

        /// compute in-place element-wise division by another vector v
        virtual DataHandler<data_t>& operator/=(const DataHandler<data_t>& v) = 0;

        /// compute in-place addition of a scalar
        virtual DataHandler<data_t>& operator+=(data_t scalar) = 0;

        /// compute in-place subtraction of a scalar
        virtual DataHandler<data_t>& operator-=(data_t scalar) = 0;

        /// compute in-place multiplication by a scalar
        virtual DataHandler<data_t>& operator*=(data_t scalar) = 0;

        /// compute in-place division by a scalar
        virtual DataHandler<data_t>& operator/=(data_t scalar) = 0;

        /// assign a scalar to all elements of the data vector
        virtual DataHandler<data_t>& operator=(data_t scalar) = 0;

        /// copy assignment operator
        DataHandler<data_t>& operator=(const DataHandler<data_t>& other)
        {
            if (other.getSize() != getSize())
                throw std::invalid_argument("DataHandler: assignment argument has wrong size");

            assign(other);
            return *this;
        }

        /// move assignment operator
        DataHandler<data_t>& operator=(DataHandler<data_t>&& other)
        {
            if (other.getSize() != getSize())
                throw std::invalid_argument("DataHandler: assignment argument has wrong size");

            assign(std::move(other));
            return *this;
        }

        /// return a reference to the sequential block starting at startIndex and containing
        /// numberOfElements elements
        virtual std::unique_ptr<DataHandler<data_t>> getBlock(index_t startIndex,
                                                              index_t numberOfElements) = 0;

        /// return a const reference to the sequential block starting at startIndex and containing
        /// numberOfElements elements
        virtual std::unique_ptr<const DataHandler<data_t>>
            getBlock(index_t startIndex, index_t numberOfElements) const = 0;

    protected:
        /// slow element-wise dot product fall-back for when DataHandler types do not match
        data_t slowDotProduct(const DataHandler<data_t>& v) const
        {
            data_t result = 0;
            for (index_t i = 0; i < getSize(); ++i)
                result += (*this)[i] * v[i];
            return result;
        }

        /// slow element-wise addition fall-back for when DataHandler types do not match
        void slowAddition(const DataHandler<data_t>& v)
        {
            for (index_t i = 0; i < getSize(); ++i)
                (*this)[i] += v[i];
        }

        /// slow element-wise subtraction fall-back for when DataHandler types do not match
        void slowSubtraction(const DataHandler<data_t>& v)
        {
            for (index_t i = 0; i < getSize(); ++i)
                (*this)[i] -= v[i];
        }

        /// slow element-wise multiplication fall-back for when DataHandler types do not match
        void slowMultiplication(const DataHandler<data_t>& v)
        {
            for (index_t i = 0; i < getSize(); ++i)
                (*this)[i] *= v[i];
        }

        /// slow element-wise division fall-back for when DataHandler types do not match
        void slowDivision(const DataHandler<data_t>& v)
        {
            for (index_t i = 0; i < getSize(); ++i)
                (*this)[i] /= v[i];
        }

        /// slow element-wise assignment fall-back for when DataHandler types do not match
        void slowAssign(const DataHandler<data_t>& other)
        {
            for (index_t i = 0; i < getSize(); ++i)
                (*this)[i] = other[i];
        }

        /// derived classes should override this method to implement copy assignment
        virtual void assign(const DataHandler<data_t>& other) = 0;

        /// derived classes should override this method to implement move assignment
        virtual void assign(DataHandler<data_t>&& other) = 0;
    };

    /// element-wise addition of two DataHandlers
    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> operator+(const DataHandler<data_t>& left,
                                                   const DataHandler<data_t>& right)
    {
        auto result = left.clone();
        *result += right;
        return result;
    }

    /// element-wise subtraction of two DataHandlers
    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> operator-(const DataHandler<data_t>& left,
                                                   const DataHandler<data_t>& right)
    {
        auto result = left.clone();
        *result -= right;
        return result;
    }

    /// element-wise multiplication of two DataHandlers
    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> operator*(const DataHandler<data_t>& left,
                                                   const DataHandler<data_t>& right)
    {
        auto result = left.clone();
        *result *= right;
        return result;
    }

    /// element-wise division of two DataHandlers
    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> operator/(const DataHandler<data_t>& left,
                                                   const DataHandler<data_t>& right)
    {
        auto result = left.clone();
        *result /= right;
        return result;
    }

    /// addition of DataHandler with scalar
    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> operator+(data_t left, const DataHandler<data_t>& right)
    {
        auto result = right.clone();
        *result += left;
        return result;
    }

    /// addition of scalar with DataHandler
    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> operator+(const DataHandler<data_t>& left, data_t right)
    {
        auto result = left.clone();
        *result += right;
        return result;
    }

    /// subtraction of DataHandler from scalar
    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> operator-(data_t left, const DataHandler<data_t>& right)
    {
        auto result = right.clone();
        *result *= -1;
        *result += left;
        return result;
    }

    /// subtraction of scalar from DataHandler
    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> operator-(const DataHandler<data_t>& left, data_t right)
    {
        auto result = left.clone();
        *result -= right;
        return result;
    }

    /// multiplication of DataHandler with scalar
    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> operator*(data_t left, const DataHandler<data_t>& right)
    {
        auto result = right.clone();
        *result *= left;
        return result;
    }

    /// multiplication of scalar with DataHandler
    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> operator*(const DataHandler<data_t>& left, data_t right)
    {
        auto result = left.clone();
        *result *= right;
        return result;
    }

    /// division of scalar by DataHandler
    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> operator/(data_t left, const DataHandler<data_t>& right)
    {
        auto result = right.clone();
        *result = left;
        *result /= right;
        return result;
    }

    /// division of DataHandler by scalar
    template <typename data_t>
    std::unique_ptr<DataHandler<data_t>> operator/(const DataHandler<data_t>& left, data_t right)
    {
        auto result = left.clone();
        *result /= right;
        return result;
    }

} // namespace elsa
