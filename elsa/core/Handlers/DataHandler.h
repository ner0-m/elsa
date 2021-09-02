#pragma once

#include "elsaDefines.h"
#include "Cloneable.h"
#include "Error.h"
#include "ExpressionPredicates.h"

#ifdef ELSA_CUDA_VECTOR
#include "Quickvec.h"
#endif

#include <Eigen/Core>

namespace elsa
{

    class DataDescriptor;

    /**
     * @brief Base class encapsulating data handling. The data is stored transparently, for example
     * on CPU or GPU.
     *
     * @author David Frank - initial code
     * @author Tobias Lasser - modularization, modernization
     * @author Nikola Dinev - add block support
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
    protected:
        /// convenience typedef for the Eigen::Matrix data vector
        using DataVector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

        /// convenience typedef for the Eigen::Map
        using DataMap_t = Eigen::Map<DataVector_t>;

    public:
        /// convenience typedef to access data type that is internally stored
        using value_type = data_t;

        /// return the size of the stored data (i.e. number of elements in linearized data vector)
        virtual index_t getSize() const = 0;

        /// return the index-th element of the data vector (not bounds-checked!)
        virtual data_t& operator[](index_t index) = 0;

        /// return the index-th element of the data vector as read-only (not bounds-checked!)
        virtual const data_t& operator[](index_t index) const = 0;

        /// return the dot product of the data vector with vector v
        virtual data_t dot(const DataHandler<data_t>& v) const = 0;

        /// return the squared l2 norm of the data vector (dot product with itself)
        virtual GetFloatingPointType_t<data_t> squaredL2Norm() const = 0;

        /// return the l2 norm of the data vector (square root of dot product with itself)
        virtual GetFloatingPointType_t<data_t> l2Norm() const = 0;

        /// return the l0 pseudo-norm of the data vector (number of non-zero values)
        virtual index_t l0PseudoNorm() const = 0;

        /// return the l1 norm of the data vector (sum of absolute values)
        virtual GetFloatingPointType_t<data_t> l1Norm() const = 0;

        /// return the linf norm of the data vector (maximum of absolute values)
        virtual GetFloatingPointType_t<data_t> lInfNorm() const = 0;

        /// return the sum of all elements of the data vector
        virtual data_t sum() const = 0;

        /// in-place create the fourier transformed of the data vector
        virtual DataHandler<data_t>& fft(const DataDescriptor& source_desc) = 0;

        /// in-place create the inverse fourier transformed of the data vector
        virtual DataHandler<data_t>& ifft(const DataDescriptor& source_desc) = 0;

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
                throw InvalidArgumentError("DataHandler: assignment argument has wrong size");

            assign(other);
            return *this;
        }

        /// move assignment operator
        DataHandler<data_t>& operator=(DataHandler<data_t>&& other)
        {
            if (other.getSize() != getSize())
                throw InvalidArgumentError("DataHandler: assignment argument has wrong size");

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
} // namespace elsa
