#pragma once


#include "elsa.h"
#include "DataDescriptor.h"
#include "DataHandler.h"

#include <memory>

namespace elsa
{

    /**
     * \brief class representing and storing a linearized n-dimensional signal
     *
     * \author Matthias Wieczorek - initial code
     * \author Tobias Lasser - rewrite, modularization, modernization
     * \author David Frank - added DataHandler concept
     *
     * \tparam data_t - data type that is stored in the DataContainer, defaulting to real_t.
     *
     * This class provides a container for a signal that is stored in memory. This signal can
     * be n-dimensional, and will be stored in memory in a linearized fashion. The information
     * on how this linearization is performed is provided by an associated DataDescriptor.
     */
    template <typename data_t = real_t>
    class DataContainer {
    public:
        /**
         * \brief type of the DataHandler used to store the actual data
         *
         * The following handler types are currently supported:
         *    - CPU: data is stored as an Eigen::Matrix in CPU main memory
         *    - MAP: data is not explicitly stored, but using an Eigen::Map to refer to other storage
         */
        enum class DataHandlerType { CPU };

        /// delete default constructor (without metadata there can be no valid container)
        DataContainer() = delete;

        /**
         * \brief Constructor for empty DataContainer, initializing the data with zeros.
         *
         * \param[in] dataDescriptor containing the associated metadata
         */
        explicit DataContainer(const DataDescriptor& dataDescriptor,
                                DataHandlerType handlerType = DataHandlerType::CPU);
         
         
        /**
         * \brief Constructor for DataContainer, initializing it with a RealVector
         * 
         * \param[in] dataDescriptor containing the associated metadata
         * \param[in] data vector containing the initialization data
         */
        DataContainer(const DataDescriptor& dataDescriptor, const RealVector_t& data, 
                       DataHandlerType handlerType = DataHandlerType::CPU);


        /**
         * \brief Copy constructor for DataContainer
         *
         * \param[in] other DataContainer to copy
         */
        DataContainer(const DataContainer<data_t>& other);

        /**
         * \brief copy assignment for DataContainer
         *
         * \param[in] other DataContainer to copy
         */
        DataContainer<data_t>& operator=(const DataContainer<data_t>& other);

        /**
         * \brief Move constructor for DataContainer
         *
         * \param[in] other DataContainer to move from
         */
        DataContainer(DataContainer<data_t>&& other);

        /**
         * \brief Move assignment for DataContainer
         *
         * \param[in] other DataContainer to move from
         */
        DataContainer<data_t>& operator=(DataContainer<data_t>&& other);


        /// return the current DataDescriptor
        const DataDescriptor& getDataDescriptor() const;

        /// return the size of the stored data (i.e. the number of elements in the linearized signal)
        index_t getSize() const;


        /// return the index-th element of linearized signal (not bounds-checked!)
        data_t& operator[](index_t index);

        /// return the index-th element of the linearized signal as read-only (not bounds-checked!)
        const data_t& operator[](index_t index) const;

        /// return an element by n-dimensional coordinate (not bounds-checked!)
        data_t& operator()(IndexVector_t coordinate);

        /// return an element by n-dimensional coordinate as read-only (not bounds-checked!)
        const data_t& operator()(IndexVector_t coordinate) const;


        /// return the dot product of this signal with the one from container other
        data_t dot(const DataContainer<data_t>& other) const;

        /// return the squared l2 norm of this signal (dot product with itself)
        data_t squaredL2Norm() const;

        /// return the l1 norm of this signal (sum of absolute values)
        data_t l1Norm() const;

        /// return the linf norm of this signal (maximum of absolute values)
        data_t lInfNorm() const;

        /// return the sum of all elements of this signal
        data_t sum() const;


        /// return a new DataContainer with element-wise squared values of this one
        DataContainer<data_t> square() const;

        /// return a new DataContainer with element-wise square roots of this one
        DataContainer<data_t> sqrt() const;

        /// return a new DataContainer with element-wise exponentials of this one
        DataContainer<data_t> exp() const;

        /// return a new DataContainer with element-wise logarithms of this one
        DataContainer<data_t> log() const;


        /// compute in-place element-wise addition of another container
        DataContainer<data_t>& operator+=(const DataContainer<data_t>& dc);

        /// compute in-place element-wise subtraction of another container
        DataContainer<data_t>& operator-=(const DataContainer<data_t>& dc);

        /// compute in-place element-wise multiplication with another container
        DataContainer<data_t>& operator*=(const DataContainer<data_t>& dc);

        /// compute in-place element-wise division by another container
        DataContainer<data_t>& operator/=(const DataContainer<data_t>& dc);


        /// compute in-place addition of a scalar
        DataContainer<data_t>& operator+=(data_t scalar);

        /// compute in-place subtraction of a scalar
        DataContainer<data_t>& operator-=(data_t scalar);

        /// compute in-place multiplication with a scalar
        DataContainer<data_t>& operator*=(data_t scalar);

        /// compute in-place division by a scalar
        DataContainer<data_t>& operator/=(data_t scalar);


        /// assign a scalar to the DataContainer
        DataContainer<data_t>& operator=(data_t scalar);


        /// comparison with another DataContainer
        bool operator==(const DataContainer<data_t>& other) const;

        /// comparison with another DataContainer
        bool operator!=(const DataContainer<data_t>& other) const;


    private:
        /// the current DataDescriptor
        std::unique_ptr<DataDescriptor> _dataDescriptor;
        /// the current DataHandler
        std::unique_ptr<DataHandler<data_t>> _dataHandler;

        /// factory method to create DataHandlers based on handlerType with perfect forwarding of constructor arguments
        template <typename ... Args>
        std::unique_ptr<DataHandler<data_t>> createDataHandler(DataHandlerType handlerType, Args&& ... args);

        /// private constructor accepting a DataDescriptor and a DataHandler
        explicit DataContainer(const DataDescriptor& dataDescriptor, std::unique_ptr<DataHandler<data_t>> dataHandler);
    };


    /// element-wise addition of two DataContainers
    template <typename data_t>
    DataContainer<data_t> operator+(const DataContainer<data_t>& left, const DataContainer<data_t>& right)
    {
        DataContainer<data_t> result(left);
        result += right;
        return result;
    }

    /// element-wise subtraction of two DataContainers
    template <typename data_t>
    DataContainer<data_t> operator-(const DataContainer<data_t>& left, const DataContainer<data_t>& right)
    {
        DataContainer<data_t> result(left);
        result -= right;
        return result;
    }

    /// element-wise multiplication of two DataContainers
    template <typename data_t>
    DataContainer<data_t> operator*(const DataContainer<data_t>& left, const DataContainer<data_t>& right)
    {
        DataContainer<data_t> result(left);
        result *= right;
        return result;
    }

    /// element-wise division of two DataContainers
    template <typename data_t>
    DataContainer<data_t> operator/(const DataContainer<data_t>& left, const DataContainer<data_t>& right)
    {
        DataContainer<data_t> result(left);
        result /= right;
        return result;
    }


    /// addition of DataContainer and scalar
    template <typename data_t>
    DataContainer<data_t> operator+(const DataContainer<data_t>& left, data_t right)
    {
        DataContainer<data_t> result(left);
        result += right;
        return result;
    }

    /// addition of scalar and DataContainer
    template <typename data_t>
    DataContainer<data_t> operator+(data_t left, const DataContainer<data_t>&  right)
    {
        DataContainer<data_t> result(right);
        result += left;
        return result;
    }

    /// subtraction of scalar from DataContainer
    template <typename data_t>
    DataContainer<data_t> operator-(const DataContainer<data_t>& left, data_t right)
    {
        DataContainer<data_t> result(left);
        result -= right;
        return result;
    }

    /// subtraction of DataContainer from scalar
    template <typename data_t>
    DataContainer<data_t> operator-(data_t left, const DataContainer<data_t>& right)
    {
        DataContainer<data_t> result(right);
        result *= -1;
        result += left;
        return result;
    }

    /// multiplication of DataContainer and scalar
    template <typename data_t>
    DataContainer<data_t> operator*(const DataContainer<data_t>& left, data_t right)
    {
        DataContainer<data_t> result(left);
        result *= right;
        return result;
    }

    /// multiplication of scalar and DataContainer
    template <typename data_t>
    DataContainer<data_t> operator*(data_t left, const DataContainer<data_t>&  right)
    {
        DataContainer<data_t> result(right);
        result *= left;
        return result;
    }

    /// division of DataContainer by scalar
    template <typename data_t>
    DataContainer<data_t> operator/(const DataContainer<data_t>& left, data_t right)
    {
        DataContainer<data_t> result(left);
        result /= right;
        return result;
    }

    /// division of DataContainer and scalar
    template <typename data_t>
    DataContainer<data_t> operator/(data_t left, const DataContainer<data_t>&  right)
    {
        DataContainer<data_t> result(right);
        result = left;
        result /= right;
        return result;
    }



} // namespace elsa
