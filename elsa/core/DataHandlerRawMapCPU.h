#pragma once

#include "elsaDefines.h"
#include "DataHandler.h"

#include <Eigen/Core>

#include <list>

namespace elsa
{
    /// forward declaration, allows mutual friending
    template <typename data_t>
    class DataHandlerCPU;

    /**
     * \brief Class referencing raw memory (using Eigen::Map)
     *
     * \tparam data_t data type of vector
     *
     * \author David Tellenbach - main code
     *
     * This class provides a view on raw memory that can be allocated outside of elsa.
     */
    template <typename data_t = real_t>
    class DataHandlerRawMapCPU : public DataHandler<data_t>
    {
    public:
        // Constructor that accepts raw data and its size
        DataHandlerRawMapCPU(data_t* data, index_t n);

        /// copy constructor
        DataHandlerRawMapCPU(const DataHandlerRawMapCPU<data_t>& other);

        /// default move constructor
        DataHandlerRawMapCPU(DataHandlerRawMapCPU<data_t>&& other) = default;

        /// default destructor
        ~DataHandlerRawMapCPU() override;

        /// return the size of the vector
        index_t getSize() const override;

        /// return the index-th element of the data vector (not bounds checked!)
        data_t& operator[](index_t index) override;

        /// return the index-th element of the data vector as read-only (not bound checked!)
        const data_t& operator[](index_t index) const override;

        /// return the dot product of the data vector with vector v
        data_t dot(const DataHandler<data_t>& v) const override;

        /// return the squared l2 norm of the data vector (dot product with itself)
        GetFloatingPointType_t<data_t> squaredL2Norm() const override;

        /// return the l1 norm of the data vector (sum of absolute values)
        GetFloatingPointType_t<data_t> l1Norm() const override;

        /// return the linf norm of the data vector (maximum of absolute values)
        GetFloatingPointType_t<data_t> lInfNorm() const override;

        /// return the sum of all elements of the data vector
        data_t sum() const override;

        /// compute in-place element-wise addition of another vector v
        DataHandler<data_t>& operator+=(const DataHandler<data_t>& v) override;

        /// compute in-place element-wise subtraction of another vector v
        DataHandler<data_t>& operator-=(const DataHandler<data_t>& v) override;

        /// compute in-place element-wise multiplication by another vector v
        DataHandler<data_t>& operator*=(const DataHandler<data_t>& v) override;

        /// compute in-place element-wise division by another vector v
        DataHandler<data_t>& operator/=(const DataHandler<data_t>& v) override;

        /// copy assign another DataHandlerRawMapCPU to this, other types handled in assign()
        DataHandlerRawMapCPU<data_t>& operator=(const DataHandlerRawMapCPU<data_t>& v);

        DataHandlerRawMapCPU<data_t>& operator=(DataHandlerRawMapCPU<data_t>&&) = default;

        /// lift copy and move assignment operators from base class
        using DataHandler<data_t>::operator=;

        /// compute in-place addition of a scalar
        DataHandler<data_t>& operator+=(data_t scalar) override;

        /// compute in-place subtraction of a scalar
        DataHandler<data_t>& operator-=(data_t scalar) override;

        /// compute in-place multiplication by a scalar
        DataHandler<data_t>& operator*=(data_t scalar) override;

        /// compute in-place division by a scalar
        DataHandler<data_t>& operator/=(data_t scalar) override;

        /// assign a scalar to all elements of the data vector
        DataHandler<data_t>& operator=(data_t scalar) override;

        /// return a reference to the sequential block starting at startIndex and containing
        /// numberOfElements elements
        std::unique_ptr<DataHandler<data_t>> getBlock(index_t startIndex,
                                                      index_t numberOfElements) override;

        /// return a const reference to the sequential block starting at startIndex and containing
        /// numberOfElements elements
        std::unique_ptr<const DataHandler<data_t>>
            getBlock(index_t startIndex, index_t numberOfElements) const override;

    protected:
        /// convenience typedef for the Eigen::Matrix data vector
        using DataVector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

        /// convenience typedef for the Eigen::Map
        using DataMap_t = Eigen::Map<DataVector_t>;

        /// vector mapping of the data
        DataMap_t _map;

        /// implement the polymorphic clone operation
        DataHandlerCPU<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const DataHandler<data_t>& other) const override;

        void assign(const DataHandler<data_t>& other) override;

        void assign(DataHandler<data_t>&& other) override;

        /// return non-const version of the data
        DataMap_t accessData();

        /// return const version of the data
        DataMap_t accessData() const;

    private:
        /// declare DataHandlerCPU as friend, allows the use of Eigen for improved performance
        friend class DataHandlerCPU<data_t>;

        /// friend constexpr function to implement expression templates
        template <bool GPU, class Operand, std::enable_if_t<isDataContainer<Operand>, int>>
        friend constexpr auto evaluateOrReturn(Operand const& operand);

        /// for enabling accessData()
        friend DataContainer<data_t>;
    };
} // namespace elsa
