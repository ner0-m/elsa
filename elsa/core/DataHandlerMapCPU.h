#pragma once

#include "elsa.h"
#include "DataHandler.h"

#include <Eigen/Core>

#include <list>

namespace elsa
{
    /// forward declaration, allows mutual friending
    template <typename data_t>
    class DataHandlerCPU;

    /**
     * \brief Class referencing a vector stored in CPU main memory, or a part thereof (using
     * Eigen::Map)
     *
     * \tparam data_t data type of vector
     *
     * \author David Frank - main code
     * \author Tobias Lasser - modularization, fixes
     * \author Nikola Dinev - integration with the copy-on-write mechanism
     *
     * This class does not own or manage its own memory. It is bound to a DataHandlerCPU (the data
     * owner) at its creation, and serves as a reference to a sequential block of memory owned by
     * the DataHandlerCPU. As such, changes to the Map will affect the DataHandlerCPU and vice
     * versa.
     *
     * Maps do not support move assignment, and remain bound to the original data owner until
     * destructed.
     *
     * Maps provide only limited support for copy-on-write. Unless the Map is referencing the
     * entirety of the vector managed by the data owner, assigning to the Map or cloning will always
     * trigger a deep copy.
     *
     * Cloning a Map produces a new DataHandlerCPU, managing a new chunk of memory. The contents of
     * the memory are equivalent to the contents of the block referenced by the Map, but the two are
     * not associated.
     */
    template <typename data_t = real_t>
    class DataHandlerMapCPU : public DataHandler<data_t>
    {
        /// declare DataHandlerCPU as friend, allows the use of Eigen for improved performance
        friend class DataHandlerCPU<data_t>;

    protected:
        /// convenience typedef for the Eigen::Matrix data vector
        using DataVector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;
        /// convenience typedef for the Eigen::Map
        using DataMap_t = Eigen::Map<DataVector_t>;

    public:
        /// copy constructor
        DataHandlerMapCPU(const DataHandlerMapCPU<data_t>& other);

        /// default move constructor
        DataHandlerMapCPU(DataHandlerMapCPU<data_t>&& other) = default;

        /// default destructor
        ~DataHandlerMapCPU() override;

        /// return the size of the vector
        index_t getSize() const override;

        /// return the index-th element of the data vector (not bounds checked!)
        data_t& operator[](index_t index) override;

        /// return the index-th element of the data vector as read-only (not bound checked!)
        const data_t& operator[](index_t index) const override;

        /// return the dot product of the data vector with vector v
        data_t dot(const DataHandler<data_t>& v) const override;

        /// return the squared l2 norm of the data vector (dot product with itself)
        data_t squaredL2Norm() const override;

        /// return the l1 norm of the data vector (sum of absolute values)
        data_t l1Norm() const override;

        /// return the linf norm of the data vector (maximum of absolute values)
        data_t lInfNorm() const override;

        /// return the sum of all elements of the data vector
        data_t sum() const override;

        /// return a new DataHandler with element-wise squared values of this one
        std::unique_ptr<DataHandler<data_t>> square() const override;

        /// return a new DataHandler with element-wise square roots of this one
        std::unique_ptr<DataHandler<data_t>> sqrt() const override;

        /// return a new DataHandler with element-wise exponentials of this one
        std::unique_ptr<DataHandler<data_t>> exp() const override;

        /// return a new DataHandler with element-wise logarithms of this one
        std::unique_ptr<DataHandler<data_t>> log() const override;

        /// compute in-place element-wise addition of another vector v
        DataHandler<data_t>& operator+=(const DataHandler<data_t>& v) override;

        /// compute in-place element-wise subtraction of another vector v
        DataHandler<data_t>& operator-=(const DataHandler<data_t>& v) override;

        /// compute in-place element-wise multiplication by another vector v
        DataHandler<data_t>& operator*=(const DataHandler<data_t>& v) override;

        /// compute in-place element-wise division by another vector v
        DataHandler<data_t>& operator/=(const DataHandler<data_t>& v) override;

        /// copy assign another DataHandlerMapCPU to this, other types handled in assign()
        DataHandlerMapCPU<data_t>& operator=(const DataHandlerMapCPU<data_t>& v);

        DataHandlerMapCPU<data_t>& operator=(DataHandlerMapCPU<data_t>&&) = default;

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
        /// vector mapping of the data
        DataMap_t _map;

        /// pointer to the data-owning handler
        DataHandlerCPU<data_t>* _dataOwner;

        /// handle to this in the list of Maps associated with the data-owning handler
        typename std::list<DataHandlerMapCPU<data_t>*>::iterator _handle;

        /// implement the polymorphic clone operation
        DataHandlerCPU<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const DataHandler<data_t>& other) const override;

        void assign(const DataHandler<data_t>& other) override;

        void assign(DataHandler<data_t>&& other) override;

    private:
        /**
         * \brief Construct a DataHandlerMapCPU referencing a sequential block of data owned by
         * DataHandlerCPU
         *
         * \param[in] dataOwner pointer to the DataHandlerCPU owning the data vector
         * \param[in] data pointer to start of segment
         * \param[in] n number of elements in block
         */
        DataHandlerMapCPU(DataHandlerCPU<data_t>* dataOwner, data_t* data, index_t n);
    };
} // namespace elsa