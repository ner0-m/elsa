#pragma once

#include "elsaDefines.h"
#include "DataHandler.h"
#include "Quickvec.h"
#include "Badge.hpp"

#include <Eigen/Core>

#include <list>

namespace elsa
{
    /// forward declaration, allows mutual friending
    template <typename data_t>
    class DataHandlerGPU;

    /**
     * @brief Class referencing a vector stored in GPU main memory, or a part thereof
     *
     * @tparam data_t data type of vector
     *
     * @author David Frank - main code
     * @author Tobias Lasser - modularization, fixes
     * @author Nikola Dinev - integration with the copy-on-write mechanism
     * @author Jens Petit - adaption of CPU version for GPU
     *
     * This class does not own or manage its own memory. It is bound to a DataHandlerGPU (the data
     * owner) at its creation, and serves as a reference to a sequential block of memory owned by
     * the DataHandlerGPU. As such, changes to the Map will affect the DataHandlerGPU and vice
     * versa.
     *
     * Maps do not support move assignment, and remain bound to the original data owner until
     * destructed.
     *
     * Maps provide only limited support for copy-on-write. Unless the Map is referencing the
     * entirety of the vector managed by the data owner, assigning to the Map or cloning will always
     * trigger a deep copy.
     *
     * Cloning a Map produces a new DataHandlerGPU, managing a new chunk of memory. The contents of
     * the memory are equivalent to the contents of the block referenced by the Map, but the two are
     * not associated.
     */
    template <typename data_t = real_t>
    class DataHandlerMapGPU : public DataHandler<data_t>
    {
        /// declare DataHandlerGPU as friend, allows the use of Eigen for improved performance
        friend class DataHandlerGPU<data_t>;

        /// friend constexpr function to implement expression templates
        template <bool GPU, class Operand, std::enable_if_t<isDataContainer<Operand>, int>>
        friend constexpr auto evaluateOrReturn(Operand const& operand);

        /// for enabling accessData()
        friend DataContainer<data_t>;

    public:
        /**
         * @brief Construct a DataHandlerMapGPU referencing a sequential block of data owned by
         * DataHandlerGPU. Only accessible by DataHandlerGPU
         *
         * @param[in] badge Badge that it can only be called
         * @param[in] dataOwner pointer to the DataHandlerGPU owning the data vector
         * @param[in] data pointer to start of segment
         * @param[in] n number of elements in block
         */
        DataHandlerMapGPU(Badge<DataHandlerGPU<data_t>> badge, DataHandlerGPU<data_t>* dataOwner,
                          data_t* data, index_t n);

        /**
         * @brief Construct a DataHandlerMapGPU referencing a sequential block of data owned by
         * DataHandlerGPU. Only accessible by DataHandlerMapGPU.
         *
         * @param[in] badge Badge that it can only be called
         * @param[in] dataOwner pointer to the DataHandlerGPU owning the data vector
         * @param[in] data pointer to start of segment
         * @param[in] n number of elements in block
         */
        DataHandlerMapGPU(Badge<DataHandlerMapGPU<data_t>> badge, DataHandlerGPU<data_t>* dataOwner,
                          data_t* data, index_t n);

        /// copy constructor
        DataHandlerMapGPU(const DataHandlerMapGPU<data_t>& other);

        /// default move constructor
        DataHandlerMapGPU(DataHandlerMapGPU<data_t>&& other) = default;

        /// default destructor
        ~DataHandlerMapGPU() override;

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

        /// return the l2 norm of the data vector (square root of the dot product with itself)
        GetFloatingPointType_t<data_t> l2Norm() const override;

        /// return the l0 pseudo-norm of the data vector (number of non-zero values)
        index_t l0PseudoNorm() const override;

        /// return the l1 norm of the data vector (sum of absolute values)
        GetFloatingPointType_t<data_t> l1Norm() const override;

        /// return the linf norm of the data vector (maximum of absolute values)
        GetFloatingPointType_t<data_t> lInfNorm() const override;

        /// return the sum of all elements of the data vector
        data_t sum() const override;

        /// return the min of all elements of the data vector
        data_t minElement() const override;

        /// return the max of all elements of the data vector
        data_t maxElement() const override;

        /// create the fourier transformed of the data vector
        DataHandler<data_t>& fft(const DataDescriptor& source_desc, FFTNorm norm) override;

        /// create the inverse fourier transformed of the data vector
        DataHandler<data_t>& ifft(const DataDescriptor& source_desc, FFTNorm norm) override;

        /// compute in-place element-wise addition of another vector v
        DataHandler<data_t>& operator+=(const DataHandler<data_t>& v) override;

        /// compute in-place element-wise subtraction of another vector v
        DataHandler<data_t>& operator-=(const DataHandler<data_t>& v) override;

        /// compute in-place element-wise multiplication by another vector v
        DataHandler<data_t>& operator*=(const DataHandler<data_t>& v) override;

        /// compute in-place element-wise division by another vector v
        DataHandler<data_t>& operator/=(const DataHandler<data_t>& v) override;

        /// copy assign another DataHandlerMapGPU to this, other types handled in assign()
        DataHandlerMapGPU<data_t>& operator=(const DataHandlerMapGPU<data_t>& v);

        DataHandlerMapGPU<data_t>& operator=(DataHandlerMapGPU<data_t>&&) = default;

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
        quickvec::Vector<data_t> _map;

        /// pointer to the data-owning handler
        DataHandlerGPU<data_t>* _dataOwner;

        /// handle to this in the list of Maps associated with the data-owning handler
        typename std::list<DataHandlerMapGPU<data_t>*>::iterator _handle;

        /// implement the polymorphic clone operation
        DataHandlerGPU<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const DataHandler<data_t>& other) const override;

        void assign(const DataHandler<data_t>& other) override;

        void assign(DataHandler<data_t>&& other) override;

        /// return non-const version of the data
        quickvec::Vector<data_t> accessData();

        /// return const version of the data
        quickvec::Vector<data_t> accessData() const;

    private:
        /**
         * @brief Construct a DataHandlerMapGPU referencing a sequential block of data owned by
         * DataHandlerGPU. Private implementation, called by multiple different callers
         * which are granted access via a `Badge`
         *
         * @param[in] dataOwner pointer to the DataHandlerGPU owning the data vector
         * @param[in] data pointer to start of segment
         * @param[in] n number of elements in block
         */
        DataHandlerMapGPU(DataHandlerGPU<data_t>* dataOwner, data_t* data, index_t n);
    };
} // namespace elsa
