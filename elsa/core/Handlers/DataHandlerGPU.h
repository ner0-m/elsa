#pragma once

#include "elsaDefines.h"
#include "DataHandler.h"
#include "Quickvec.h"

#include <list>

namespace elsa
{
    // forward declaration, allows mutual friending
    template <typename data_t>
    class DataHandlerMapGPU;

    // forward declaration for friend test function
    template <typename data_t = real_t>
    class DataHandlerGPU;
    // forward declaration, used for testing and defined in test file (declared as friend)
    template <typename data_t>
    long useCount(const DataHandlerGPU<data_t>&);

    /**
     * \brief Class representing and owning a vector stored in CPU main memory (using
     * Eigen::Matrix).
     *
     * \tparam data_t - data type that is stored, defaulting to real_t.
     *
     * \author David Frank - main code
     * \author Tobias Lasser - modularization and modernization
     * \author Nikola Dinev - integration of map and copy-on-write concepts
     *
     * The class implements copy-on-write. Therefore any non-const functions should call the
     * detach() function first to trigger the copy-on-write mechanism.
     *
     * DataHandlerGPU and DataHandlerMapCPU are mutual friend classes allowing for the vectorization
     * of arithmetic operations with the help of Eigen. A strong bidirectional link exists
     * between the two classes. A Map is associated with the DataHandlerGPU from which it was
     * created for the entirety of its lifetime. If the DataHandlerGPU starts managing a new vector
     * (e.g. through a call to detach()), all associated Maps will also be updated.
     */
    template <typename data_t>
    class DataHandlerGPU : public DataHandler<data_t>
    {
        /// for enabling accessData()
        friend DataContainer<data_t>;

        /// declare DataHandlerMapGPU as friend, allows the use of Eigen for improved performance
        friend DataHandlerMapGPU<data_t>;

        /// used for testing only and defined in test file
        friend long useCount<>(const DataHandlerGPU<data_t>& dh);

        /// friend constexpr function to implement expression templates
        template <bool GPU, class Operand, std::enable_if_t<isDataContainer<Operand>, int>>
        friend constexpr auto evaluateOrReturn(Operand const& operand);

    protected:
        /// convenience typedef for the Eigen::Matrix data vector
        using DataVector_t = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;

        /// convenience typedef for the Eigen::Map
        using DataMap_t = Eigen::Map<DataVector_t>;

    public:
        /// delete default constructor (having no information makes no sense)
        DataHandlerGPU() = delete;

        /// default destructor
        ~DataHandlerGPU() override;

        /**
         * \brief Constructor initializing an appropriately sized vector with zeros
         *
         * \param[in] size of the vector
         * \param[in] initialize - set to false if you do not need initialization with zeros
         * (default: true)
         *
         * \throw std::invalid_argument if the size is non-positive
         */
        explicit DataHandlerGPU(index_t size);

        /**
         * \brief Constructor initializing a data vector with a given vector
         *
         * \param[in] vector that is used for initializing the data
         */
        explicit DataHandlerGPU(DataVector_t const& vector);

        /**
         * \brief Constructor initializing a data vector with a given vector
         *
         * \param[in] vector that is used for initializing the data
         */
        explicit DataHandlerGPU(quickvec::Vector<data_t> const& vector);
        /// copy constructor
        DataHandlerGPU(const DataHandlerGPU<data_t>& other);

        /// move constructor
        DataHandlerGPU(DataHandlerGPU<data_t>&& other) noexcept;

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

        /// copy assign another DataHandlerGPU
        DataHandlerGPU<data_t>& operator=(const DataHandlerGPU<data_t>& v);

        /// move assign another DataHandlerGPU
        DataHandlerGPU<data_t>& operator=(DataHandlerGPU<data_t>&& v);

        /// lift copy and move assignment operators from base class
        using DataHandler<data_t>::operator=;

        /// compute in-place element-wise addition of another vector v
        DataHandler<data_t>& operator+=(const DataHandler<data_t>& v) override;

        /// compute in-place element-wise subtraction of another vector v
        DataHandler<data_t>& operator-=(const DataHandler<data_t>& v) override;

        /// compute in-place element-wise multiplication by another vector v
        DataHandler<data_t>& operator*=(const DataHandler<data_t>& v) override;

        /// compute in-place element-wise division by another vector v
        DataHandler<data_t>& operator/=(const DataHandler<data_t>& v) override;

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
        /// the vector storing the data
        std::shared_ptr<quickvec::Vector<data_t>> _data;

        /// list of DataHandlerMaps referring to blocks of this
        std::list<DataHandlerMapGPU<data_t>*> _associatedMaps;

        /// implement the polymorphic clone operation
        DataHandlerGPU<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const DataHandler<data_t>& other) const override;

        /// copy the data stored in other
        void assign(const DataHandler<data_t>& other) override;

        /// move the data stored in other if other is of the same type, otherwise copy the data
        void assign(DataHandler<data_t>&& other) override;

        /// return non-const version of data
        quickvec::Vector<data_t> accessData();

        /// return const version of data
        quickvec::Vector<data_t> accessData() const;

    private:
        /// creates the deep copy for the copy-on-write mechanism
        void detach();

        /// same as detach() but leaving an uninitialized block of numberOfElements elements
        /// starting at index startIndex
        void detachWithUninitializedBlock(index_t startIndex, index_t numberOfElements);

        /// change the vector being handled
        void attach(const std::shared_ptr<quickvec::Vector<data_t>>& data);

        /// change the vector being handled (rvalue version)
        void attach(std::shared_ptr<quickvec::Vector<data_t>>&& data);
    };
} // namespace elsa