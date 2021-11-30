#pragma once

#include "elsaDefines.h"
#include "DataHandler.h"

#include "AccessPointer.hpp"

#include <Eigen/Core>

#include <list>

namespace elsa
{
    namespace detail
    {
        template <typename T>
        struct DataHandlerCPUTraits;

        template <typename data_t>
        struct DataHandlerCPUTraits<DataHandler<data_t>> {
            using Scalar = data_t;
        };

        /**
         * @brief CRTP class to share most of the code necessary for the CPU handlers (i.e. the
         * handlers wrapping an Eigen vector). The child classes are expected to override the
         * `write()` and `read()` methods, which return an (const) Eigen map, which is used to call
         * all the necessary functionality.
         *
         * The child classes can choose what to do before any read or write (i.e. copy-on-write)
         *
         * NOTE: This class should not be instantiated (and it shouldn't be possible), as this is
         * only do de-duplicate code.
         *
         * The class implements all but `clone`, and `getBlock`. They should be implemented
         * in the child classes
         *
         * @tparam Derived - CRTP Derived template
         *
         * @author David Frank - initial code and idea
         */
        template <typename Derived>
        class DataHandlerCPUBase
            : public DataHandler<typename DataHandlerCPUTraits<Derived>::Scalar>
        {
        public:
            using data_t = typename DataHandlerCPUTraits<Derived>::Scalar;

            /// convenience typedef for the Eigen::Map
            using DataMap_t = Eigen::Map<Vector_t<data_t>>;

            /// convenience typedef for the const Eigen::Map
            using ConstDataMap_t = const Eigen::Map<Vector_t<data_t>>;

            DataMap_t write();

            ConstDataMap_t read() const;

            ~DataHandlerCPUBase() override = default;

            using DataHandler<data_t>::operator=;

            /// return the size of the vector
            index_t getSize() const override;

            /// return the index-th element of the data vector (not bounds checked!)
            data_t& operator[](index_t index) override;

            /// return the index-th element of the data vector (not bounds checked!)
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

            /// return non-const version of data
            DataMap_t accessData();

            /// return const version of data
            ConstDataMap_t accessData() const;

        private:
            /// implement the polymorphic comparison operation
            auto isEqual(const DataHandler<data_t>& other) const -> bool override;

            /// copy the data stored in other
            void assign(const DataHandler<data_t>& other) override;

            /// move the data stored in other if other is of the same type, otherwise copy the data
            void assign(DataHandler<data_t>&& other) override;

            template <bool is_forward>
            void base_fft(const DataDescriptor& source_desc, FFTNorm norm = FFTNorm::BACKWARD);

            Derived& self();

            const Derived& self() const;
        };

    } // namespace detail

    // forward declaration for traits
    template <typename data_t = real_t>
    class DataHandlerCPU;

    template <typename data_t = real_t>
    class DataHandlerMapCPU;

    namespace detail
    {
        template <typename data_t>
        struct DataHandlerCPUTraits<DataHandlerCPU<data_t>> {
            using Scalar = data_t;
        };

        template <typename data_t>
        struct DataHandlerCPUTraits<DataHandlerMapCPU<data_t>> {
            using Scalar = data_t;
        };
    } // namespace detail

    /**
     * @brief Class representing an owning vector stored in CPU main memory (using
     * Eigen::Matrix).
     *
     * @tparam data_t - data type that is stored, defaulting to real_t.
     *
     * @author David Frank - main code, refactor to CRTP
     * @author Tobias Lasser - modularization and modernization
     * @author Nikola Dinev - integration of map and copy-on-write concepts
     */
    template <typename data_t>
    class DataHandlerCPU final : public detail::DataHandlerCPUBase<DataHandlerCPU<data_t>>
    {
        using Base = detail::DataHandlerCPUBase<DataHandlerCPU<data_t>>;
        using DataMap_t = typename Base::DataMap_t;
        using ConstDataMap_t = typename Base::ConstDataMap_t;

    public:
        using Base::operator=;

        DataMap_t write();

        ConstDataMap_t read() const;

        explicit DataHandlerCPU(index_t size);

        explicit DataHandlerCPU(Vector_t<data_t> vec);

        DataHandlerCPU(const DataHandlerCPU& other);

        DataHandlerCPU(DataHandlerCPU&& other) noexcept;

        DataHandlerCPU<data_t>& operator=(const DataHandlerCPU& other);

        DataHandlerCPU<data_t>& operator=(DataHandlerCPU&& other) noexcept;

        friend void swap(DataHandlerCPU<data_t>& x, DataHandlerCPU<data_t>& y)
        {
            std::swap(x.data_, y.data_);
        }

        auto getBlock(index_t startIndex, index_t numberOfElements)
            -> std::unique_ptr<DataHandler<data_t>> override;

        auto getBlock(index_t startIndex, index_t numberOfElements) const
            -> std::unique_ptr<const DataHandler<data_t>> override;

        /// implement the polymorphic clone operation
        auto cloneImpl() const -> DataHandlerCPU<data_t>* override
        {
            return new DataHandlerCPU(data_.read());
        }

        auto use_count() const -> index_t { return data_.use_count(); }

        friend bool operator==(const DataHandlerCPU<data_t>& x,
                               const DataHandlerCPU<data_t>& y) noexcept
        {
            return x.getSize() == y.getSize() && x.data_ == y.data_;
        }

        friend bool operator!=(const DataHandlerCPU& x, const DataHandlerCPU& y) noexcept
        {
            return !(x == y);
        }

    private:
        CopyOnWritePointer<Vector_t<data_t>> data_;
    };

    /**
     * @brief Class representing a view/map into a vector stored in CPU main memory (using
     * Eigen::Matrix).
     *
     * @tparam data_t - data type that is stored, defaulting to real_t.
     *
     * @author David Frank - main code, refactor to CRTP
     * @author Tobias Lasser - modularization and modernization
     * @author Nikola Dinev - integration of map and copy-on-write concepts
     */
    template <typename data_t>
    class DataHandlerMapCPU final : public detail::DataHandlerCPUBase<DataHandlerMapCPU<data_t>>
    {
        using Base = detail::DataHandlerCPUBase<DataHandlerMapCPU<data_t>>;
        using DataMap_t = typename Base::DataMap_t;
        using ConstDataMap_t = typename Base::ConstDataMap_t;

    public:
        DataMap_t write();

        ConstDataMap_t read() const;

        DataHandlerMapCPU(DataHandlerCPU<data_t>& ref, index_t start, index_t end)
            : ptr_(&ref), start_(start), end_(end)
        {
        }

        DataHandlerMapCPU(DataHandlerCPU<data_t>* ptr, index_t start, index_t end)
            : ptr_(ptr), start_(start), end_(end)
        {
        }

        DataHandlerMapCPU(const DataHandlerMapCPU& other);

        DataHandlerMapCPU(DataHandlerMapCPU&& other) noexcept;

        DataHandlerMapCPU<data_t>& operator=(const DataHandlerMapCPU& other);

        DataHandlerMapCPU<data_t>& operator=(DataHandlerMapCPU&& other) noexcept;

        friend void swap(DataHandlerMapCPU<data_t>& x, DataHandlerMapCPU<data_t>& y)
        {
            std::swap(x.ptr_, y.ptr_);
            std::swap(x.start_, y.start_);
            std::swap(x.end_, y.end_);
        }

        auto getBlock(index_t startIndex, index_t numberOfElements) const
            -> std::unique_ptr<const DataHandler<data_t>> override;

        auto getBlock(index_t startIndex, index_t numberOfElements)
            -> std::unique_ptr<DataHandler<data_t>> override;

        /// implement the polymorphic clone operation
        DataHandlerMapCPU* cloneImpl() const override
        {
            return new DataHandlerMapCPU(ptr_, start_, end_);
        }

        friend bool operator==(const DataHandlerMapCPU& x, const DataHandlerMapCPU& y) noexcept
        {
            return x.getSize() == y.getSize() && x.ptr_ && y.ptr_ && *x.ptr_ == *y.ptr_;
        }

        friend bool operator==(const DataHandlerCPU<data_t>& x, const DataHandlerMapCPU& y) noexcept
        {
            return x.getSize() == y.getSize() && y.ptr_ && x == *y.ptr_;
        }

        friend bool operator==(const DataHandlerMapCPU& x, const DataHandlerCPU<data_t>& y) noexcept
        {
            return y == x;
        }

        friend bool operator!=(const DataHandlerMapCPU& x, const DataHandlerMapCPU& y) noexcept
        {
            return !(x == y);
        }

        friend bool operator!=(const DataHandlerCPU<data_t>& x, const DataHandlerMapCPU& y) noexcept
        {
            return !(x == y);
        }

        friend bool operator!=(const DataHandlerMapCPU& x, const DataHandlerCPU<data_t>& y) noexcept
        {
            return !(x == y);
        }

    private:
        DataHandlerCPU<data_t>* ptr_;
        index_t start_;
        index_t end_;
    };

} // namespace elsa
