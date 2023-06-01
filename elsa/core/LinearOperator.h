#pragma once

#include "TypeCasts.hpp"
#include "elsaDefines.h"
#include "Cloneable.h"
#include "DataDescriptor.h"
#include "DataContainer.h"

#include <memory>
#include <optional>
#include <vector>

namespace elsa
{

    /**
     * @brief Base class representing a linear operator A. Also implements operator expression
     * functionality.
     *
     * @author Matthias Wieczorek - initial code
     * @author Maximilian Hornung - composite rewrite
     * @author Tobias Lasser - rewrite, modularization, modernization
     *
     * @tparam data_t data type for the domain and range of the operator, defaulting to real_t
     *
     * This class represents a linear operator A, expressed through its apply/applyAdjoint methods,
     * which implement Ax and A^ty for DataContainers x,y of appropriate sizes. Concrete
     * implementations of linear operators will derive from this class and override the
     * applyImpl/applyAdjointImpl methods.
     *
     * LinearOperator also provides functionality to support constructs like the operator expression
     * A^t*B+C, where A,B,C are linear operators. This operator composition is implemented via
     * evaluation trees.
     *
     * LinearOperator and all its derived classes are expected to be light-weight and easily
     * copyable/cloneable, due to the implementation of evaluation trees. Hence any
     * pre-computations/caching should only be done in a lazy manner (e.g. during the first call of
     * apply), and not in the constructor.
     */
    template <typename data_t = real_t>
    class LinearOperator : public Cloneable<LinearOperator<data_t>>
    {
    public:
        /**
         * @brief Constructor for the linear operator A, mapping from domain to range
         *
         * @param[in] domainDescriptor DataDescriptor describing the domain of the operator
         * @param[in] rangeDescriptor DataDescriptor describing the range of the operator
         */
        LinearOperator(const DataDescriptor& domainDescriptor,
                       const DataDescriptor& rangeDescriptor);

        /// default destructor
        ~LinearOperator() override = default;

        /// copy construction
        LinearOperator(const LinearOperator<data_t>& other);
        /// copy assignment
        LinearOperator<data_t>& operator=(const LinearOperator<data_t>& other);

        /// return the domain DataDescriptor
        const DataDescriptor& getDomainDescriptor() const;

        /// return the range DataDescriptor
        const DataDescriptor& getRangeDescriptor() const;

        /**
         * @brief apply the operator A to an element in the operator's domain
         *
         * @param[in] x input DataContainer (in the domain of the operator)
         *
         * @returns Ax DataContainer containing the application of operator A to data x,
         * i.e. in the range of the operator
         *
         * Please note: this method uses apply(x, Ax) to perform the actual operation.
         */
        DataContainer<data_t> apply(const DataContainer<data_t>& x) const;

        /**
         * @brief apply the operator A to an element in the operator's domain
         *
         * @param[in] x input DataContainer (in the domain of the operator)
         * @param[out] Ax output DataContainer (in the range of the operator)
         *
         * Please note: this method calls the method applyImpl that has to be overridden in derived
         * classes. (Why is this method not virtual itself? Because you cannot have a non-virtual
         * function overloading a virtual one [apply with one vs. two arguments]).
         */
        void apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const;

        /**
         * @brief apply the adjoint of operator A to an element of the operator's range
         *
         * @param[in] y input DataContainer (in the range of the operator)
         *
         * @returns A^ty DataContainer containing the application of A^t to data y,
         * i.e. in the domain of the operator
         *
         * Please note: this method uses applyAdjoint(y, Aty) to perform the actual operation.
         */
        DataContainer<data_t> applyAdjoint(const DataContainer<data_t>& y) const;

        /**
         * @brief apply the adjoint of operator A to an element of the operator's range
         *
         * @param[in] y input DataContainer (in the range of the operator)
         * @param[out] Aty output DataContainer (in the domain of the operator)
         *
         * Please note: this method calls the method applyAdjointImpl that has to be overridden in
         * derived classes. (Why is this method not virtual itself? Because you cannot have a
         * non-virtual function overloading a virtual one [applyAdjoint with one vs. two args]).
         */
        void applyAdjoint(const DataContainer<data_t>& y, DataContainer<data_t>& Aty) const;

        /// friend operator+ to support composition of LinearOperators (and its derivatives)
        friend LinearOperator<data_t> operator+(const LinearOperator<data_t>& lhs,
                                                const LinearOperator<data_t>& rhs)
        {
            return LinearOperator(lhs, rhs, CompositeMode::ADD);
        }

        /// friend operator* to support composition of LinearOperators (and its derivatives)
        friend LinearOperator<data_t> operator*(const LinearOperator<data_t>& lhs,
                                                const LinearOperator<data_t>& rhs)
        {
            return LinearOperator(lhs, rhs, CompositeMode::MULT);
        }

        /// friend operator* to support composition of a scalar and a LinearOperator
        friend LinearOperator<data_t> operator*(data_t scalar, const LinearOperator<data_t>& op)
        {
            return LinearOperator(scalar, op);
        }

        /// friend function to return the adjoint of a LinearOperator (and its derivatives)
        friend LinearOperator<data_t> adjoint(const LinearOperator<data_t>& op)
        {
            return LinearOperator(op, true);
        }

        /// friend function to return a leaf node of a LinearOperator (and its derivatives)
        friend LinearOperator<data_t> leaf(const LinearOperator<data_t>& op)
        {
            return LinearOperator(op, false);
        }

    protected:
        /// the data descriptor of the domain of the operator
        std::unique_ptr<DataDescriptor> _domainDescriptor;

        /// the data descriptor of the range of the operator
        std::unique_ptr<DataDescriptor> _rangeDescriptor;

        /// implement the polymorphic clone operation
        LinearOperator<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;

        /// the apply method that has to be overridden in derived classes
        virtual void applyImpl(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const;

        /// the applyAdjoint  method that has to be overridden in derived classes
        virtual void applyAdjointImpl(const DataContainer<data_t>& y,
                                      DataContainer<data_t>& Aty) const;

    private:
        /// pointers to nodes in the evaluation tree
        std::unique_ptr<LinearOperator<data_t>> _lhs{}, _rhs{};

        std::optional<data_t> _scalar = {};

        /// flag whether this is a leaf-node
        bool _isLeaf{false};

        /// flag whether this is a leaf-node to implement an adjoint operator
        bool _isAdjoint{false};

        /// flag whether this is a composite (internal node) of the evaluation tree
        bool _isComposite{false};

        /// enum class denoting the mode of composition (+, *)
        enum class CompositeMode { ADD, MULT, SCALAR_MULT };

        /// variable storing the composition mode (+, *)
        CompositeMode _mode{CompositeMode::MULT};

        /// constructor to produce an adjoint leaf node
        LinearOperator(const LinearOperator<data_t>& op, bool isAdjoint);

        /// constructor to produce a composite (internal node) of the evaluation tree
        LinearOperator(const LinearOperator<data_t>& lhs, const LinearOperator<data_t>& rhs,
                       CompositeMode mode);

        /// constructor to produce a composite (internal node) of the evaluation tree
        LinearOperator(data_t scalar, const LinearOperator<data_t>& op);
    };

    namespace detail
    {
        template <class data_t, class... Ts>
        std::vector<std::unique_ptr<LinearOperator<data_t>>> makeLinearOpVector(Ts&&... ts)
        {
            std::vector<std::unique_ptr<LinearOperator<data_t>>> v;
            v.reserve(sizeof...(ts));

            (v.emplace_back(std::forward<Ts>(ts).clone()), ...);
            return std::move(v);
        }
    } // namespace detail

    /// @brief A thin confinient wrapper around
    /// `std::vector<std::unique_ptr<LinearOperator>>>`. As that they is not
    /// easily handable, we wrap it to make it a regular type. Hence, the type
    /// is both easily copyiable and moveable.
    ///
    /// Further, when iterating over the list the access functions return a
    /// reference to the `LinearOperator` instead of an
    /// `std::unique_ptr<LinearOperator>`.
    template <class data_t>
    class LinearOperatorList
    {
        struct Iterator {
            using iterator =
                typename std::vector<std::unique_ptr<LinearOperator<data_t>>>::iterator;

            using iterator_category = std::forward_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = LinearOperator<data_t>;
            using pointer = value_type*;
            using reference = value_type&;

            Iterator(iterator iter) : cur_(iter) {}

            reference operator*() const { return **cur_; }
            pointer operator->() { return (*cur_).get(); }

            Iterator& operator++()
            {
                ++cur_;
                return *this;
            }

            Iterator operator++(int)
            {
                auto copy = *this;
                ++(*this);
                return copy;
            }

            friend bool operator==(const Iterator& a, const Iterator& b)
            {
                return a.cur_ == b.cur_;
            }

            friend bool operator!=(const Iterator& a, const Iterator& b) { return !(a == b); }

            iterator cur_;
        };

        struct ConstIterator {
            using const_iterator =
                typename std::vector<std::unique_ptr<LinearOperator<data_t>>>::const_iterator;

            using iterator_category = std::forward_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = LinearOperator<data_t>;
            using pointer = const value_type*;
            using reference = const value_type&;

            ConstIterator(const_iterator iter) : cur_(iter) {}

            const_iterator cur_;

            reference operator*() const { return **cur_; }
            pointer operator->() { return (*cur_).get(); }

            ConstIterator& operator++()
            {
                ++cur_;
                return *this;
            }

            ConstIterator operator++(int)
            {
                auto copy = *this;
                ++(*this);
                return copy;
            }

            friend bool operator==(const ConstIterator& a, const ConstIterator& b)
            {
                return a.cur_ == b.cur_;
            }

            friend bool operator!=(const ConstIterator& a, const ConstIterator& b)
            {
                return !(a == b);
            }
        };

    public:
        using iterator = Iterator;
        using const_iterator = ConstIterator;

        LinearOperatorList(std::vector<std::unique_ptr<LinearOperator<data_t>>> list);

        LinearOperatorList(const LinearOperator<data_t>& op);

        LinearOperatorList(const LinearOperator<data_t>& op1, const LinearOperator<data_t>& op2);

        LinearOperatorList(const LinearOperator<data_t>& op1, const LinearOperator<data_t>& op2,
                           const LinearOperator<data_t>& op3);

        LinearOperatorList(const LinearOperator<data_t>& op1, const LinearOperator<data_t>& op2,
                           const LinearOperator<data_t>& op3, const LinearOperator<data_t>& op4);

        template <class... Args>
        LinearOperatorList(const LinearOperator<data_t>& op1, const LinearOperator<data_t>& op2,
                           const LinearOperator<data_t>& op3, const LinearOperator<data_t>& op4,
                           const LinearOperator<data_t>& op5, Args&&... ops)
            : ops_(detail::makeLinearOpVector<data_t>(op1, op2, op3, op4, op5,
                                                      std::forward<Args>(ops)...))
        {
        }

        /// Copy constructor
        LinearOperatorList(const LinearOperatorList& other);

        /// Move constructor
        LinearOperatorList(LinearOperatorList&& other) noexcept;

        /// Copy assignment
        LinearOperatorList& operator=(const LinearOperatorList& other);

        /// Move assignment
        LinearOperatorList& operator=(LinearOperatorList&& other) noexcept;

        index_t getSize() const { return asSigned(ops_.size()); }

        const LinearOperator<data_t>& operator[](index_t idx) const
        {
            return *ops_[asUnsigned(idx)];
        }

        LinearOperator<data_t>& operator[](index_t idx) { return *ops_[asUnsigned(idx)]; }

        iterator begin() { return Iterator(ops_.begin()); }

        iterator end() { return Iterator(ops_.end()); }

        const_iterator begin() const { return cbegin(); }

        const_iterator end() const { return cend(); }

        const_iterator cbegin() const { return ConstIterator(ops_.begin()); }

        const_iterator cend() const { return ConstIterator(ops_.end()); }

    private:
        std::vector<std::unique_ptr<LinearOperator<data_t>>> ops_;
    };

} // namespace elsa
