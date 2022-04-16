#pragma once

#include "elsaDefines.h"
#include "Cloneable.h"
#include "DataDescriptor.h"
#include "DataContainer.h"

#include <memory>

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

        std::optional<data_t> getScalar();

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

} // namespace elsa
