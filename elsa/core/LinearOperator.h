#pragma once

#include "elsaDefines.h"
#include "Cloneable.h"
#include "DataDescriptor.h"
#include "DataContainer.h"

#include <memory>
#include <type_traits>

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
     * apply/applyAdjoint methods.
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
         */
        virtual void apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const = 0;

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
         */
        virtual void applyAdjoint(const DataContainer<data_t>& y,
                                  DataContainer<data_t>& Aty) const = 0;

    protected:
        /// the data descriptor of the domain of the operator
        std::unique_ptr<DataDescriptor> _domainDescriptor;

        /// the data descriptor of the range of the operator
        std::unique_ptr<DataDescriptor> _rangeDescriptor;

        /// implement the polymorphic comparison operation
        bool isEqual(const LinearOperator<data_t>& other) const override;
    };

    template <typename data_t = real_t>
    class AdjointLinearOperator : public LinearOperator<data_t>
    {
    public:
        AdjointLinearOperator(const LinearOperator<data_t>& op);

        /// the apply method that has to be overridden in derived classes
        void apply(const DataContainer<data_t>& y, DataContainer<data_t>& Aty) const override;

        /// the applyAdjoint  method that has to be overridden in derived classes
        void applyAdjoint(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        // Pull in apply and applyAdjoint with single argument from base class
        using LinearOperator<data_t>::apply;
        using LinearOperator<data_t>::applyAdjoint;

    protected:
        /// implement the polymorphic clone operation
        AdjointLinearOperator<data_t>* cloneImpl() const override;

        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        std::unique_ptr<LinearOperator<data_t>> op_;
    };

    template <typename data_t = real_t>
    class ScalarMulLinearOperator : public LinearOperator<data_t>
    {
    public:
        ScalarMulLinearOperator(data_t scalar, const LinearOperator<data_t>& op);

        /// the apply method that has to be overridden in derived classes
        void apply(const DataContainer<data_t>& y, DataContainer<data_t>& Aty) const override;

        /// the applyAdjoint  method that has to be overridden in derived classes
        void applyAdjoint(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        // Pull in apply and applyAdjoint with single argument from base class
        using LinearOperator<data_t>::apply;
        using LinearOperator<data_t>::applyAdjoint;

    protected:
        /// implement the polymorphic clone operation
        ScalarMulLinearOperator<data_t>* cloneImpl() const override;

        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        data_t scalar_;
        std::unique_ptr<LinearOperator<data_t>> op_;
    };

    template <typename data_t = real_t>
    class CompositeAddLinearOperator : public LinearOperator<data_t>
    {
    public:
        CompositeAddLinearOperator(const LinearOperator<data_t>& lhs,
                                   const LinearOperator<data_t>& rhs);

        /// the apply method that has to be overridden in derived classes
        void apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// the applyAdjoint  method that has to be overridden in derived classes
        void applyAdjoint(const DataContainer<data_t>& y,
                          DataContainer<data_t>& Aty) const override;

        // Pull in apply and applyAdjoint with single argument from base class
        using LinearOperator<data_t>::apply;
        using LinearOperator<data_t>::applyAdjoint;

    protected:
        /// implement the polymorphic clone operation
        CompositeAddLinearOperator<data_t>* cloneImpl() const override;

        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        /// pointers to nodes in the evaluation tree
        std::unique_ptr<LinearOperator<data_t>> lhs_{}, rhs_{};
    };

    template <typename data_t = real_t>
    class CompositeMulLinearOperator : public LinearOperator<data_t>
    {
    public:
        CompositeMulLinearOperator(const LinearOperator<data_t>& lhs,
                                   const LinearOperator<data_t>& rhs);

        /// the apply method that has to be overridden in derived classes
        void apply(const DataContainer<data_t>& x, DataContainer<data_t>& Ax) const override;

        /// the applyAdjoint  method that has to be overridden in derived classes
        void applyAdjoint(const DataContainer<data_t>& y,
                          DataContainer<data_t>& Aty) const override;

        // Pull in apply and applyAdjoint with single argument from base class
        using LinearOperator<data_t>::apply;
        using LinearOperator<data_t>::applyAdjoint;

    protected:
        /// implement the polymorphic clone operation
        CompositeMulLinearOperator<data_t>* cloneImpl() const override;

        bool isEqual(const LinearOperator<data_t>& other) const override;

    private:
        /// pointers to nodes in the evaluation tree
        std::unique_ptr<LinearOperator<data_t>> lhs_{}, rhs_{};
    };

    /// operator+ to support composition of LinearOperators (and its derivatives)
    template <typename data_t>
    CompositeAddLinearOperator<data_t> operator+(const LinearOperator<data_t>& lhs,
                                                 const LinearOperator<data_t>& rhs)
    {
        return CompositeAddLinearOperator(lhs, rhs);
    }

    /// operator* to support composition of LinearOperators (and its derivatives)
    template <typename data_t>
    CompositeMulLinearOperator<data_t> operator*(const LinearOperator<data_t>& lhs,
                                                 const LinearOperator<data_t>& rhs)
    {
        return CompositeMulLinearOperator(lhs, rhs);
    }

    /// operator* to support composition of a scalar and a LinearOperator, enable if magic, to make
    /// it work without casting to `data_t`
    template <typename data_t, typename Scalar = data_t,
              typename = std::enable_if_t<isArithmetic<Scalar>>>
    ScalarMulLinearOperator<data_t> operator*(Scalar scalar, const LinearOperator<data_t>& op)
    {
        return ScalarMulLinearOperator(static_cast<data_t>(scalar), op);
    }

    /// function to return the adjoint of a LinearOperator (and its derivatives)
    template <typename data_t>
    AdjointLinearOperator<data_t> adjoint(const LinearOperator<data_t>& op)
    {
        return AdjointLinearOperator(op);
    }
} // namespace elsa
