#pragma once

#include "elsaDefines.h"
#include "Cloneable.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "StrongTypes.h"

namespace elsa
{
    /// Customization point for ProximalOperators. For an object to be type erased by the
    /// ProximalOperator interface, you either need to provide a member function `apply` for the
    /// type, with the given parameters. Or you can overload this function.
    template <class T, class data_t>
    void applyProximal(const T& proximal, const DataContainer<data_t>& v,
                       geometry::Threshold<data_t> tau, DataContainer<data_t>& out)
    {
        proximal.apply(v, tau, out);
    }

    /**
     * @brief Base class representing a proximal operator prox.
     *
     * This class represents a proximal operator prox, expressed through its apply methods,
     * which implement the proximal operator of f with penalty r i.e.
     * @f$ prox_{f,\rho}(v) = argmin_{x}(f(x) + (\rho/2)·\| x - v \|^2_2). @f$
     *
     * The class implements type erasure. Classes, that bind to this wrapper should overload
     * either the `applyProximal`, or implement the `apply`  member function, with the signature
     * given in this class. Base types also are assumed to be Semi-Regular.
     *
     * References:
     * https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
     *
     * @tparam data_t data type for the values of the operator, defaulting to real_t
     *
     * @author
     * - Andi Braimllari - initial code
     * - David Frank - Type Erasure
     */
    template <typename data_t = real_t>
    class ProximalOperator
    {
    private:
        /// Concept for proximal operators, they should be cloneable, and have an apply method
        struct ProxConcept {
            virtual ~ProxConcept() = default;
            virtual std::unique_ptr<ProxConcept> clone() const = 0;
            virtual void apply(const DataContainer<data_t>& v, geometry::Threshold<data_t> t,
                               DataContainer<data_t>& out) const = 0;
        };

        /// Bridge wrapper for concrete types
        template <class T>
        struct ProxModel : public ProxConcept {
            ProxModel(T self) : self_(std::move(self)) {}

            /// Just clone the type (assumes regularity, i.e. copy constructible)
            std::unique_ptr<ProxConcept> clone() const override
            {
                return std::make_unique<ProxModel<T>>(self_);
            }

            /// Apply proximal by calling `apply_proximal`, this enables flexible extension, without
            /// classes as well. The default implementation of `apply_proximal`,
            /// just calls the member function
            void apply(const DataContainer<data_t>& v, geometry::Threshold<data_t> t,
                       DataContainer<data_t>& out) const override
            {
                applyProximal(self_, v, t, out);
            }

        private:
            T self_;
        };

    public:
        /// defaulted default constructor for base-class (will point to nothing)
        ProximalOperator() = default;

        /// Type erasure constructor, taking everything that kan bind to the above provided
        /// interface
        template <typename T>
        ProximalOperator(T proxOp) : ptr_(std::make_unique<ProxModel<T>>(std::move(proxOp)))
        {
        }

        template <typename T>
        ProximalOperator& operator=(T proxOp)
        {
            ptr_ = std::make_unique<ProxModel<T>>(std::move(proxOp));
            return *this;
        }

        /// Copy constructor
        ProximalOperator(const ProximalOperator& other);

        /// Default move constructor
        ProximalOperator(ProximalOperator&& other) noexcept = default;

        /// Copy assignment
        ProximalOperator& operator=(const ProximalOperator& other);

        /// Default move assignment
        ProximalOperator& operator=(ProximalOperator&& other) noexcept = default;

        /// default destructor
        ~ProximalOperator() = default;

        /**
         * @brief apply the proximal operator to an element in the operator's domain
         *
         * @param[in] v input DataContainer
         * @param[in] t input Threshold
         *
         * @returns prox DataContainer containing the application of the proximal operator to
         * data v, i.e. in the range of the operator
         *
         * Please note: this method uses apply(v, t, prox(v)) to perform the actual operation.
         */
        auto apply(const DataContainer<data_t>& v, geometry::Threshold<data_t> t) const
            -> DataContainer<data_t>;

        /**
         * @brief apply the proximal operator to an element in the operator's domain
         *
         * @param[in] v input DataContainer
         * @param[in] t input Threshold
         * @param[out] prox output DataContainer
         *
         * Please note: this method calls the method applyImpl that has to be overridden in derived
         * classes. (Why is this method not virtual itself? Because you cannot have a non-virtual
         * function overloading a virtual one [apply with one vs. two arguments]).
         */
        void apply(const DataContainer<data_t>& v, geometry::Threshold<data_t> t,
                   DataContainer<data_t>& prox) const;

    private:
        std::unique_ptr<ProxConcept> ptr_ = {};
    };
} // namespace elsa
