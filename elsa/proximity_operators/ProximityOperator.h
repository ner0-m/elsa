#pragma once

#include "elsaDefines.h"
#include "Cloneable.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "StrongTypes.h"

namespace elsa
{
    /**
     * @brief Base class representing a proximity operator prox.
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the values of the operator, defaulting to real_t
     *
     * This class represents a proximity operator prox, expressed through its apply methods,
     * which implement the proximity operator of f with penalty r i.e.
     * @f$ prox_{f,\rho}(v) = argmin_{x}(f(x) + (\rho/2)·\| x - v \|^2_2). @f$
     *
     * Concrete implementations of proximity operators will derive from this class and override the
     * applyImpl method.
     *
     * References:
     * https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
     */
    template <typename data_t = real_t>
    class ProximityOperator : public Cloneable<ProximityOperator<data_t>>
    {
    public:
        /// delete no-args constructor to prevent creation of an object without a DataDescriptor
        ProximityOperator() = delete;

        /**
         * @brief Override to construct an actual proximity operator for one of the derived classes
         * from the given DataDescriptor descriptor
         *
         * @param[in] descriptor DataDescriptor describing the operator values
         */
        ProximityOperator(const DataDescriptor& descriptor);

        /// delete copy construction
        ProximityOperator(const ProximityOperator<data_t>&) = delete;

        /// delete implicitly declared copy assignment to prevent copy assignment
        auto operator=(const ProximityOperator&) -> ProximityOperator& = delete;

        /// default destructor
        ~ProximityOperator() override = default;

        /// return the DataDescriptor
        auto getRangeDescriptor() const -> const DataDescriptor&;

        /**
         * @brief apply the proximity operator to an element in the operator's domain
         *
         * @param[in] v input DataContainer
         * @param[in] t input Threshold
         *
         * @returns prox DataContainer containing the application of the proximity operator to
         * data v, i.e. in the range of the operator
         *
         * Please note: this method uses apply(v, t, prox(v)) to perform the actual operation.
         */
        auto apply(const DataContainer<data_t>& v, geometry::Threshold<data_t> t) const
            -> DataContainer<data_t>;

        /**
         * @brief apply the proximity operator to an element in the operator's domain
         *
         * @param[in] v input DataContainer
         * @param[in] thresholds input vector<Threshold>
         *
         * @returns prox DataContainer containing the application of the proximity operator to
         * data v, i.e. in the range of the operator
         *
         * Please note: this method uses apply(v, thresholds, prox(v)) to perform the actual
         * operation.
         */
        auto apply(const DataContainer<data_t>& v,
                   std::vector<geometry::Threshold<data_t>> thresholds) const
            -> DataContainer<data_t>;

        /**
         * @brief apply the proximity operator to an element in the operator's domain
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
        /**
         * @brief apply the proximity operator to an element in the operator's domain
         *
         * @param[in] v input DataContainer
         * @param[in] thresholds input vector<Threshold>
         * @param[out] prox output DataContainer
         *
         * Please note: this method calls the method applyImpl that has to be overridden in derived
         * classes. (Why is this method not virtual itself? Because you cannot have a non-virtual
         * function overloading a virtual one [apply with one vs. two arguments]).
         */
        void apply(const DataContainer<data_t>& v,
                   std::vector<geometry::Threshold<data_t>> thresholds,
                   DataContainer<data_t>& prox) const;

    protected:
        std::unique_ptr<DataDescriptor> _rangeDescriptor;

        /// the apply method that has to be overridden in derived classes
        virtual void applyImpl(const DataContainer<data_t>& v, geometry::Threshold<data_t> t,
                               DataContainer<data_t>& prox) const = 0;

        /// the apply method that has to be overridden in derived classes
        virtual void applyImpl(const DataContainer<data_t>& v,
                               std::vector<geometry::Threshold<data_t>> thresholds,
                               DataContainer<data_t>& prox) const = 0;

        /// overridden comparison method based on the DataDescriptor
        auto isEqual(const ProximityOperator<data_t>& other) const -> bool override;
    };
} // namespace elsa
