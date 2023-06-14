#pragma once

#include "BlockDescriptor.h"
#include "Functional.h"
#include "IdenticalBlocksDescriptor.h"
#include "RandomBlocksDescriptor.h"
#include "TypeCasts.hpp"
#include <algorithm>
#include <memory>

namespace elsa
{
    namespace detail
    {
        /// Helper to create a vector of unique_ptrs from references with a clone method.
        template <class data_t, class... Ts>
        std::vector<std::unique_ptr<Functional<data_t>>> make_vector(Ts&&... ts)
        {
            std::vector<std::unique_ptr<Functional<data_t>>> v;
            v.reserve(sizeof...(ts));

            (v.emplace_back(std::forward<Ts>(ts).clone()), ...);
            return std::move(v);
        }

        /// Create a BlockDescriptor given a list of functionals. If all functionals have the same
        /// data descriptor, a `IdenticalBlocksDescriptor` returned, else a `RandomBlocksDescriptor`
        /// is returned.
        template <class data_t>
        std::unique_ptr<BlockDescriptor>
            determineDescriptor(const std::vector<std::unique_ptr<Functional<data_t>>>& fns);
    } // namespace detail

    /**
     * @brief Class representing a separable sum of functionals. Given a sequence
     * of \f$k\f$ functions \f$ ( f_i )_{i=1}^k \f$, where \f$f_{i}: X_{i} \rightarrow (-\infty,
     * \infty]\f$, the separable sum \f$F\f$ is defined as:
     *
     * \f[
     * F:X_{1}\times X_{2}\cdots\times X_{m} \rightarrow (-\infty, \infty] \\
     * F(x_{1}, x_{2}, \cdots, x_{k}) = \sum_{i=1}^k f_{i}(x_{i})
     * \f]
     *
     * The great benefit of the separable sum, is that its proximal is easily derived.
     *
     * @see CombinedProximal
     */
    template <class data_t>
    class SeparableSum final : public Functional<data_t>
    {
    public:
        /// Create a separable sum from a vector of unique_ptrs to functionals
        explicit SeparableSum(std::vector<std::unique_ptr<Functional<data_t>>> fns);

        /// Create a separable sum from a single functional
        explicit SeparableSum(const Functional<data_t>& fn);

        /// Create a separable sum from two functionals
        SeparableSum(const Functional<data_t>& fn1, const Functional<data_t>& fn2);

        /// Create a separable sum from three functionals
        SeparableSum(const Functional<data_t>& fn1, const Functional<data_t>& fn2,
                     const Functional<data_t>& fn3);

        /// Create a separable sum from variadic number of functionals
        template <class... Args>
        SeparableSum(const Functional<data_t>& fn1, const Functional<data_t>& fn2,
                     const Functional<data_t>& fn3, const Functional<data_t>& fn4, Args&&... fns)
            : SeparableSum<data_t>(
                detail::make_vector<data_t>(fn1, fn2, fn3, fn4, std::forward<Args>(fns)...))
        {
        }

    private:
        /// Evaluate functional. Requires `Rx` to be a blocked `DataContainer`
        /// (i.e. its descriptor is of type `BlockDescriptor`), the functions
        /// throws if not meat.
        data_t evaluateImpl(const DataContainer<data_t>& Rx) override;

        /// The derivative of the sum of functions, is the sum of the derivatives
        void getGradientImpl(const DataContainer<data_t>& Rx, DataContainer<data_t>& out) override;

        /// Not yet implemented
        LinearOperator<data_t> getHessianImpl(const DataContainer<data_t>& Rx) override;

        /// Polymorphic clone implementations
        SeparableSum<data_t>* cloneImpl() const override;

        /// Polymorphic equalty implementations
        bool isEqual(const Functional<data_t>& other) const override;

        std::vector<std::unique_ptr<Functional<data_t>>> fns_{};
    };
} // namespace elsa
