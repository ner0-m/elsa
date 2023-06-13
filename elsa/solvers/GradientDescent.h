#pragma once

#include <optional>

#include "Solver.h"
#include "Functional.h"
#include "MaybeUninitialized.hpp"

namespace elsa
{

    /**
     * @brief Class representing a simple gradient descent solver with a fixed, given step size.
     *
     * This class implements a simple gradient descent iterative solver with a fixed, given step
     * size. No particular stopping rule is currently implemented (only a fixed number of
     * iterations, default to 100).
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * @see \verbatim embed:rst
     For a basic introduction and problem statement of first-order methods, see
     :ref:`here <elsa-first-order-methods-doc>` \endverbatim
     *
     * @author
     * - Tobias Lasser - initial code
     */
    template <typename data_t = real_t>
    class GradientDescent : public Solver<data_t>
    {
    public:
        /// Scalar alias
        using Scalar = typename Solver<data_t>::Scalar;

        /**
         * @brief Constructor for gradient descent, accepting a problem and a fixed step size
         *
         * @param[in] problem the problem that is supposed to be solved
         * @param[in] stepSize the fixed step size to be used while solving
         */
        GradientDescent(const Functional<data_t>& problem, data_t stepSize);

        /**
         * @brief Constructor for gradient descent, accepting a problem. The step size will be
         * computed as \f$ 1 \over L \f$ with \f$ L \f$ being the Lipschitz constant of the
         * function.
         *
         * @param[in] problem the problem that is supposed to be solved
         */
        explicit GradientDescent(const Functional<data_t>& problem);

        /// make copy constructor deletion explicit
        GradientDescent(const GradientDescent<data_t>&) = delete;

        /// default destructor
        ~GradientDescent() override = default;

        /**
         * @brief Solve the optimization problem, i.e. apply iterations number of iterations of
         * gradient descent
         *
         * @param[in] iterations number of iterations to execute
         *
         * @returns the approximated solution
         */
        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

    private:
        /// the differentiable optimizaion problem
        std::unique_ptr<Functional<data_t>> _problem;

        /// the step size
        MaybeUninitialized<data_t> _stepSize;

        /// implement the polymorphic clone operation
        GradientDescent<data_t>* cloneImpl() const override;

        /// implement the polymorphic comparison operation
        bool isEqual(const Solver<data_t>& other) const override;
    };
} // namespace elsa
