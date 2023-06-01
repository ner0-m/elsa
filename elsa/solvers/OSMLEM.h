#pragma once

#include <memory>
#include <vector>

#include "DataContainer.h"
#include "LinearOperator.h"
#include "Solver.h"
#include "elsaDefines.h"

namespace elsa
{

    /**
     * @brief Ordered Subsets Maximum Likelihood Expectation Maximation algorithm
     *
     * The algorithm maximizes the likhelihood that:
     * \f[
     * \max_x L(x | g)
     * \f]
     * where, \f$L(x | g)\f$ is the likelihood of $x\$f given \$fg\f$. \$fg\f$
     * being the measured data.
     *
     * The update step for the MLEM algorithm is given as:
     * \f[
     * x_{k+1} = \frac{x_k}{A^\ast \bm{1}} A^\ast (\frac{g}{Ax_k}
     * \f]
     *
     * For the subset version, the forward model is split into \f$k\f$ subsets,
     * and the update step is performed for each subset individually. Thou, no
     * convergence is guarantted, in many practicall applications the algorithm
     * performs very well, and converges faster then the classical MLEM algorithm.
     *
     * @see MLEM
     */
    template <class data_t>
    class OSMLEM : public Solver<data_t>
    {
    public:
        OSMLEM(LinearOperatorList<data_t> ops, std::vector<DataContainer<data_t>> data,
               SelfType_t<data_t> eps = 1e-8)
            : Solver<data_t>(), ops_(std::move(ops)), data_(std::move(data)), eps_(eps)
        {
        }

        ~OSMLEM() = default;

        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

    protected:
        /// implement the polymorphic clone operation
        auto cloneImpl() const -> OSMLEM<data_t>* override;

        /// implement the polymorphic comparison operation
        auto isEqual(const Solver<data_t>& other) const -> bool override;

    private:
        LinearOperatorList<data_t> ops_{};

        std::vector<DataContainer<data_t>> data_{};

        data_t eps_ = 1e-8;
    };

    namespace detail
    {
        /// Implementation of a single iteration of one ordered substep in MLEM.
        template <class data_t>
        void mlemStep(const LinearOperator<data_t>& op, const DataContainer<data_t>& data,
                      DataContainer<data_t>& x, DataContainer<data_t>& range,
                      DataContainer<data_t>& domain, const DataContainer<data_t>& sensitivity,
                      SelfType_t<data_t> eps);
    } // namespace detail

} // namespace elsa
