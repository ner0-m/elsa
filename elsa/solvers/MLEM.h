#pragma once

#include <memory>

#include "DataContainer.h"
#include "LinearOperator.h"
#include "Solver.h"
#include "elsaDefines.h"

namespace elsa
{
    /**
     * @brief Maximum Likelihood Expectation Maximation algorithm
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
     * @see OSMLEM
     */
    template <class data_t>
    class MLEM : public Solver<data_t>
    {
    public:
        MLEM(const LinearOperator<data_t>& op, const DataContainer<data_t>& data,
             SelfType_t<data_t> eps = 1e-8)
            : Solver<data_t>(), op_(op.clone()), data_(data), eps_(eps)
        {
        }

        ~MLEM() = default;

        DataContainer<data_t>
            solve(index_t iterations,
                  std::optional<DataContainer<data_t>> x0 = std::nullopt) override;

    protected:
        /// implement the polymorphic clone operation
        auto cloneImpl() const -> MLEM<data_t>* override { return new MLEM(*op_, data_, eps_); }

        /// implement the polymorphic comparison operation
        auto isEqual(const Solver<data_t>& other) const -> bool override
        {
            auto otherMLEM = downcast_safe<MLEM>(&other);
            if (!otherMLEM)
                return false;

            if (*op_ != *otherMLEM->op_)
                return false;

            if (data_ != otherMLEM->data_)
                return false;

            if (eps_ != otherMLEM->eps_)
                return false;

            return true;
        }

    private:
        std::unique_ptr<LinearOperator<data_t>> op_;

        DataContainer<data_t> data_;

        data_t eps_;
    };

} // namespace elsa
