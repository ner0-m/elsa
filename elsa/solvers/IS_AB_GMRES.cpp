#include "IS_AB_GMRES.h"
#include "GMRES_common.h"
#include "TypeCasts.hpp"
#include "spdlog/stopwatch.h"

namespace elsa
{
    template <typename data_t>
    IS_AB_GMRES<data_t>::IS_AB_GMRES(const LinearOperator<data_t>& projector,
                                     const DataContainer<data_t>& sinogram, data_t epsilon)
        : IterativeSolver<data_t>{projector, sinogram},
          _B{adjoint(projector).clone()},
          _epsilon{epsilon}
    {
    }

    template <typename data_t>
    IS_AB_GMRES<data_t>::IS_AB_GMRES(const LinearOperator<data_t>& projector,
                                     const LinearOperator<data_t>& backprojector,
                                     const DataContainer<data_t>& sinogram, data_t epsilon)
        : IterativeSolver<data_t>{projector, sinogram}, _B{backprojector.clone()}, _epsilon{epsilon}
    {
    }

    template <typename data_t>
    DataContainer<data_t>
        IS_AB_GMRES<data_t>::solveAndRestart(index_t iterations, index_t restarts,
                                             std::optional<DataContainer<data_t>> x0)
    {
        auto x = DataContainer<data_t>(IS::A->getDomainDescriptor());
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        for (index_t k = 0; k < restarts; k++) {
            x = solve(iterations, x);
        }

        return x;
    }

    template <typename data_t>
    DataContainer<data_t> IS_AB_GMRES<data_t>::solve(index_t iterations,
                                                     std::optional<DataContainer<data_t>> x0)
    {
        detail::CalcRFn<data_t> calc_r0 =
            [](const LinearOperator<data_t>& A, const LinearOperator<data_t>& B,
               const DataContainer<data_t>& b,
               const DataContainer<data_t>& x) -> DataContainer<data_t> {
            auto Ax = A.apply(x);
            auto r0 = b - Ax;
            return r0;
        };

        detail::CalcQFn<data_t> calc_q =
            [](const LinearOperator<data_t>& A, const LinearOperator<data_t>& B,
               const DataContainer<data_t>& w_k) -> DataContainer<data_t> {
            auto Bw_k = B.apply(w_k);
            auto q = A.apply(Bw_k);
            return q;
        };

        detail::CalcXFn<data_t> calc_x =
            [](const LinearOperator<data_t>& B, const DataContainer<data_t>& x,
               const DataContainer<data_t>& wy) -> DataContainer<data_t> {
            auto x_k = x + B.apply(wy);
            return x_k;
        };

        auto x = DataContainer<data_t>(IS::A->getDomainDescriptor());
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        return detail::gmres("AB_GMRES", IS::A, _B, IS::b, _epsilon, x, iterations, calc_r0, calc_q,
                             calc_x);
    }

    template <typename data_t>
    IS_AB_GMRES<data_t>* IS_AB_GMRES<data_t>::cloneImpl() const
    {
        return new IS_AB_GMRES(*IS::A, *_B, IS::b, _epsilon);
    }

    template <typename data_t>
    bool IS_AB_GMRES<data_t>::isEqual(const IterativeSolver<data_t>& other) const
    {
        // This is basically stolen from CG

        auto otherGMRES = downcast_safe<IS_AB_GMRES>(&other);

        if (!otherGMRES)
            return false;

        if (_epsilon != otherGMRES->_epsilon)
            return false;

        if ((IS::A && !otherGMRES->A) || (!IS::A && otherGMRES->A))
            return false;

        if (IS::A && otherGMRES->A)
            if (*IS::A != *otherGMRES->A)
                return false;

        if ((_B && !otherGMRES->_B) || (!_B && otherGMRES->_B))
            return false;

        if (_B && otherGMRES->_B)
            if (*_B != *otherGMRES->_B)
                return false;

        if (IS::b != otherGMRES->b)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class IS_AB_GMRES<float>;
    template class IS_AB_GMRES<double>;

} // namespace elsa