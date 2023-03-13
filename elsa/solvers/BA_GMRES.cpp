#include "BA_GMRES.h"
#include "GMRES_common.h"
#include "TypeCasts.hpp"
#include "spdlog/stopwatch.h"

namespace elsa
{
    template <typename data_t>
    BA_GMRES<data_t>::BA_GMRES(const LinearOperator<data_t>& projector,
                               const DataContainer<data_t>& sinogram, data_t epsilon)
        : Solver<data_t>(),
          _A{projector.clone()},
          _B{adjoint(projector).clone()},
          _b{sinogram},
          _epsilon{epsilon}
    {
    }

    template <typename data_t>
    BA_GMRES<data_t>::BA_GMRES(const LinearOperator<data_t>& projector,
                               const LinearOperator<data_t>& backprojector,
                               const DataContainer<data_t>& sinogram, data_t epsilon)
        : Solver<data_t>(),
          _A{projector.clone()},
          _B{backprojector.clone()},
          _b{sinogram},
          _epsilon{epsilon}
    {
    }

    template <typename data_t>
    DataContainer<data_t> BA_GMRES<data_t>::solveAndRestart(index_t iterations, index_t restarts,
                                                            std::optional<DataContainer<data_t>> x0)
    {
        auto x = DataContainer<data_t>(_A->getDomainDescriptor());
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
    DataContainer<data_t> BA_GMRES<data_t>::solve(index_t iterations,
                                                  std::optional<DataContainer<data_t>> x0)
    {
        auto calculate_r0 = [](const LinearOperator<data_t>& A, const LinearOperator<data_t>& B,
                               const DataContainer<data_t>& b,
                               const DataContainer<data_t>& x) -> DataContainer<data_t> {
            auto Bb = B.apply(b);
            auto Ax = A.apply(x);
            auto BAx = B.apply(Ax);

            auto r0 = Bb - BAx;
            return r0;
        };

        auto calculate_q = [](const LinearOperator<data_t>& A, const LinearOperator<data_t>& B,
                              const DataContainer<data_t>& w_k) -> DataContainer<data_t> {
            auto Aw_k = A.apply(w_k);
            auto q = B.apply(Aw_k);
            return q;
        };

        auto calculate_x = [](const LinearOperator<data_t>& B, const DataContainer<data_t>& x,
                              const DataContainer<data_t>& wy) -> DataContainer<data_t> {
            auto x_k = x + wy;
            return x_k;
        };

        // explicitly casting lambda so it can be resolved by the compiler for detail::gmres
        std::function<DataContainer<data_t>(
            const LinearOperator<data_t>&, const LinearOperator<data_t>&,
            const DataContainer<data_t>&, const DataContainer<data_t>&)>
            calc_r0 = calculate_r0;

        std::function<DataContainer<data_t>(const LinearOperator<data_t>&,
                                            const LinearOperator<data_t>&,
                                            const DataContainer<data_t>&)>
            calc_q = calculate_q;

        std::function<DataContainer<data_t>(const LinearOperator<data_t>&,
                                            const DataContainer<data_t>&,
                                            const DataContainer<data_t>&)>
            calc_x = calculate_x;

        auto x = DataContainer<data_t>(_A->getDomainDescriptor());
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        return detail::gmres("BA_GMRES", _A, _B, _b, _epsilon, x, iterations, calc_r0, calc_q,
                             calc_x);
    }

    template <typename data_t>
    BA_GMRES<data_t>* BA_GMRES<data_t>::cloneImpl() const
    {
        return new BA_GMRES(*_A, *_B, _b, _epsilon);
    }

    template <typename data_t>
    bool BA_GMRES<data_t>::isEqual(const Solver<data_t>& other) const
    {
        // This is basically stolen from CG

        auto otherGMRES = downcast_safe<BA_GMRES>(&other);

        if (!otherGMRES)
            return false;

        if (_epsilon != otherGMRES->_epsilon)
            return false;

        if ((_A && !otherGMRES->_A) || (!_A && otherGMRES->_A))
            return false;

        if (_A && otherGMRES->_A)
            if (*_A != *otherGMRES->_A)
                return false;

        if ((_B && !otherGMRES->_B) || (!_B && otherGMRES->_B))
            return false;

        if (_B && otherGMRES->_B)
            if (*_B != *otherGMRES->_B)
                return false;

        if (_b != otherGMRES->_b)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class BA_GMRES<float>;
    template class BA_GMRES<double>;

} // namespace elsa