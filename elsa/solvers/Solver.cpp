#include "Solver.h"

namespace elsa
{
    template <typename data_t>
    Solver<data_t>::Solver(const Problem<data_t>& problem) : _problem{problem.clone()}
    {
    }

    template <typename data_t>
    const DataContainer<data_t>& Solver<data_t>::getCurrentSolution() const
    {
        return _problem->getCurrentSolution();
    }

    template <typename data_t>
    DataContainer<data_t>& Solver<data_t>::getCurrentSolution()
    {
        return _problem->getCurrentSolution();
    }

    template <typename data_t>
    DataContainer<data_t>& Solver<data_t>::solve(index_t iterations)
    {
        return solveImpl(iterations);
    }

    template <typename data_t>
    bool Solver<data_t>::isEqual(const Solver<data_t>& other) const
    {
        return static_cast<bool>(*_problem == *other._problem);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Solver<float>;
    template class Solver<double>;
    template class Solver<std::complex<float>>;
    template class Solver<std::complex<double>>;

} // namespace elsa
