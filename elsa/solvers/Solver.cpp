#include "Solver.h"
#include "DataContainer.h"

namespace elsa
{
    template <typename data_t>
    DataContainer<data_t>& Solver<data_t>::solve(index_t iterations)
    {
        return solveImpl(iterations);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Solver<float>;
    template class Solver<double>;
    template class Solver<complex<float>>;
    template class Solver<complex<double>>;

} // namespace elsa
