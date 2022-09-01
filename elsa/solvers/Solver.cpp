#include "Solver.h"
#include "DataContainer.h"

namespace elsa
{
    // ------------------------------------------
    // explicit template instantiation
    template class Solver<float>;
    template class Solver<double>;
    template class Solver<complex<float>>;
    template class Solver<complex<double>>;

} // namespace elsa
