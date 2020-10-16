#include "SplittingProjector.h"
#include "SiddonsMethodCUDA.h"
#include "JosephsMethodCUDA.h"

namespace elsa
{
    // template class SplittingProjector<SiddonsMethodCUDA<float>>;
    // template class SplittingProjector<SiddonsMethodCUDA<double>>;
    template class SplittingProjector<JosephsMethodCUDA<float>>;
    template class SplittingProjector<JosephsMethodCUDA<double>>;
} // namespace elsa