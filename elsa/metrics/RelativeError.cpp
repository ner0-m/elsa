#include "RelativeError.h"

namespace elsa
{
    template <typename data_t>
    long double RelativeError<data_t>::calculate(DataContainer<data_t> leftSignal,
                                                 DataContainer<data_t> rightSignal)
    {
        DataContainer<data_t> diff = leftSignal - rightSignal;
        return diff.l2Norm() / rightSignal.l2Norm();
    }
} // namespace elsa