#include "testHelpers.h"

namespace elsa
{
    template <typename data_t>
    bool isApprox(const DataContainer<data_t>& x, const DataContainer<data_t>& y, real_t prec)
    {
        // check if size is the same, but do not throw an exception
        assert(x.getSize() == y.getSize());

        DataContainer<data_t> z = x;
        z -= y;

        data_t lhs = std::sqrt(z.squaredL2Norm());
        data_t rhs = prec * std::sqrt(std::min(x.squaredL2Norm(), y.squaredL2Norm()));
        return lhs <= rhs;
    }

    // ------------------------------------------
    // explicit template instantiation
    template bool isApprox(const DataContainer<float>& x, const DataContainer<float>& y,
                           real_t prec);
    template bool isApprox(const DataContainer<double>& x, const DataContainer<double>& y,
                           real_t prec);

} // namespace elsa
