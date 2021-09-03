#include "PSNR.h"
#include "MSE.h"

namespace elsa
{
    template <typename data_t>
    data_t PSNR<data_t>::calculate(DataContainer<data_t> x, DataContainer<data_t> y,
                                   unsigned int dataRange)
    {
        if (x.getDataDescriptor() != y.getDataDescriptor()) {
            throw LogicError(std::string("PSNR: shapes of both signals should match"));
        }

        data_t err = MSE<data_t>::calculate(x, y);
        return 10 * std::log10((std::pow(dataRange, 2) / err));
    }

    template <typename data_t>
    data_t PSNR<data_t>::calculate(DataContainer<data_t> x, DataContainer<data_t> y)
    {
        if (x.getDataDescriptor() != y.getDataDescriptor()) {
            throw LogicError(std::string("PSNR: shapes of both signals should match"));
        }

        unsigned int dataRange = 255; // TODO calculate based on data_t

        return calculate(x, y, dataRange);
    }
} // namespace elsa