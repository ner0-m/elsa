#include "PGMHandler.h"

#include <ostream>
#include <iostream>
#include <exception>

#include "Error.h"
#include "spdlog/fmt/bundled/core.h"
#include "spdlog/fmt/bundled/ostream.h"

namespace elsa
{
    template <typename data_t>
    void PGM::write(const DataContainer<data_t>& data, const std::string& filename)
    {
        std::ofstream ofs(filename, std::ios_base::out);

        PGM::write(data, ofs);
    }

    template <typename data_t>
    void PGM::write(const DataContainer<data_t>& data, std::ostream& stream)
    {
        // If `data_t` is float or double use that to cast values to, if it's
        // `index_t` then use real_t for multiplications
        using CastType = std::conditional_t<std::is_floating_point_v<data_t>, data_t, real_t>;

        const auto dim = data.getDataDescriptor().getNumberOfDimensions();
        const auto shape = data.getDataDescriptor().getNumberOfCoefficientsPerDimension();

        // print 3D containers with last dim 1 (this way we can slice it)
        if (dim != 2 && !(dim == 3 && shape[dim - 1] == 1)) {
            throw InvalidArgumentError("PGM:: Can only handle 2D data");
        }

        const auto dims = data.getDataDescriptor().getNumberOfCoefficientsPerDimension();

        const auto maxValue = data.maxElement();
        const auto minValue = data.minElement();

        // Scale all values from DataContainer to a range from [0, 255]
        const auto scaleFactor =
            255.f / static_cast<CastType>(std::ceil(maxValue + std::abs(minValue)));

        // P2: Magic number specifying grey scale, then the two dimensions in the next line
        // Then the maximum value of the image in our case always 255
        stream << "P2\n" << dims[0] << " " << dims[1] << "\n" << 255 << "\n";

        // Write data, ugly casts to silence warnings
        for (int i = 0; i < data.getSize(); ++i) {
            // Scale everything to the range [0, 1]
            const auto normalized = (data[i] - minValue) / (maxValue - minValue);

            // Then scale it up to the range [0, 255]
            stream << static_cast<int>(normalized * 255.f) << " ";
        }
    }

    template void PGM::write(const DataContainer<float>&, const std::string&);
    template void PGM::write(const DataContainer<double>&, const std::string&);
    template void PGM::write(const DataContainer<index_t>&, const std::string&);

    template void PGM::write(const DataContainer<float>&, std::ostream&);
    template void PGM::write(const DataContainer<double>&, std::ostream&);
    template void PGM::write(const DataContainer<index_t>&, std::ostream&);
} // namespace elsa
