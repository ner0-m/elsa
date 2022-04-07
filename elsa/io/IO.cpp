#include "IO.h"

#include "EDFHandler.h"
#include "Error.h"
#include "MHDHandler.h"
#include "PGMHandler.h"
#include <iostream>

namespace elsa::io
{
    [[nodiscard]] std::optional<std::string_view> get_extension(std::string_view str)
    {
        const auto delimiter = '.';
        const auto last_part = str.find_last_of(delimiter);

        // String doesn't contain delimiter
        if (last_part == std::string_view::npos) {
            return std::nullopt;
        }

        return str.substr(last_part);
    }

    template <typename data_t>
    [[nodiscard]] DataContainer<data_t> read(std::string_view filename)
    {
        const auto opt = get_extension(filename);

        // No dot present in filename, so throw right away
        if (!opt.has_value()) {
            throw Error("No extension found in filename (\"{}\")", filename);
        }

        const auto extension = *opt;

        if (extension == ".edf") {
            return EDF::read<data_t>(std::string{filename});
        }

        throw Error("Can not read with unsupported file extension \"{}\"", extension);
    }

    template <typename data_t>
    void write(DataContainer<data_t> x, std::string_view filename)
    {
        const auto opt = get_extension(filename);

        // No dot present in filename, so throw right away
        if (!opt.has_value()) {
            throw Error("No extension found in filename (\"{}\")", filename);
        }

        const auto extension = *opt;

        if (extension == ".edf") {
            return EDF::write<data_t>(x, std::string{filename});
        } else if (extension == ".pgm") {
            return PGM::write<data_t>(x, std::string{filename});
        }

        throw Error("Can not write with unsupported file extension \"{}\"", extension);
    }

    template DataContainer<float> read<float>(std::string_view);
    template DataContainer<double> read<double>(std::string_view);

    template void write<float>(DataContainer<float>, std::string_view);
    template void write<double>(DataContainer<double>, std::string_view);
} // namespace elsa::io
