#pragma once

#include "DataContainer.h"

#include <optional>
#include <string_view>

namespace elsa::io
{
    /**
     * @brief Read from filename and create a DataContainer. Filename is expected to have a valid
     * (i.e. supported) extension, else this function will throw.
     *
     * @param filename filename to read data from
     */
    template <typename data_t>
    DataContainer<data_t> read(std::string_view filename);

    /**
     * @brief Write DataContainer to a file with given filename. Filename is expected to have a
     * valid (i.e. supported) extension, else this function will throw.
     *
     * @param x DataContainer to write to file
     * @param filename filename to write data to
     */
    template <typename data_t>
    void write(DataContainer<data_t> x, std::string_view filename);
} // namespace elsa::io
