#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "ioUtils.h"

namespace elsa
{
    /**
     * @brief Class to handle writing PGM image files from DataContainers.
     *
     * The "portable gray map" (PGM) fileformat is split into the header and the body (with image
     * data). The header starts with a magic number, then width and height (in ASCII) are specified
     * and the last part of the header is the maximum value of the colour component. The Magic
     * number in our case is "P2" as we want to write a grey scale image. The width and height are
     * taken from the `DataContainer` and the maximum value is 255 (values are scaled accordingly)
     *
     * An example of a header would be:
     * ```
     * P2
     * 100 50 # width height
     * 255    # Max value, 1 byte
     * ```
     *
     * Then the body prints one value in each line
     *
     * Note: This class currently only handles 2D `DataContainer`s.
     *
     * Reference: http://paulbourke.net/dataformats/ppm/
     *
     * @author David Frank - initial code
     */
    class PGM
    {
    public:
        /// write the DataContainer to the file named filename. Currently we only handle 2D images
        template <typename data_t = real_t>
        static void write(const DataContainer<data_t>& data, const std::string& filename);

        /// write the DataContainer to ostream with the specified format. Only 2D buffers are
        /// supported
        template <typename data_t = real_t>
        static void write(const DataContainer<data_t>& data, std::ostream& stream);
    };
} // namespace elsa
