#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "ioUtils.h"

#include <fstream>
#include <map>
#include <string>
#include <tuple>

namespace elsa
{
    /**
     * @brief class to read and write MHD files. MHD files are compatible with ITK/VTK based
     * applications.
     *
     * @author Matthias Wieczorek - initial code
     * @author Maximilian Hornung - modularization
     * @author Tobias Lasser - modernization
     *
     * Class to handle reading MHD files into DataContainers and writing of DataContainers to MHD
     * files.
     *
     * MHD files consist of two parts, a text file (ending in .mhd) containing meta data, and a raw
     * file containing the image data (filename is referenced in meta data). It allows storing
     * images of arbitrary dimensions in a low-overhead manner.
     *
     * Please note: we assume little endian byte order.
     */
    class MHD
    {
    public:
        /// read from filename into a DataContainer
        template <typename data_t = real_t>
        static DataContainer<data_t> read(std::string filename);

        /// write the DataContainer to the file named filename
        template <typename data_t = real_t>
        static void write(const DataContainer<data_t>& data, std::string metaFilename,
                          std::string rawFilename);

    private:
        /// read the MHD header into a property map
        static std::map<std::string, std::string> readHeader(std::ifstream& metaFile);

        /// parse the MHD header property map into a DataDescriptor and DataType
        static std::tuple<std::unique_ptr<DataDescriptor>, std::string, DataUtils::DataType>
            parseHeader(const std::map<std::string, std::string>& properties);

        /// write the MHD header to file
        template <typename data_t>
        static void writeHeader(std::ofstream& metaFile, const DataContainer<data_t>& data,
                                std::string rawFilename);

        /// return the MHD string for data type of DataContainer
        template <typename data_t>
        static std::string getDataTypeName(const DataContainer<data_t>& data);
    };

} // namespace elsa
