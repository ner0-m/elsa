#pragma once

#include "elsa.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "ioUtils.h"

#include <fstream>
#include <map>
#include <string>

namespace elsa
{
    /**
     * \brief class to read and write EDF files.
     *
     * \author Matthias Wieczorek - initial code
     * \author Maximilian Hornung - modularization
     * \author Tobias Lasser - modernization
     *
     * Class to handle reading EDF files into DataContainers and writing of DataContainers to EDF files.
     *
     * EDF files are non-standard files that include a 1024 Byte header containing textual meta data
     * about the following raw data. It allows storing images of arbitrary dimensions in a low-overhead manner.
     *
     * Please note: we assume little endian byte order.
     */
    class EDF {
    public:
        /// read from filename into a DataContainer
        template <typename data_t = real_t>
        static DataContainer<data_t> read(std::string filename);

        /// write the DataContainer to the file named filename
        template <typename data_t = real_t>
        static void write(const DataContainer<data_t>& data, std::string filename);

    private:
        /// read the EDF header into a property map
        static std::map<std::string, std::string> readHeader(std::ifstream& file);

        /// parse the EDF header property map into a DataDescriptor and DataType
        static std::pair<DataDescriptor, DataUtils::DataType> parseHeader(const std::map<std::string, std::string>& properties);

        /// write the EDF header to file
        template <typename data_t>
        static void writeHeader(std::ofstream& file, const DataContainer<data_t>& data);

        /// return the EDF string for data type of DataContainer
        template <typename data_t>
        static std::string getDataTypeName(const DataContainer<data_t>& data);
    };

} // namespace elsa
