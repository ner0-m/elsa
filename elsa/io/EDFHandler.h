#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "ioUtils.h"

#include <fstream>
#include <istream>
#include <map>
#include <ostream>
#include <string>

namespace elsa
{
    /**
     * @brief class to read and write EDF files.
     *
     * Class to handle reading EDF files into DataContainers and writing of DataContainers to EDF
     * files.
     *
     * EDF files are non-standard files that include a 1024 Byte header containing textual meta data
     * about the following raw data. It allows storing images of arbitrary dimensions in a
     * low-overhead manner.
     *
     * Please note: we assume little endian byte order.
     *
     * @author
     * - Matthias Wieczorek - initial code
     * - Maximilian Hornung - modularization
     * - Tobias Lasser - modernization
     * - David Frank - istream overloads, improved testability
     */
    class EDF
    {
    public:
        /// read from filename into a DataContainer
        template <typename data_t = real_t>
        static DataContainer<data_t> read(std::string filename);

        /// read from stream into a DataContainer
        template <typename data_t = real_t>
        static DataContainer<data_t> read(std::istream& input);

        /// write the DataContainer to the file named filename
        template <typename data_t = real_t>
        static void write(const DataContainer<data_t>& data, std::string filename);

        /// write the DataContainer to the file named filename
        template <typename data_t = real_t>
        static void write(const DataContainer<data_t>& data, std::ostream& output);

    private:
        /// read the EDF header into a property map
        static std::map<std::string, std::string> readHeader(std::istream& file);

        /// parse the EDF header property map into a DataDescriptor and DataType
        static std::pair<std::unique_ptr<DataDescriptor>, DataUtils::DataType>
            parseHeader(const std::map<std::string, std::string>& properties);

        /// write the EDF header to file
        template <typename data_t>
        static void writeHeader(std::ostream& file, const DataContainer<data_t>& data);

        /// return the EDF string for data type of DataContainer
        template <typename data_t>
        static std::string getDataTypeName(const DataContainer<data_t>& data);
    };

} // namespace elsa
