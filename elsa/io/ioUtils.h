#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"

#include <fstream>
#include <string>
#include <vector>

namespace elsa
{
    /**
     * @brief class providing string handling utility functions.
     *
     * @author Maximilian Hornung - initial code
     * @author Tobias Lasser - rewrite
     */
    struct StringUtils {
    public:
        /// trim whitespace from beginning/end of string
        static void trim(std::string& str);

        /// convert string to lower case
        static void toLower(std::string& str);

        /// convert string to upper case
        static void toUpper(std::string& str);
    };

    /**
     * @brief class providing utility functions for reading/writing data
     *
     * @author Matthias Wieczorek - first version of code
     * @author Maximilian Hornung - modularization
     * @author Tobias Lasser - rewrite
     */
    struct DataUtils {
    public:
        /// byte order
        enum class ByteOrder { LOW_BYTE_FIRST, HIGH_BYTE_FIRST };

        /// our default byte order is little endian (low byte first)
        static const ByteOrder DEFAULT_BYTE_ORDER = ByteOrder::LOW_BYTE_FIRST;

        /// data types
        enum class DataType { INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32, FLOAT64 };

        /// return the size in bytes for the respective DataType
        static index_t getSizeOfDataType(DataType type);

        /// parse a data_t value from a string
        template <typename data_t>
        static data_t parse(const std::string& str);

        /// parse a vector of data_t values from a string
        template <typename data_t>
        static std::vector<data_t> parseVector(const std::string& str);

        /// read in raw data (of type raw_data_t) into a data container (of type data_t)
        template <typename raw_data_t, typename data_t>
        static void parseRawData(std::istream& file, DataContainer<data_t>& data);
    };

    /**
     * @brief class providing utility functions for the filesystem
     *
     * @author Maximilian Hornung - initial code
     * @author Tobias Lasser - rewrite
     */
    struct FileSystemUtils {
    public:
        /// return the absolute path of path with respect to base
        static std::string getAbsolutePath(std::string path, std::string base);
    };

} // namespace elsa
