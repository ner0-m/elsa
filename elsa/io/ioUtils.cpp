#include "ioUtils.h"
#include "elsa.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace elsa
{
    void StringUtils::trim(std::string& str)
    {
        // trim whitespace from beginning
        str.erase(str.begin(),
                  std::find_if(str.begin(), str.end(), [](int ch) { return !std::isspace(ch); }));
        // trim whitespace from end
        str.erase(
            std::find_if(str.rbegin(), str.rend(), [](int ch) { return !std::isspace(ch); }).base(),
            str.end());
    }

    void StringUtils::toLower(std::string& str)
    {
        std::transform(str.begin(), str.end(), str.begin(),
                       [](unsigned char c) { return std::tolower(c); });
    }

    void StringUtils::toUpper(std::string& str)
    {
        std::transform(str.begin(), str.end(), str.begin(),
                       [](unsigned char c) { return std::toupper(c); });
    }

    index_t DataUtils::getSizeOfDataType(DataUtils::DataType type)
    {
        switch (type) {
            case DataType::INT8:
            case DataType::UINT8:
                return 1;
            case DataType::INT16:
            case DataType::UINT16:
                return 2;
            case DataType::INT32:
            case DataType::UINT32:
            case DataType::FLOAT32:
                return 4;
            case DataType::FLOAT64:
                return 8;

            default:
                throw std::invalid_argument("DataUtils::getSizeOfDataType: unknown data type");
        }
    }

    template <typename data_t>
    data_t DataUtils::parse(const std::string& str)
    {
        data_t value;
        std::stringstream convert(str);
        convert >> value;
        if (convert.fail())
            throw std::runtime_error("DataUtils::parse: failed to interpret string");
        return value;
    }

    template <typename data_t>
    std::vector<data_t> DataUtils::parseVector(const std::string& str)
    {
        std::vector<data_t> dataVector;

        data_t value;
        std::stringstream convert(str);
        while (!convert.eof()) {
            convert >> value;
            if (convert.fail())
                throw std::runtime_error("DataUtils::parseVector: failed to interpret string");
            dataVector.push_back(value);
        }

        return dataVector;
    }

    template <typename raw_data_t, typename data_t>
    void DataUtils::parseRawData(std::ifstream& file, DataContainer<data_t>& data)
    {
        index_t sizeInElements = data.getSize();
        index_t sizeInBytes = sizeInElements * sizeof(raw_data_t);

        // allocate temporary storage
        auto ptr = std::make_unique<raw_data_t[]>(sizeInElements);
        if (!ptr)
            throw std::runtime_error("DataUtils::parseRawData: failed allocating memory");

        // parse data into the storage
        file.read(reinterpret_cast<char*>(ptr.get()), sizeInBytes);
        if (file.gcount() != sizeInBytes)
            throw std::runtime_error("DataUtils::parseRawData: failed to read sufficient data");

        // perform a component-wise copy to the data container
        for (index_t i = 0; i < sizeInElements; ++i)
            data[i] = static_cast<data_t>(ptr[i]);
    }

    std::string FileSystemUtils::getAbsolutePath(std::string path, std::string base)
    {
        // note: this should really be done with C++17 <filesystem>... if it were universally
        // available

        // split off filename at end
        auto found = base.find_last_of("/\\");
        if (found == std::string::npos)
            base = ".";
        else
            base = base.substr(0, found);

        // use POSIX realpath [TODO: will not work on Windows!]
        char* resolved = realpath(base.c_str(), nullptr);
        std::string basePath(resolved);
        free(resolved);

        return (basePath + "/" + path);
    }

    // ------------------------------------------
    // explicit template instantiation
    template float DataUtils::parse(const std::string&);
    template double DataUtils::parse(const std::string&);
    template index_t DataUtils::parse(const std::string&);

    template std::vector<float> DataUtils::parseVector(const std::string&);
    template std::vector<double> DataUtils::parseVector(const std::string&);
    template std::vector<index_t> DataUtils::parseVector(const std::string&);

    template void DataUtils::parseRawData<uint16_t, float>(std::ifstream&, DataContainer<float>&);
    template void DataUtils::parseRawData<float, float>(std::ifstream&, DataContainer<float>&);
    template void DataUtils::parseRawData<double, float>(std::ifstream&, DataContainer<float>&);
    template void DataUtils::parseRawData<uint16_t, double>(std::ifstream&, DataContainer<double>&);
    template void DataUtils::parseRawData<float, double>(std::ifstream&, DataContainer<double>&);
    template void DataUtils::parseRawData<double, double>(std::ifstream&, DataContainer<double>&);
    template void DataUtils::parseRawData<uint16_t, index_t>(std::ifstream&,
                                                             DataContainer<index_t>&);
    template void DataUtils::parseRawData<float, index_t>(std::ifstream&, DataContainer<index_t>&);
    template void DataUtils::parseRawData<double, index_t>(std::ifstream&, DataContainer<index_t>&);
} // namespace elsa
