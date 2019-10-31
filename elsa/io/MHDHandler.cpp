#include "MHDHandler.h"
#include "Logger.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    DataContainer<data_t> MHD::read(std::string filename)
    {
        Logger::get("MHD")->info("Reading data from {}", filename);

        // open the meta file
        std::ifstream metaFile(filename);
        if (!metaFile.good())
            throw std::runtime_error("MHD::read: cannot read from '" + filename + "'");

        // get the meta data from the meta file
        auto properties = readHeader(metaFile);
        auto [descriptor, dataPath, dataType] = parseHeader(properties);

        std::string dataFilename = FileSystemUtils::getAbsolutePath(dataPath, filename);
        std::ifstream dataFile(dataFilename, std::ios::binary | std::ios::in);
        if (!dataFile.good())
            throw std::runtime_error("MHD::read: can not read from '" + dataPath + "'");

        // read in the data
        DataContainer<data_t> dataContainer(descriptor);

        if (dataType == DataUtils::DataType::UINT16)
            DataUtils::parseRawData<uint16_t, data_t>(dataFile, dataContainer);
        else if (dataType == DataUtils::DataType::FLOAT32)
            DataUtils::parseRawData<float, data_t>(dataFile, dataContainer);
        else if (dataType == DataUtils::DataType::FLOAT64)
            DataUtils::parseRawData<double, data_t>(dataFile, dataContainer);
        else
            throw std::logic_error("MHD::read: invalid/unsupported data type");

        return dataContainer;
    }

    template <typename data_t>
    void MHD::write(const DataContainer<data_t>& data, std::string metaFilename,
                    std::string rawFilename)
    {
        Logger::get("MHD")->info("Writing meta data to {} and raw data to {}", metaFilename,
                                 rawFilename);

        // open the meta file
        std::ofstream metaFile(metaFilename);
        if (!metaFile.good())
            throw std::runtime_error("MHD::write: cannot write to '" + metaFilename + "'");

        // output the header to the meta file
        writeHeader(metaFile, data, rawFilename);

        // open the raw file
        std::ofstream rawFile(rawFilename, std::ios::binary);
        if (!rawFile.good())
            throw std::runtime_error("MHD::write: cannot write to '" + rawFilename + "'");

        // output the raw data
        // TODO: this would be more efficient if we had a data pointer...
        for (index_t i = 0; i < data.getSize(); ++i)
            rawFile.write(reinterpret_cast<const char*>(&data[i]), sizeof(data_t));
    }

    std::map<std::string, std::string> MHD::readHeader(std::ifstream& metaFile)
    {
        std::map<std::string, std::string> properties;

        // read header data
        std::string metaLine;
        while (!metaFile.eof()) {
            // read next header line
            std::getline(metaFile, metaLine);
            StringUtils::trim(metaLine);

            if (metaLine.length() == 0u)
                continue;

            // split the header line into name and value
            size_t delim = metaLine.find('=');
            if (delim == std::string::npos)
                throw std::runtime_error(
                    "MHD::readHeader: found non-empty line without name/value pair");

            std::string name = metaLine.substr(0, delim);
            StringUtils::trim(name);
            std::string value = metaLine.substr(delim + 1);
            StringUtils::trim(value);

            StringUtils::toLower(name);
            properties[name] = value;
        }

        return properties;
    }

    std::tuple<DataDescriptor, std::string, DataUtils::DataType>
        MHD::parseHeader(const std::map<std::string, std::string>& properties)
    {
        // check the dimensions
        auto nDimsIt = properties.find("ndims");
        std::size_t nDims;
        if (nDimsIt != properties.end())
            nDims = DataUtils::parse<index_t>(nDimsIt->second);
        else
            throw std::runtime_error("MHD::parseHeader: tag 'ndims' not found");

        // check for a byte order tag, but fall back to the default value
        auto byteOrderIt = properties.find("elementbyteordermsb");
        if (byteOrderIt != properties.end()) {
            std::string byteOrderValue = byteOrderIt->second;
            StringUtils::toLower(byteOrderValue);

            if (byteOrderValue != "false" && byteOrderValue != "no")
                throw std::runtime_error(
                    "MHD::parseHeader: only supporting little endian byte order");
        }

        // check for the 'element type' value
        DataUtils::DataType dataType;
        auto elementTypeIt = properties.find("elementtype");
        if (elementTypeIt != properties.end()) {
            std::string elementTypeValue = elementTypeIt->second;
            StringUtils::toUpper(elementTypeValue);

            if (elementTypeValue == "MET_CHAR")
                dataType = DataUtils::DataType::INT8;
            else if (elementTypeValue == "MET_UCHAR")
                dataType = DataUtils::DataType::UINT8;
            else if (elementTypeValue == "MET_SHORT")
                dataType = DataUtils::DataType::INT16;
            else if (elementTypeValue == "MET_USHORT")
                dataType = DataUtils::DataType::UINT16;
            else if (elementTypeValue == "MET_FLOAT")
                dataType = DataUtils::DataType::FLOAT32;
            else if (elementTypeValue == "MET_DOUBLE")
                dataType = DataUtils::DataType::FLOAT64;
            else
                throw std::runtime_error(
                    "MHD::parseHeader: tag 'element type' of unsupported value");
        } else
            throw std::runtime_error("MHD::parseHeader: tag 'element type' not found");

        // extract the data path
        std::string rawDataPath;
        auto elementDataFileIt = properties.find("elementdatafile");
        if (elementDataFileIt != properties.end())
            rawDataPath = elementDataFileIt->second;
        else
            throw std::runtime_error("MHD::parseHeader: tag 'element data file' not found");

        // parse the extents
        std::vector<index_t> dimSizeVec;
        auto dimSizeIt = properties.find("dimsize");
        if (dimSizeIt != properties.end()) {
            dimSizeVec = DataUtils::parseVector<index_t>(dimSizeIt->second);
            if (dimSizeVec.size() != nDims)
                throw std::runtime_error("MHD::parseHeader: dimension size mismatch");
        } else
            throw std::runtime_error("MHD::parseHeader: tag 'dim size' not found");

        // check for spacing
        std::vector<real_t> dimSpacingVec;
        auto dimSpacingIt = properties.find("elementspacing");
        if (dimSpacingIt != properties.end()) {
            dimSpacingVec = DataUtils::parseVector<real_t>(dimSpacingIt->second);
            if (dimSpacingVec.size() != nDims)
                throw std::runtime_error("MHD::parseHeader: spacing size mismatch");
        }

        // convert size
        IndexVector_t dimSizes(nDims);
        for (index_t i = 0; i < nDims; ++i)
            dimSizes[i] = dimSizeVec[i];

        // convert spacing
        RealVector_t dimSpacing(RealVector_t::Ones(nDims));
        if (!dimSpacingVec.empty()) {
            for (index_t i = 0; i < nDims; ++i)
                dimSpacing[i] = dimSpacingVec[i];
        }

        // the data descriptor condensed form the info
        DataDescriptor dataDescriptor(dimSizes, dimSpacing);

        return std::make_tuple(dataDescriptor, rawDataPath, dataType);
    }

    template <typename data_t>
    void MHD::writeHeader(std::ofstream& metaFile, const DataContainer<data_t>& data,
                          std::string rawFilename)
    {
        auto descriptor = data.getDataDescriptor();

        // write dimension, size and spacing
        metaFile << "NDims = " << descriptor.getNumberOfDimensions() << "\n";
        metaFile << "DimSize = " << (descriptor.getNumberOfCoefficientsPerDimension()).transpose()
                 << "\n";
        metaFile << "ElementSpacing = " << (descriptor.getSpacingPerDimension()).transpose()
                 << "\n";

        // write the data type and the byte swapping flag
        metaFile << "ElementType = " << getDataTypeName(data) << "\n";

        metaFile << "ElementByteOrderMSB = False"
                 << "\n";

        // write the data path and close
        metaFile << "ElementDataFile = " << rawFilename << std::endl;
    }

    template <typename data_t>
    std::string MHD::getDataTypeName(const DataContainer<data_t>& data)
    {
        throw std::invalid_argument("MHD::getDataTypeName: invalid/unsupported data type");
    }

    template <>
    std::string MHD::getDataTypeName(const DataContainer<int8_t>& data)
    {
        return "MET_CHAR";
    }
    template <>
    std::string MHD::getDataTypeName(const DataContainer<uint8_t>& data)
    {
        return "MET_UCHAR";
    }
    template <>
    std::string MHD::getDataTypeName(const DataContainer<int16_t>& data)
    {
        return "MET_SHORT";
    }
    template <>
    std::string MHD::getDataTypeName(const DataContainer<uint16_t>& data)
    {
        return "MET_USHORT";
    }
    template <>
    std::string MHD::getDataTypeName(const DataContainer<float>& data)
    {
        return "MET_FLOAT";
    }
    template <>
    std::string MHD::getDataTypeName(const DataContainer<double>& data)
    {
        return "MET_DOUBLE";
    }

    // ------------------------------------------
    // explicit template instantiation
    template DataContainer<float> MHD::read(std::string);
    template DataContainer<double> MHD::read(std::string);
    template DataContainer<index_t> MHD::read(std::string);
    template void MHD::write(const DataContainer<float>&, std::string, std::string);
    template void MHD::write(const DataContainer<double>&, std::string, std::string);
    template void MHD::write(const DataContainer<index_t>&, std::string, std::string);

    template void MHD::writeHeader(std::ofstream&, const DataContainer<float>&, std::string);
    template void MHD::writeHeader(std::ofstream&, const DataContainer<double>&, std::string);
    template void MHD::writeHeader(std::ofstream&, const DataContainer<index_t>&, std::string);

} // namespace elsa
