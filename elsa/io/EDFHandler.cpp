#include "EDFHandler.h"
#include "Logger.h"
#include "VolumeDescriptor.h"

#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    DataContainer<data_t> EDF::read(std::string filename)
    {
        Logger::get("EDF")->info("Reading data from {}", filename);

        // open the file
        std::ifstream file(filename, std::ios::binary);
        if (!file.good())
            throw Error("EDF::read: cannot read from '" + filename + "'");

        // get the meta data from the header
        auto properties = readHeader(file);
        auto [descriptor, dataType] = parseHeader(properties);

        // read in the data
        DataContainer<data_t> dataContainer(*descriptor);

        if (dataType == DataUtils::DataType::UINT16)
            DataUtils::parseRawData<uint16_t, data_t>(file, dataContainer);
        else if (dataType == DataUtils::DataType::FLOAT32)
            DataUtils::parseRawData<float, data_t>(file, dataContainer);
        else if (dataType == DataUtils::DataType::FLOAT64)
            DataUtils::parseRawData<double, data_t>(file, dataContainer);
        else
            throw Error("EDF::read: invalid/unsupported data type");

        return dataContainer;
    }

    template <typename data_t>
    void EDF::write(const DataContainer<data_t>& data, std::string filename)
    {
        Logger::get("EDF")->info("Writing data to {}", filename);

        // open the file
        std::ofstream file(filename, std::ios::binary);
        if (!file.good())
            throw Error("EDF::write: cannot write to '" + filename + "'");

        // output the header
        writeHeader(file, data);

        // output the raw data
        // TODO: this would be more efficient if we had a data pointer...
        for (index_t i = 0; i < data.getSize(); ++i)
            file.write(reinterpret_cast<const char*>(&data[i]), sizeof(data_t));
    }

    std::map<std::string, std::string> EDF::readHeader(std::ifstream& file)
    {
        std::map<std::string, std::string> properties;

        // read a single character and make sure that a header is opened
        if (file.eof() || file.get() != '{')
            throw InvalidArgumentError("EDF::readHeader: no header opening marker");

        // read header data
        while (!file.eof()) {
            // skip whitespace
            while (!file.eof()) {
                const int chr = file.peek();
                if (chr != '\r' && chr != '\n' && chr != ' ' && chr != '\t')
                    break;
                file.ignore(1);
            }

            // abort if the header is closed
            if (file.eof() || file.peek() == '}')
                break;

            // extract the property assignment
            bool quotesSingle = false, quotesDouble = false;
            std::string assignment;
            while (!file.eof()) {
                auto chr = static_cast<char>(file.get());

                // abort on end-of-assignment
                if (chr == ';' && !(quotesSingle || quotesDouble))
                    break;

                // check for quote characters
                if (chr == '\'')
                    quotesSingle = !quotesSingle;
                if (chr == '\"')
                    quotesDouble = !quotesDouble;

                assignment += chr;
            }

            // split the assignment
            auto delim = assignment.find('=');
            if (delim == std::string::npos)
                throw InvalidArgumentError("failed reading name/value delimiter");

            std::string name = assignment.substr(0, delim);
            StringUtils::trim(name);

            std::string value = assignment.substr(delim + 1);
            StringUtils::trim(value);

            // remove quotes (if they exist)
            if (value[0] == value[value.size() - 1] && (value[0] == '\'' || value[0] == '\"'))
                value = value.substr(1, value.size() - 2);

            StringUtils::toLower(name);
            properties[name] = value;
        }
        file.ignore(2); // end of header marker

        return properties;
    }

    std::pair<std::unique_ptr<DataDescriptor>, DataUtils::DataType>
        EDF::parseHeader(const std::map<std::string, std::string>& properties)
    {
        // read the dimensions
        std::vector<index_t> dim;
        for (index_t i = 1;; i++) {
            // assemble the property name
            std::stringstream aux;
            aux << "dim_" << i;

            // try to find the property
            auto dimIt = properties.find(aux.str());
            if (dimIt == properties.end())
                break;

            dim.push_back(DataUtils::parse<index_t>(dimIt->second));
        }
        const auto nDims = static_cast<index_t>(dim.size());
        if (nDims == 0u)
            throw Error("EDF::parseHeader: dimension information not found");

        // parse the (non-standard) spacing tag
        std::vector<real_t> spacing;
        auto spacingIt = properties.find("spacing");
        if (spacingIt != properties.end())
            spacing = DataUtils::parseVector<real_t>(spacingIt->second);

        // check for a byte order tag, but fall back to the default value
        auto byteorderIt = properties.find("byteorder");
        if (byteorderIt != properties.end()) {
            std::string byteorderValue = byteorderIt->second;
            StringUtils::toLower(byteorderValue);

            if (byteorderValue != "lowbytefirst")
                throw Error("EDF::parseHeader: unsupported byte order value");
        }

        // check for the 'element type' value
        DataUtils::DataType dataType;
        auto datatypeIt = properties.find("datatype");
        if (datatypeIt != properties.end()) {
            std::string datatypeValue = datatypeIt->second;
            StringUtils::toLower(datatypeValue);

            if (datatypeValue == "signedbyte")
                dataType = DataUtils::DataType::INT8;
            else if (datatypeValue == "unsignedbyte")
                dataType = DataUtils::DataType::UINT8;
            else if (datatypeValue == "signedshort")
                dataType = DataUtils::DataType::INT16;
            else if (datatypeValue == "unsignedshort")
                dataType = DataUtils::DataType::UINT16;
            else if (datatypeValue == "float" || datatypeValue == "floatvalue"
                     || datatypeValue == "real")
                dataType = DataUtils::DataType::FLOAT32;
            else if (datatypeValue == "double" || datatypeValue == "doublevalue")
                dataType = DataUtils::DataType::FLOAT64;
            else
                throw Error("EDF::parseHeader: invalid/unsupported data type");
        } else
            throw Error("EDF::parseHeader: data type not found");

        auto compressionIt = properties.find("compression");
        if (compressionIt != properties.end())
            throw Error("EDF::parseHeader: compression not supported");

        index_t size = 0;
        auto sizeIt = properties.find("size");
        if (sizeIt != properties.end())
            size = DataUtils::parse<index_t>(sizeIt->second);

        auto imageIt = properties.find("image");
        if (imageIt != properties.end() && DataUtils::parse<index_t>(imageIt->second) != 1)
            throw Error("EDF::parseHeader: image not set to 1");

        // convert size
        IndexVector_t dimSizeVec(nDims);
        for (index_t i = 0; i < nDims; ++i)
            dimSizeVec[i] = dim[static_cast<std::size_t>(i)];
        if (dimSizeVec.prod() * DataUtils::getSizeOfDataType(dataType) != size)
            throw Error("EDF::parseHeader: size inconsistency");

        // convert spacing
        RealVector_t dimSpacingVec(RealVector_t::Ones(nDims));
        if (!spacing.empty()) {
            if (nDims != static_cast<index_t>(spacing.size()))
                throw Error("EDF::parseHeader: spacing inconsistency");
            for (index_t i = 0; i < nDims; ++i)
                dimSpacingVec[i] = spacing[static_cast<std::size_t>(i)];
        }

        return std::make_pair(std::make_unique<VolumeDescriptor>(dimSizeVec, dimSpacingVec),
                              dataType);
    }

    template <typename data_t>
    void EDF::writeHeader(std::ofstream& file, const DataContainer<data_t>& data)
    {
        // open the header
        file << "{\n";

        file << "HeaderID = EH:000001:000000:000000;\n";
        file << "Image = " << 1 << ";\n";
        file << "ByteOrder = LowByteFirst;\n";
        file << "DataType = " << getDataTypeName(data) << ";\n";

        auto& descriptor = data.getDataDescriptor();

        // write dimension and size
        for (index_t i = 0; i < descriptor.getNumberOfDimensions(); ++i)
            file << "Dim_" << (i + 1) << " = "
                 << descriptor.getNumberOfCoefficientsPerDimension()[i] << ";\n";
        file << "Size = "
             << descriptor.getNumberOfCoefficients() * static_cast<index_t>(sizeof(data_t))
             << ";\n";

        // write spacing
        file << "Spacing =";
        for (index_t i = 0; i < descriptor.getNumberOfDimensions(); ++i)
            file << ' ' << descriptor.getSpacingPerDimension()[i];
        file << ";\n";

        // pad the header by adding spaces such that the header ends on a kilobyte boundary
        index_t n = 1024;
        while (n < (static_cast<index_t>(file.tellp()) + 3))
            n += 1024;
        n -= static_cast<index_t>(file.tellp()) + 3;
        while (n > 0) {
            file.put(' ');
            n--;
        }

        // close the header
        file << "\n}\n";
    }

    template <typename data_t>
    std::string EDF::getDataTypeName([[maybe_unused]] const DataContainer<data_t>& data)
    {
        throw InvalidArgumentError("EDF::getDataTypeName: invalid/unsupported data type");
    }

    template <>
    std::string EDF::getDataTypeName([[maybe_unused]] const DataContainer<int8_t>& data)
    {
        return "SignedByte";
    }
    template <>
    std::string EDF::getDataTypeName([[maybe_unused]] const DataContainer<uint8_t>& data)
    {
        return "UnsignedByte";
    }
    template <>
    std::string EDF::getDataTypeName([[maybe_unused]] const DataContainer<int16_t>& data)
    {
        return "SignedShort";
    }
    template <>
    std::string EDF::getDataTypeName([[maybe_unused]] const DataContainer<uint16_t>& data)
    {
        return "UnsignedShort";
    }
    template <>
    std::string EDF::getDataTypeName([[maybe_unused]] const DataContainer<float>& data)
    {
        return "FloatValue";
    }
    template <>
    std::string EDF::getDataTypeName([[maybe_unused]] const DataContainer<double>& data)
    {
        return "DoubleValue";
    }

    // ------------------------------------------
    // explicit template instantiation
    template DataContainer<float> EDF::read(std::string);
    template DataContainer<double> EDF::read(std::string);
    template DataContainer<index_t> EDF::read(std::string);
    template void EDF::write(const DataContainer<float>&, std::string);
    template void EDF::write(const DataContainer<double>&, std::string);
    template void EDF::write(const DataContainer<index_t>&, std::string);

    template void EDF::writeHeader(std::ofstream&, const DataContainer<float>&);
    template void EDF::writeHeader(std::ofstream&, const DataContainer<double>&);
    template void EDF::writeHeader(std::ofstream&, const DataContainer<index_t>&);

} // namespace elsa
