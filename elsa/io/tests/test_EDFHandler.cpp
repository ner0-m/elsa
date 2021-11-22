/**
 * @file test_EDFHandler.cpp
 *
 * @brief Tests for the EDFHandler class
 *
 * @author Tobias Lasser - initial code
 */

#include "doctest/doctest.h"
#include "EDFHandler.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("io");

std::vector<std::string> split(const std::string& s, char delim)
{
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        elems.push_back(std::move(item));
    }
    return elems;
}

TEST_CASE("EDFHandler: Reading and writing data")
{
    GIVEN("a 1D DataContainer")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 5;
        DataContainer dc(VolumeDescriptor{numCoeff});
        dc = 0;

        WHEN("Writing it to an edf format")
        {
            std::stringstream buffer;
            EDF::write(dc, buffer);

            THEN("The header is correct")
            {
                auto buffer_string = buffer.str();
                auto header = buffer_string.substr(0, buffer_string.find('}' + 1));

                auto parts = split(header, '\n');
                parts.erase(std::begin(parts)); // remove the '{'
                parts.pop_back();               // remove empty newline
                parts.pop_back();               // remove closing '}'
                parts.pop_back();               // remove filling spaces

                CHECK_EQ(parts[0], "HeaderID = EH:000001:000000:000000;");
                CHECK_EQ(parts[1], "Image = 1;");
                CHECK_EQ(parts[2], "ByteOrder = LowByteFirst;");
                CHECK_EQ(parts[3], "DataType = FloatValue;");
                CHECK_EQ(parts[4], "Dim_1 = 5;");   // dim
                CHECK_EQ(parts[5], "Size = 20;");   // dim * sizeof(float)
                CHECK_EQ(parts[6], "Spacing = 1;"); // default 1D spacing
            }

            WHEN("Reading it back in")
            {
                auto read = EDF::read(buffer);

                CHECK_EQ(dc.getSize(), read.getSize());
                CHECK_EQ(dc.getDataDescriptor(), read.getDataDescriptor());

                CHECK_EQ(dc, read);
            }
        }
    }

    GIVEN("a 2D DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 5, 10;
        DataContainer dc(VolumeDescriptor{numCoeff});
        dc = 0;

        WHEN("Writing it to an edf format")
        {
            std::stringstream buffer;
            EDF::write(dc, buffer);

            THEN("The header is correct")
            {
                auto buffer_string = buffer.str();
                auto header = buffer_string.substr(0, buffer_string.find('}' + 1));

                auto parts = split(header, '\n');
                parts.erase(std::begin(parts)); // remove the '{'
                parts.pop_back();               // remove empty newline
                parts.pop_back();               // remove closing '}'
                parts.pop_back();               // remove filling spaces

                CHECK_EQ(parts[0], "HeaderID = EH:000001:000000:000000;");
                CHECK_EQ(parts[1], "Image = 1;");
                CHECK_EQ(parts[2], "ByteOrder = LowByteFirst;");
                CHECK_EQ(parts[3], "DataType = FloatValue;");
                CHECK_EQ(parts[4], "Dim_1 = 5;");     // dim 1
                CHECK_EQ(parts[5], "Dim_2 = 10;");    // dim 2
                CHECK_EQ(parts[6], "Size = 200;");    // dim 1 * dim 2 * sizeof(float)
                CHECK_EQ(parts[7], "Spacing = 1 1;"); // default 2D spacing
            }

            WHEN("Reading it back in")
            {
                auto read = EDF::read(buffer);

                CHECK_EQ(dc.getSize(), read.getSize());
                CHECK_EQ(dc.getDataDescriptor(), read.getDataDescriptor());

                CHECK_EQ(dc, read);
            }
        }
    }

    GIVEN("a DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 17;
        VolumeDescriptor dd(numCoeff);
        DataContainer dc(dd);
        dc = 1;

        WHEN("writing out and reading in this DataContainer")
        {
            std::stringstream buffer;
            EDF::write(dc, buffer);
            auto dcRead = EDF::read(buffer);

            THEN("the read in DataContainer contains the expected data")
            {
                CHECK_EQ(dc.getSize(), dcRead.getSize());
                CHECK_EQ(dc.getDataDescriptor(), dcRead.getDataDescriptor());

                CHECK_EQ(dcRead, dc);
            }
        }
    }
}

TEST_SUITE_END();
