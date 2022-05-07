/**
 * @file test_PGMHandler.cpp
 *
 * @brief Tests for the PGMHandler class
 *
 * @author David Frank - initial code
 */

#include "doctest/doctest.h"
#include "Error.h"
#include "PGMHandler.h"

#include "VolumeDescriptor.h"

#include <sstream>
#include <string_view>

using namespace elsa;
using namespace doctest;

inline bool file_exists(std::string name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

TEST_SUITE_BEGIN("io");

TEST_CASE("PGMHandler: Write PGM file")
{
    GIVEN("A 2D DataContainer")
    {
        IndexVector_t numCoeff{{10, 10}};
        VolumeDescriptor dd(numCoeff);
        DataContainer dc(dd);

        for (int i = 0; i < dc.getSize(); ++i) {
            dc[i] = static_cast<real_t>(i) + 10;
        }

        // Also compute the max value and the scale factor
        const auto maxVal = dc.maxElement();
        const auto minVal = dc.minElement();

        WHEN("Writing the DataContainer to the PGM format")
        {
            std::stringstream buf;
            PGM::write(dc, buf);

            // Get string from buffer
            auto str = buf.str();

            // Lambda to get the next token from `str` until `delimiter`
            // remove the part from `str` and return it
            auto nextToken = [](std::string& str, std::string_view delimiter) {
                auto pos = str.find(delimiter);
                auto token = str.substr(0, pos);
                str.erase(0, pos + delimiter.length());
                return token;
            };

            THEN("The header is correct")
            {
                CHECK_EQ(nextToken(str, "\n"), "P2");
                CHECK_EQ(nextToken(str, "\n"), "10 10");
                CHECK_EQ(nextToken(str, "\n"), "255");
            }

            THEN("The body is correct")
            {
                // Pop the header again
                nextToken(str, "\n");
                nextToken(str, "\n");
                nextToken(str, "\n");

                int counter = 0;
                while (!str.empty()) {
                    const auto normalized = (dc[counter] - minVal) / (maxVal - minVal);
                    auto val = static_cast<int>(normalized * 255.f);
                    REQUIRE_EQ(std::to_string(val), nextToken(str, " "));
                    counter++;
                }
            }
        }
    }

    GIVEN("A 1D DataContainer")
    {
        IndexVector_t numCoeff1d{{10}};
        VolumeDescriptor dd1d(numCoeff1d);
        DataContainer dc1d(dd1d);
        THEN("An exception is raised")
        {
            REQUIRE_THROWS_AS(PGM::write(dc1d, "test.pgm"), InvalidArgumentError);
        }
    }

    GIVEN("A 3D DataContainer")
    {
        IndexVector_t numCoeff3d{{10, 8, 17}};
        VolumeDescriptor dd(numCoeff3d);
        DataContainer dc(dd);
        THEN("An exception is raised")
        {
            REQUIRE_THROWS_AS(PGM::write(dc, "test.pgm"), InvalidArgumentError);
        }
    }
}

TEST_SUITE_END();
