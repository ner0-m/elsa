/**
 * @file test_MHDHandler.cpp
 *
 * @brief Tests for the MHDHandler class
 *
 * @author Tobias Lasser - initial code
 */

#include "doctest/doctest.h"
#include "MHDHandler.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("io");

TEST_CASE("MHDHandler: Reading and writing data")
{
    GIVEN("a DataContainer")
    {
        VolumeDescriptor dd({11, 17});
        DataContainer dc(dd);
        dc = 1;

        WHEN("writing out and reading in this DataContainer")
        {
            std::string filename{"test.mhd"};
            std::string rawFile{"test.raw"};
            MHD::write(dc, filename, rawFile);
            auto dcRead = MHD::read(filename);

            THEN("the read in DataContainer contains the expected data")
            {
                REQUIRE_EQ(dc.getSize(), dcRead.getSize());
                REQUIRE_EQ(dc.getDataDescriptor(), dcRead.getDataDescriptor());

                REQUIRE_EQ(dcRead, dc);
            }
        }
    }
}

TEST_SUITE_END();
