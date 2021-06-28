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

TEST_CASE("EDFHandler: Reading and writing data")
{
    GIVEN("a DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 17;
        VolumeDescriptor dd(numCoeff);
        DataContainer dc(dd);
        dc = 1;

        WHEN("writing out and reading in this DataContainer")
        {
            std::string filename{"test.edf"};
            EDF::write(dc, filename);
            auto dcRead = EDF::read(filename);

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
