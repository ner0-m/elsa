/**
 * \file test_MHDHandler.cpp
 *
 * \brief Tests for the MHDHandler class
 *
 * \author Tobias Lasser - initial code
 */

#include <catch2/catch.hpp>
#include "MHDHandler.h"

using namespace elsa;

SCENARIO("Reading and write data with MHDHandler")
{
    GIVEN("a DataContainer")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 17;
        DataDescriptor dd(numCoeff);
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
                REQUIRE(dc.getSize() == dcRead.getSize());
                REQUIRE(dc.getDataDescriptor() == dcRead.getDataDescriptor());

                REQUIRE(dcRead == dc);
            }
        }
    }
}