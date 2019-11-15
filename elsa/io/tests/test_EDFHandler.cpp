/**
 * \file test_EDFHandler.cpp
 *
 * \brief Tests for the EDFHandler class
 *
 * \author Tobias Lasser - initial code
 */

#include <catch2/catch.hpp>
#include "EDFHandler.h"

using namespace elsa;

SCENARIO("Reading and write data with EDFHandler")
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
            std::string filename{"test.edf"};
            EDF::write(dc, filename);
            auto dcRead = EDF::read(filename);

            THEN("the read in DataContainer contains the expected data")
            {
                REQUIRE(dc.getSize() == dcRead.getSize());
                REQUIRE(dc.getDataDescriptor() == dcRead.getDataDescriptor());

                REQUIRE(dcRead == dc);
            }
        }
    }
}