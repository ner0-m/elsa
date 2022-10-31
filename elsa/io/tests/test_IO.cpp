
#include "doctest/doctest.h"
#include "IO.h"

#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;
TEST_SUITE_BEGIN("io");

TEST_CASE_TEMPLATE("IO: Testing exception behaviour of read", data_t, float, double)
{
    WHEN("Trying to read from a string, without an extension")
    {
        CHECK_THROWS_WITH_AS(io::read<data_t>("hello"),
                             "No extension found in filename (\"hello\")", Error);
    }

    WHEN("Trying to read from a string, with an unsupported extension")
    {
        CHECK_THROWS_WITH_AS(io::read<data_t>("hello.something"),
                             "Can not read with unsupported file extension \".something\"", Error);
    }

    WHEN("Trying to read from a string, with a valid extension")
    {
        // Only throws as file is not present
        CHECK_THROWS_WITH_AS(io::read<data_t>("hello.edf"),
                             "EDF::read: cannot read from 'hello.edf'", Error);
    }

    WHEN("Trying to read from a string, with an multiple dots")
    {
        // Only throws as file is not present
        CHECK_THROWS_WITH_AS(io::read<data_t>("hello.something.edf"),
                             "EDF::read: cannot read from 'hello.something.edf'", Error);
    }
}

TEST_CASE_TEMPLATE("IO: Testing exception behaviour of write", data_t, float, double)
{
    DataContainer<data_t> x(VolumeDescriptor({32, 32}));
    x = 1;

    WHEN("Writing an edf file")
    {
        THEN("it works")
        {
            CHECK_NOTHROW(io::write(x, "hellosomething.edf"));
        }
    }

    WHEN("Writing an pgm file")
    {
        THEN("it works")
        {
            CHECK_NOTHROW(io::write(x, "hellosomething.pgm"));
        }
    }

    WHEN("Writing an with an unsupported extension")
    {
        CHECK_THROWS_WITH_AS(io::write(x, "hellosomethingelse.png"),
                             "Can not write with unsupported file extension \".png\"", Error);
    }

    WHEN("Writing without an extension given")
    {
        CHECK_THROWS_WITH_AS(io::write(x, "hello"), "No extension found in filename (\"hello\")",
                             Error);
    }
}

TEST_SUITE_END();
