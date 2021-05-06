/**
 * @file test_ioUtils.cpp
 *
 * @brief Tests for the ioUtils
 *
 * @author Tobias Lasser - initial code
 */

#include "doctest/doctest.h"
#include "ioUtils.h"
#include "elsaDefines.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("io");

TEST_CASE("ioUtils: testing the StringUtils")
{
    GIVEN("a string with whitespace at beginning/end")
    {
        std::string testString{"   Test String   "};

        WHEN("trimming the string")
        {
            StringUtils::trim(testString);

            THEN("the string is trimmed")
            {
                std::string expectedTrimmedString("Test String");
                REQUIRE(testString == expectedTrimmedString);
            }
        }
    }

    GIVEN("a mixed upper/lower case string")
    {
        std::string testString("aBcDeFGhIJKlM");

        WHEN("transforming the string to lower case")
        {
            StringUtils::toLower(testString);

            THEN("the string is lower case")
            {
                std::string expectedLowerCaseString("abcdefghijklm");
                REQUIRE(testString == expectedLowerCaseString);
            }
        }

        WHEN("transforming the string to upper case")
        {
            StringUtils::toUpper(testString);

            THEN("the string is upper case")
            {
                std::string expectedUpperCaseString("ABCDEFGHIJKLM");
                REQUIRE(testString == expectedUpperCaseString);
            }
        }
    }
}

TEST_CASE("ioUtils: testing the DataUtils")
{
    GIVEN("a data type")
    {
        WHEN("testing for its size")
        {
            THEN("the results is as expected")
            {
                REQUIRE(DataUtils::getSizeOfDataType(DataUtils::DataType::INT8) == 1);
                REQUIRE(DataUtils::getSizeOfDataType(DataUtils::DataType::UINT8) == 1);
                REQUIRE(DataUtils::getSizeOfDataType(DataUtils::DataType::INT16) == 2);
                REQUIRE(DataUtils::getSizeOfDataType(DataUtils::DataType::UINT16) == 2);
                REQUIRE(DataUtils::getSizeOfDataType(DataUtils::DataType::INT32) == 4);
                REQUIRE(DataUtils::getSizeOfDataType(DataUtils::DataType::UINT32) == 4);
                REQUIRE(DataUtils::getSizeOfDataType(DataUtils::DataType::FLOAT32) == 4);
                REQUIRE(DataUtils::getSizeOfDataType(DataUtils::DataType::FLOAT64) == 8);
            }
        }
    }

    GIVEN("strings containing numbers")
    {
        std::string testStringInt("12345");
        std::string testStringFloat("12.456");

        WHEN("parsing these strings")
        {
            auto intNumber = DataUtils::parse<index_t>(testStringInt);
            auto floatNumber = DataUtils::parse<real_t>(testStringFloat);

            THEN("the result is correct")
            {
                REQUIRE(intNumber == 12345);
                REQUIRE(floatNumber == 12.456f);
            }
        }

        WHEN("supplying garbage")
        {
            std::string testStringGarbage("b6c");

            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(DataUtils::parse<index_t>(testStringGarbage), Error);
            }
        }
    }

    GIVEN("strings containing a vector of numbers")
    {
        std::string testStringInt("1 2 3");
        std::string testStringFloat("1.2 3.4");

        WHEN("parsing these strings")
        {
            auto intVector = DataUtils::parseVector<index_t>(testStringInt);
            auto floatVector = DataUtils::parseVector<real_t>(testStringFloat);

            THEN("the result is correct")
            {
                REQUIRE(intVector.size() == 3);
                REQUIRE(intVector[0] == 1);
                REQUIRE(intVector[1] == 2);
                REQUIRE(intVector[2] == 3);

                REQUIRE(floatVector.size() == 2);
                REQUIRE(floatVector[0] == 1.2f);
                REQUIRE(floatVector[1] == 3.4f);
            }
        }

        WHEN("supplying garbage")
        {
            std::string testStringGarbage("1 3.4 abc");

            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(DataUtils::parseVector<index_t>(testStringGarbage), Error);
            }
        }
    }
}

TEST_SUITE_END();
