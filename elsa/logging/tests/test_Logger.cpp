/**
 * \file test_Logger.cpp
 *
 * \brief Tests for the Logger class
 *
 * \author Tobias Lasser - initial code
 */

#include <catch2/catch.hpp>
#include "Logger.h"

using namespace elsa;

SCENARIO("Using Loggers")
{
    GIVEN("a logger")
    {
        std::string name{"test"};
        auto testLogger = Logger::get(name);

        WHEN("checking the logger")
        {
            THEN("the parameters fit")
            {
                REQUIRE(testLogger->name() == name);
                REQUIRE(testLogger->level() == spdlog::level::info);
                REQUIRE(testLogger->sinks().size() == 1);
            }
        }

        WHEN("getting the logger again")
        {
            auto sameLogger = Logger::get(name);

            THEN("it has the same settings")
            {
                REQUIRE(sameLogger->name() == name);
                REQUIRE(sameLogger->level() == spdlog::level::info);
                REQUIRE(sameLogger->sinks().size() == 1);
            }
        }

        WHEN("setting the log level")
        {
            Logger::setLevel(Logger::LogLevel::ERR);

            THEN("our logger is updated to that level")
            {
                REQUIRE(testLogger->level() == spdlog::level::err);
            }

            THEN("new loggers have the correct level")
            {
                auto newLogger = Logger::get("newLogger");
                REQUIRE(newLogger->level() == spdlog::level::err);
            }
        }

        WHEN("actually logging")
        {
            testLogger->info("This is a test");
            testLogger->warn("This is a warning test");

            THEN("things didnt blow up... :-)") { REQUIRE(true); }
        }

        WHEN("adding file logging")
        {
            std::string filename = "log.txt";
            Logger::enableFileLogging(filename);

            THEN("file sink should be active") { REQUIRE(testLogger->sinks().size() == 2); }

            THEN("a new logger has file logging enabled")
            {
                auto newLogger = Logger::get("fileLogger");
                REQUIRE(newLogger->sinks().size() == 2);

                newLogger->info("This is an info");
                REQUIRE(true);
            }

            THEN("actually logging works")
            {
                testLogger->info("This is another warning");
                REQUIRE(true);
            }
        }
    }
}
