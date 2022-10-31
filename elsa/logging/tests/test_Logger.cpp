/**
 * @file test_Logger.cpp
 *
 * @brief Tests for the Logger class
 *
 * @author Tobias Lasser - initial code
 */

#include "doctest/doctest.h"
#include "Logger.h"
#include <sstream>

using namespace elsa;
using namespace doctest;

using namespace std::string_literals;

TEST_SUITE_BEGIN("logging");

TEST_CASE("Logger: Use test")
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

        WHEN("adding file logging")
        {
            std::string filename = "log.txt";
            Logger::enableFileLogging(filename);

            THEN("We still should only have one sink")
            {
                REQUIRE(testLogger->sinks().size() == 1);
            }

            THEN("a new logger has file logging enabled")
            {
                auto newLogger = Logger::get("fileLogger");
                REQUIRE(newLogger->sinks().size() == 1);

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

    // Add buffer as sink to test the logging
    std::stringstream buffer;

    // Keep this flat, as it would at everything multiple times, which duplicates the messages
    GIVEN("A logger at info level")
    {
        Logger::addSink(buffer);

        Logger::setLevel(Logger::LogLevel::INFO);
        auto logger = Logger::get("logger");

        // Set pattern that we can easily check it
        logger->set_pattern("%v");

        // we expect there to be exactly 1 sink
        REQUIRE(logger->sinks().size() == 1);

        auto msg = "This is a test"s;
        logger->info(msg);

        // Get string from buffer
        auto resultString = buffer.str();
        REQUIRE(resultString == (msg + '\n'));

        // reset buffer
        buffer = std::stringstream();

        msg = "This is a test"s;
        logger->debug(msg);

        // Get string from buffer
        resultString = buffer.str();
        REQUIRE(resultString == "");

        // reset buffer
        buffer = std::stringstream();

        msg = "This is a warning"s;
        logger->warn(msg);

        resultString = buffer.str();
        REQUIRE(resultString == (msg + '\n'));
    }

    GIVEN("A logger at debug level")
    {
        // reset buffer
        buffer = std::stringstream();

        Logger::setLevel(Logger::LogLevel::DEBUG);
        auto logger = Logger::get("logger");

        // Set pattern that we can easily check it
        logger->set_pattern("%v");

        // we expect there to be exactly 1 sink
        REQUIRE(logger->sinks().size() == 1);

        auto msg = "This is a test"s;
        logger->info(msg);

        // Get string from buffer
        auto resultString = buffer.str();
        REQUIRE(resultString == (msg + '\n'));

        // reset buffer
        buffer = std::stringstream();

        msg = "This is a debug"s;
        logger->debug(msg);

        // Get string from buffer
        resultString = buffer.str();
        REQUIRE(resultString == (msg + '\n'));

        // reset buffer
        buffer = std::stringstream();

        msg = "This is a warning"s;
        logger->warn(msg);

        resultString = buffer.str();
        REQUIRE(resultString == (msg + '\n'));
    }
}

TEST_SUITE_END();
