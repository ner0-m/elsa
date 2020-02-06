/**
 * \file test_Logger.cpp
 *
 * \brief Tests for the Logger class
 *
 * \author Tobias Lasser - initial code
 */

#include <catch2/catch.hpp>
#include "Logger.h"
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>

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

        WHEN("Adding Custom Sink")
        {
            spdlog::sink_ptr custom_sink = std::make_shared<spdlog::sinks::stdout_color_sink_st>();
            Logger::addCustomSink(custom_sink, name);
            auto otherLogger = Logger::get("other");

            THEN("The sink is only added to one specific Logger")
            {
                REQUIRE(testLogger->sinks().size() == otherLogger->sinks().size() + 1);
            }

            THEN("We can add multiple sinks to one Logger")
            {
                int num_sinks = testLogger->sinks().size();
                spdlog::sink_ptr new_custom_sink =
                    std::make_shared<spdlog::sinks::stdout_color_sink_st>();
                Logger::addCustomSink(new_custom_sink, name);

                REQUIRE(testLogger->sinks().size() == num_sinks + 1);
            }

            THEN("actually logging works")
            {
                testLogger->info("This is another warning");
                REQUIRE(true);
            }
        }

        WHEN("Clearing the logger map")
        {
            Logger::clearMap();
            THEN("The size of the map is 0") { REQUIRE(Logger::isEmpty()); }
            THEN("One can insert new loggers with new sinks")
            {
                testLogger = Logger::get(name);
                REQUIRE(testLogger->sinks().size() == 2);
            }
        }
    }
}
