
#include "doctest/doctest.h"
#include "Error.h"
#include <sstream>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

void throw_error()
{
    throw Error("I'm thrown from throw_error");
}

TEST_CASE("Error:")
{
    GIVEN("A constructed Error")
    {
        Error err("Here is my custom message");

        THEN("It's message is not empty")
        {
            const auto msg = std::string{err.what()};

            CHECK_UNARY_FALSE(msg.empty());
            CHECK_NE(msg[0], '\0');
        }

        THEN("It's streamable")
        {
            std::ostringstream stream;
            stream << err;

            const auto mes = stream.str();

            CHECK_UNARY_FALSE(mes.empty());
            CHECK_NE(mes[0], '\0');
        }
    }

    GIVEN("A thrown Error") { CHECK_THROWS_AS(throw_error(), Error); }
}

TEST_SUITE_END();
