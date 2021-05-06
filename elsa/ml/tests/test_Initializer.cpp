#include "doctest/doctest.h"
#include "Initializer.h"

using namespace elsa;
using namespace elsa::ml;
using namespace doctest;

TEST_SUITE_BEGIN("ml");

// TODO(dfrank): remove and replace with proper doctest usage of test cases
#define SECTION(name) DOCTEST_SUBCASE(name)

TEST_CASE("Initializer")
{
    using InitializerImpl = elsa::ml::detail::InitializerImpl<real_t>;

    const index_t size = 1000;
    real_t* data = new real_t[size];

    SECTION("Zeros")
    {
        InitializerImpl::initialize(data, size, Initializer::Zeros);

        for (index_t i = 0; i < size; ++i) {
            REQUIRE(data[i] == Approx(0.f));
        }
    }

    SECTION("Ones")
    {
        InitializerImpl::initialize(data, size, Initializer::Ones);

        for (index_t i = 0; i < size; ++i) {
            REQUIRE(data[i] == Approx(1.f));
        }
    }
    delete[] data;
}

TEST_SUITE_END();
