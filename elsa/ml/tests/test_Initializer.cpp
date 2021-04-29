#include <catch2/catch.hpp>
#include "Initializer.h"

using namespace elsa;
using namespace elsa::ml;

TEST_CASE("Initializer", "[ml]")
{
    using InitializerImpl = detail::InitializerImpl<real_t>;

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
