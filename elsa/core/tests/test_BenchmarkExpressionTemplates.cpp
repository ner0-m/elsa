/**
 * \file test_BenchmarkExpressionTemplates.cpp
 *
 * \brief Benchmarks for expression templates
 *
 * \author Jens Petit
 */

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>
#include "DataContainer.h"
#include <string>
#include <cstdlib>
#include <ctime>

using namespace elsa;
static const index_t dimension = 16;

TEST_CASE("Expression benchmark using Eigen with n=" + std::to_string(dimension) + "^3")
{
    index_t size = dimension * dimension * dimension;
    Eigen::VectorXf randVec = Eigen::VectorXf::Random(size);
    Eigen::VectorXf randVec2 = Eigen::VectorXf::Random(size);
    Eigen::VectorXf randVec3 = Eigen::VectorXf::Random(size);

    BENCHMARK("exp = dc - dc2;") { return (randVec - randVec2).eval(); };

    BENCHMARK("exp = dc - dc2 + dc;") { return (randVec - randVec2 + randVec).eval(); };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3;")
    {
        return (randVec * randVec2 - randVec2 / 2.38).eval();
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3 + dc;")
    {
        return (randVec * randVec2 - randVec2 / 2.38 + randVec).eval();
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3 + dc * dc3;")
    {
        return (randVec * randVec2 - randVec2 / 2.38 + randVec * randVec3).eval();
    };
}

TEST_CASE("Expression benchmark using expression templates with n=" + std::to_string(dimension)
          + "^3")
{
    srand(static_cast<unsigned>(time(nullptr)));

    IndexVector_t numCoeff(3);
    numCoeff << dimension, dimension, dimension;
    DataDescriptor desc(numCoeff);
    DataContainer dc(desc);
    DataContainer dc2(desc);
    DataContainer dc3(desc);
    DataContainer result(desc);

    for (index_t i = 0; i < dc.getSize(); ++i) {
        dc[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
        dc2[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
        dc3[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
    }

    BENCHMARK("exp = dc - dc2;")
    {
        result = dc - dc2;
        return result;
    };

    BENCHMARK("exp = dc - dc2 + dc;")
    {
        result = dc - dc2 + dc;
        return result;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3;")
    {
        result = dc * dc2 - dc2 / dc3;
        return result;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3 + dc;")
    {
        result = dc * dc2 - dc2 / dc3 + dc;
        return result;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3 + dc * dc3;")
    {
        result = dc * dc2 - dc2 / dc3 + dc * dc3;
        return result;
    };
}

TEST_CASE("Expression benchmark without expression templates with n=" + std::to_string(dimension)
          + "^3")
{
    srand(static_cast<unsigned>(time(nullptr)));

    IndexVector_t numCoeff(3);
    numCoeff << dimension, dimension, dimension;
    DataDescriptor desc(numCoeff);
    DataContainer dc(desc);
    DataContainer dc2(desc);
    DataContainer dc3(desc);
    DataContainer result(desc);

    for (index_t i = 0; i < dc.getSize(); ++i) {
        dc[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
        dc2[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
        dc3[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
    }

    // to avoid using expression templates
    using namespace elsa::detail;

    BENCHMARK("exp = dc - dc2;")
    {
        return dc - dc2;
    };

    BENCHMARK("exp = dc - dc2 + dc;")
    {
        return dc - dc2 + dc;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3;")
    {
        return dc * dc2 - dc2 / dc3;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3 + dc;")
    {
        return dc * dc2 - dc2 / dc3 + dc;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3 + dc * dc3;")
    {
        return dc * dc2 - dc2 / dc3 + dc * dc3;
    };
}
