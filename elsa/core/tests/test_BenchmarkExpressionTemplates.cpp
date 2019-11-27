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

using namespace elsa;
static const index_t dimension = 256;

TEST_CASE("Expression benchmark using Eigen with n=" + std::to_string(dimension) + "^3")
{
    index_t size = dimension * dimension * dimension;

    Eigen::Matrix<float, Eigen::Dynamic, 1> randVec(size);
    Eigen::Matrix<float, Eigen::Dynamic, 1> randVec2(size);
    Eigen::Matrix<float, Eigen::Dynamic, 1> randVec3(size);

    Eigen::Matrix<float, Eigen::Dynamic, 1> result(size);

    for (index_t i = 0; i < size; ++i) {
        randVec[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
        randVec2[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
        randVec3[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
    }

    BENCHMARK("exp = dc - dc2;") {
        result = randVec - randVec2;
    };

    BENCHMARK("exp = dc - dc2 + dc;") {
        result = randVec - randVec2 + randVec;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3;")
    {
        result = (randVec.array() * randVec2.array()).matrix()
                 - (randVec2.array() / randVec3.array()).matrix();
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3 + dc;")
    {
    result = (randVec.array() * randVec2.array()).matrix()
             - (randVec2.array() / randVec3.array()).matrix() + randVec;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3 + dc * dc3;")
    {
    result = (randVec.array() * randVec2.array()).matrix()
             - (randVec2.array() / randVec3.array()).matrix()
             + (randVec.array() * randVec3.array()).matrix();
    };
}

TEST_CASE("Expression benchmark using expression templates with n=" + std::to_string(dimension)
          + "^3")
{
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
    };

    BENCHMARK("exp = dc - dc2 + dc;")
    {
        result = dc - dc2 + dc;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3;")
    {
        result = dc * dc2 - dc2 / dc3;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3 + dc;")
    {
        result = dc * dc2 - dc2 / dc3 + dc;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3 + dc * dc3;")
    {
        result = dc * dc2 - dc2 / dc3 + dc * dc3;
    };
}

TEST_CASE("Expression benchmark without expression templates with n=" + std::to_string(dimension)
          + "^3")
{
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
        result = dc - dc2;
    };

    BENCHMARK("exp = dc - dc2 + dc;")
    {
        result = dc - dc2 + dc;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3;")
    {
        result = dc * dc2 - dc2 / dc3;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3 + dc;")
    {
        result = dc * dc2 - dc2 / dc3 + dc;
    };

    BENCHMARK("exp =  dc * dc2 - dc2 / dc3 + dc * dc3;")
    {
        result = dc * dc2 - dc2 / dc3 + dc * dc3;
    };
}

TEST_CASE("Expression benchmark openmp n=" + std::to_string(dimension)
          + "^3")
{
    IndexVector_t numCoeff(3);
    numCoeff << dimension, dimension, dimension;
    DataDescriptor desc(numCoeff);
    DataContainer dc(desc);

    for (index_t i = 0; i < dc.getSize(); ++i) {
        dc[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
    }

    BENCHMARK("for loop")
    {
        auto result = dc.test();
        return result;
    };

    BENCHMARK("for loop with omp")
    {
        auto result = dc.test_omp();
        return result;
    };

    BENCHMARK("s for loop")
    {
        auto result = dc.test_s();
        return result;
    };

    BENCHMARK("s for loop with omp")
    {
        auto result = dc.test_s_omp();
        return result;
    };
}
