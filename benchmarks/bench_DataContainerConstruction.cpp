/**
 * \file test_BenchmarkDataContainerConstruction.cpp
 *
 * \brief Benchmarks for constructing an empty DataContainer
 *
 * \author Jens Petit
 */

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>
#include "DataContainer.h"
#include "VolumeDescriptor.h"
#include <string>
#include <cstdlib>

using namespace elsa;

TEST_CASE("DataContainer construction benchmark")
{
    const index_t dim = 1024;
    IndexVector_t numCoeff(3);
    numCoeff << dim, dim, dim;
    VolumeDescriptor desc(numCoeff);

    BENCHMARK("only allocating")
    {
        DataContainer dc(desc);
        return dc;
    };

    BENCHMARK("allocating and assigning")
    {
        DataContainer dc(desc);
        return dc = 0;
    };

    BENCHMARK("Eigen directly only allocating ")
    {
        Eigen::Matrix<float, Eigen::Dynamic, 1> vec(dim * dim * dim);
        return vec;
    };

    BENCHMARK("Eigen directly allocating and assigning")
    {
        Eigen::Matrix<float, Eigen::Dynamic, 1> vec(dim * dim * dim);
        return vec.setZero();
    };
}
