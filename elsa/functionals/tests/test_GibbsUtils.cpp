/**
 * @file GibbsUtils.cpp
 *
 * @brief Tests for the GibbsUtils helper functions
 *
 */

#include <doctest/doctest.h>

#include "testHelpers.h"
#include "Gibbs/GibbsUtils.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"
#include <algorithm>
#include <cstdlib>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("gibbsutils");

TEST_CASE("Gibbs Utils: testing generate neighbour shifts")
{
    std::vector<IndexVector_t> shifts2d = elsa::Gibbs::generateNeighbourShift(2);
    std::vector<IndexVector_t> shifts3d = elsa::Gibbs::generateNeighbourShift(3);

    CHECK(shifts2d.size() == 9);
    CHECK(shifts3d.size() == 27);

    for (int i = 0; i < 9; i++) {
        IndexVector_t numCoeff(2);
        numCoeff << rand() % 3 - 1, rand() % 3 - 1;
        if (std::find(shifts2d.begin(), shifts2d.end(), numCoeff) == shifts2d.end())
            MESSAGE("missed shift: ", numCoeff[0], numCoeff[1]);
        CHECK(std::find(shifts2d.begin(), shifts2d.end(), numCoeff) != shifts2d.end());
    }

    for (int i = 0; i < 27; i++) {
        IndexVector_t numCoeff(3);
        numCoeff << rand() % 3 - 1, rand() % 3 - 1, rand() % 3 - 1;

        CHECK(std::find(shifts3d.begin(), shifts3d.end(), numCoeff) != shifts3d.end());
    }
};

TEST_CASE("Gibbs Utils: testing neighbours data")
{
    IndexVector_t sizeData(2);
    sizeData << 3, 3;
    VolumeDescriptor plainDescriptor{sizeData};

    Eigen::VectorXf vec(3 * 3);
    vec << 1, 2, 3, 10, 20, 30, 100, 200, 300;

    DataContainer<float> input(plainDescriptor, vec);

    IndexVector_t point(2);

    point << 0, 0;
    std::vector<float> extractedData = elsa::Gibbs::allNeighboursData(input, point);
    CHECK(std::accumulate(extractedData.begin(), extractedData.end(), 0) == 33);

    point[0] = 2;
    point[1] = 2;
    extractedData = elsa::Gibbs::allNeighboursData(input, point);
    CHECK(std::accumulate(extractedData.begin(), extractedData.end(), 0) == 550);

    point[0] = 0;
    point[1] = 2;
    extractedData = elsa::Gibbs::allNeighboursData(input, point);
    CHECK(std::accumulate(extractedData.begin(), extractedData.end(), 0) == 330);

    point[0] = 2;
    point[1] = 0;
    extractedData = elsa::Gibbs::allNeighboursData(input, point);
    CHECK(std::accumulate(extractedData.begin(), extractedData.end(), 0) == 55);
};

TEST_CASE("Gibbs Utils: testing neighbours sum")
{
    IndexVector_t sizeData(2);
    sizeData << 3, 3;
    VolumeDescriptor plainDescriptor{sizeData};

    Eigen::VectorXf vec(3 * 3);
    vec << 1, 2, 3, 10, 20, 30, 100, 200, 300;

    DataContainer<float> input(plainDescriptor, vec);

    IndexVector_t point(2);
    point << 0, 0;

    std::function<float(float, float)> transform = [](float point, float source) { return point; };
    std::function<float(index_t)> coef = [](index_t distance) {
        return 1.0 / (1 + distance * distance);
    };
    float transformSum = elsa::Gibbs::allNeighboursSum(input, point, transform, coef);

    CHECK(transformSum == 11.0);
};

TEST_SUITE_END();
