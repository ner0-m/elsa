/**
 * @file test_RayGenerationBench.cpp
 *
 * @brief Benchmarks for projectors
 *
 * @author David Frank
 */

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>

#include "Logger.h"
#include "Intersection.h"

#include <iostream>

using namespace elsa;

using Ray = Eigen::ParametrizedLine<real_t, Eigen::Dynamic>;

static constexpr index_t NUM_RAYS = 100;

TEST_CASE("Benchmarking 2D Ray-AABB intersections")
{
    // Turn logger off
    Logger::setLevel(Logger::LogLevel::OFF);

    index_t dim = 2;

    IndexVector_t voxel(dim);
    voxel << 10, 10;

    BoundingBox aabb(voxel);

    std::vector<Ray> rays;
    rays.reserve(NUM_RAYS);

    // intersection from below
    RealVector_t ro(dim);
    ro << 5, -5;

    for (int i = 0; i < NUM_RAYS; ++i) {
        auto rd = RealVector_t::Random(dim).normalized();
        rays.emplace_back(ro, rd);
    }

    BENCHMARK("Intersection from below")
    {
        IntersectionResult result;
        for (auto& r : rays) {
            auto opt = Intersection::withRay(aabb, r);
            if (opt)
                result = *opt;
        }
        return result;
    };

    rays.clear();

    // intersection from above
    ro << 5, 15;

    for (int i = 0; i < NUM_RAYS; ++i) {
        auto rd = RealVector_t::Random(dim).normalized();
        rays.emplace_back(ro, rd);
    }

    BENCHMARK("Intersection from above")
    {
        IntersectionResult result;
        for (auto& r : rays) {
            auto opt = Intersection::withRay(aabb, r);
            if (opt)
                result = *opt;
        }
        return result;
    };

    rays.clear();

    // intersection from the left
    ro << -5, 5;

    for (int i = 0; i < NUM_RAYS; ++i) {
        auto rd = RealVector_t::Random(dim).normalized();
        rays.emplace_back(ro, rd);
    }

    BENCHMARK("Intersection from the left")
    {
        IntersectionResult result;
        for (auto& r : rays) {
            auto opt = Intersection::withRay(aabb, r);
            if (opt)
                result = *opt;
        }
        return result;
    };

    rays.clear();

    // intersection from the right
    ro << 15, 5;

    for (int i = 0; i < NUM_RAYS; ++i) {
        auto rd = RealVector_t::Random(dim).normalized();
        rays.emplace_back(ro, rd);
    }

    BENCHMARK("Intersection from the right")
    {
        IntersectionResult result;
        for (auto& r : rays) {
            auto opt = Intersection::withRay(aabb, r);
            if (opt)
                result = *opt;
        }
        return result;
    };
}

TEST_CASE("Benchmarking 3D intersections")
{
    // Turn logger off
    Logger::setLevel(Logger::LogLevel::OFF);

    index_t dim = 3;

    IndexVector_t voxel(dim);
    voxel << 10, 10, 10;

    BoundingBox aabb(voxel);

    std::vector<Ray> rays;
    rays.reserve(NUM_RAYS);

    RealVector_t ro(dim);
    ro << 5, 5, -5;

    for (int i = 0; i < NUM_RAYS; ++i) {
        auto rd = RealVector_t::Random(dim).normalized();
        rays.emplace_back(ro, rd);
    }

    BENCHMARK("Intersection from front")
    {
        IntersectionResult result;
        for (auto& r : rays) {
            auto opt = Intersection::withRay(aabb, r);
            if (opt)
                result = *opt;
        }
        return result;
    };

    rays.clear();
    ro << 5, 5, 15;

    for (int i = 0; i < NUM_RAYS; ++i) {
        auto rd = RealVector_t::Random(dim).normalized();
        rays.emplace_back(ro, rd);
    }

    BENCHMARK("Intersection from behind")
    {
        IntersectionResult result;
        for (auto& r : rays) {
            auto opt = Intersection::withRay(aabb, r);
            if (opt)
                result = *opt;
        }
        return result;
    };

    rays.clear();
    ro << 5, -5, 5;

    for (int i = 0; i < NUM_RAYS; ++i) {
        auto rd = RealVector_t::Random(dim).normalized();
        rays.emplace_back(ro, rd);
    }

    BENCHMARK("Intersection from below")
    {
        IntersectionResult result;
        for (auto& r : rays) {
            auto opt = Intersection::withRay(aabb, r);
            if (opt)
                result = *opt;
        }
        return result;
    };

    rays.clear();
    ro << 5, 15, 5;

    for (int i = 0; i < NUM_RAYS; ++i) {
        auto rd = RealVector_t::Random(dim).normalized();
        rays.emplace_back(ro, rd);
    }

    BENCHMARK("Intersection from above")
    {
        IntersectionResult result;
        for (auto& r : rays) {
            auto opt = Intersection::withRay(aabb, r);
            if (opt)
                result = *opt;
        }
        return result;
    };

    rays.clear();
    ro << -5, 5, 5;

    for (int i = 0; i < NUM_RAYS; ++i) {
        auto rd = RealVector_t::Random(dim).normalized();
        rays.emplace_back(ro, rd);
    }

    BENCHMARK("Intersection from the left")
    {
        IntersectionResult result;
        for (auto& r : rays) {
            auto opt = Intersection::withRay(aabb, r);
            if (opt)
                result = *opt;
        }
        return result;
    };

    rays.clear();
    ro << 15, 5, 5;

    for (int i = 0; i < NUM_RAYS; ++i) {
        auto rd = RealVector_t::Random(dim).normalized();
        rays.emplace_back(ro, rd);
    }

    BENCHMARK("Intersection from the right")
    {
        IntersectionResult result;
        for (auto& r : rays) {
            auto opt = Intersection::withRay(aabb, r);
            if (opt)
                result = *opt;
        }
        return result;
    };
}
