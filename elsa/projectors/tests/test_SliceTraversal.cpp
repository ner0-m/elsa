
#include "doctest/doctest.h"

#include "SliceTraversal.h"
#include "Intersection.h"

#include <deque>

using namespace elsa;
using namespace doctest;

#include "PrettyPrint/Eigen.h"
#include "PrettyPrint/Stl.h"

TEST_SUITE_BEGIN("projectors");

Eigen::IOFormat vecfmt(10, 0, ", ", ", ", "", "", "[", "]");
Eigen::IOFormat matfmt(10, 0, ", ", "\n", "\t\t[", "]");

void checkTransformation(const RealRay_t& ray, const RealVector_t& centerOfRotation,
                         RealVector_t expectedro, RealVector_t expectedrd,
                         RealMatrix_t expectedRotation)
{
    TransformToTraversal transformation(ray, centerOfRotation);

    const RealVector_t ro = ray.origin();
    const RealVector_t rd = ray.direction();

    THEN("The rotation matrix is the same as the linear part of the transformation")
    {
        CHECK_EQ(transformation.rotation(), transformation.linear());
    }

    THEN("The rotation part of the transformation matrix is correct")
    {
        const RealMatrix_t matrix = transformation.rotation();

        INFO("R :=\n", matrix.format(matfmt));
        INFO("Expected R :=\n", expectedRotation.format(matfmt));

        CHECK_EQ(matrix, expectedRotation);
    }

    THEN("The ray direction is transformed correctly")
    {
        const RealVector_t transformed = transformation * Vec(rd);

        INFO("R := \n", transformation.linear().format(matfmt));
        INFO("R * rd := ", transformed.format(vecfmt));

        INFO("Expected R * rd := ", expectedrd.format(vecfmt));

        CHECK_UNARY(transformed.isApprox(expectedrd));
    }

    THEN("The ray origin is transformed correctly")
    {
        const RealVector_t transformed = transformation * Point(ro);

        INFO("T :=\n", transformation.transformation().format(matfmt));
        INFO("T * ro := ", transformed.format(vecfmt));

        INFO("Expected T * ro := ", expectedro.format(vecfmt));

        CHECK_UNARY(transformed.isApprox(expectedro));
    }
}
void checkTransformationBasic(const RealRay_t& ray, const RealVector_t& centerOfRotation,
                              RealMatrix_t rotation)
{
    checkTransformation(ray, centerOfRotation, RealVector_t({{-4, 0}}), RealVector_t({{1, 0}}),
                        rotation);
}

TEST_CASE("TraversalTransformation: Test transformation with ray going in +x")
{
    auto rotation0 = Eigen::Rotation2D<real_t>(0).matrix();
    const RealVector_t centerOfRotation({{1.5, 1.5}});

    GIVEN("A ray with largest component in +x direction")
    {
        const RealVector_t ro({{-2.5, 1.5}});
        INFO("ro := ", ro.format(vecfmt));

        const RealVector_t rd({{1, 0}});
        INFO("rd := ", rd.format(vecfmt));

        const RealRay_t ray(ro, rd);

        checkTransformationBasic(ray, centerOfRotation, rotation0);
    }
}

TEST_CASE("TraversalTransformation: Test transformation with ray going in -x")
{
    auto rotation180 = Eigen::Rotation2D<real_t>(pi_t).matrix();
    const RealVector_t centerOfRotation({{1.5, 1.5}});

    GIVEN("A ray with largest component in -x direction")
    {
        const RealVector_t ro({{5.5, 1.5}});
        INFO("ro := ", ro.format(vecfmt));

        const RealVector_t rd({{-1, 0}});
        INFO("rd := ", rd.format(vecfmt));

        const RealRay_t ray(ro, rd);

        checkTransformationBasic(ray, centerOfRotation, rotation180);
    }
}

TEST_CASE("TraversalTransformation: Test transformation with ray going in +y")
{
    auto rotation270 = Eigen::Rotation2D<real_t>(3 * pi_t / 2).matrix();
    const RealVector_t centerOfRotation({{1.5, 1.5}});

    GIVEN("A ray with largest component in +y direction")
    {
        const RealVector_t ro({{1.5, -2.5}});
        INFO("ro := ", ro.format(vecfmt));

        const RealVector_t rd({{0, 1}});
        INFO("rd := ", rd.format(vecfmt));

        const RealRay_t ray(ro, rd);

        checkTransformationBasic(ray, centerOfRotation, rotation270);
    }
}

TEST_CASE("TraversalTransformation: Test transformation with ray going in -y")
{
    auto rotation90 = Eigen::Rotation2D<real_t>(pi_t / 2).matrix();
    const RealVector_t centerOfRotation({{1.5, 1.5}});

    GIVEN("A ray with largest component in -y direction")
    {
        RealVector_t ro({{1.5, 5.5}});
        INFO("ro := ", ro.format(vecfmt));

        RealVector_t rd({{0, -1}});
        INFO("rd := ", rd.format(vecfmt));

        RealRay_t ray(ro, rd);

        checkTransformationBasic(ray, centerOfRotation, rotation90);
    }
}

TEST_CASE("TraversalTransformation: Test transformation with ray direction [1, 1]")
{
    auto rotation0 = Eigen::Rotation2D<real_t>(0).matrix();
    const RealVector_t centerOfRotation({{1.5, 1.5}});

    GIVEN("A ray with equally large positive component")
    {
        RealVector_t ro({{-2.5, 1.5}});
        INFO("ro := ", ro.format(vecfmt));

        RealVector_t rd({{1, 1}});
        INFO("rd := ", rd.format(vecfmt));

        RealRay_t ray(ro, rd);

        checkTransformation(ray, centerOfRotation, RealVector_t({{-4, 0}}), RealVector_t({{1, 1}}),
                            rotation0);
    }
}

TEST_CASE("TraversalTransformation: Test transformation with ray direction [1, -1]")
{
    auto rotation0 = Eigen::Rotation2D<real_t>(0).matrix();
    const RealVector_t centerOfRotation({{1.5, 1.5}});

    GIVEN("A ray with equally large component, where only x is positive")
    {
        RealVector_t ro({{-2.5, 1.5}});
        INFO("ro := ", ro.format(vecfmt));

        RealVector_t rd({{1, -1}});
        INFO("rd := ", rd.format(vecfmt));

        RealRay_t ray(ro, rd);

        checkTransformation(ray, centerOfRotation, RealVector_t({{-4, 0}}), RealVector_t({{1, -1}}),
                            rotation0);
    }
}

TEST_CASE("TraversalTransformation: Test transformation with ray direction [-1, 1]")
{
    auto rotation180 = Eigen::Rotation2D<real_t>(pi_t).matrix();
    const RealVector_t centerOfRotation({{1.5, 1.5}});

    GIVEN("A ray with equally large component, where x is negative")
    {
        RealVector_t ro({{5.5, 1.5}});
        INFO("ro := ", ro.format(vecfmt));

        RealVector_t rd({{-1, 1}});
        INFO("rd := ", rd.format(vecfmt));

        RealRay_t ray(ro, rd);

        checkTransformation(ray, centerOfRotation, RealVector_t({{-4, 0}}), RealVector_t({{1, -1}}),
                            rotation180);
    }
}

TEST_CASE("TraversalTransformation: Test transformation with ray direction [-1, -1]")
{
    auto rotation180 = Eigen::Rotation2D<real_t>(pi_t).matrix();
    const RealVector_t centerOfRotation({{1.5, 1.5}});

    GIVEN("A ray with equally large negative component")
    {
        RealVector_t ro({{5.5, 1.5}});
        INFO("ro := ", ro.format(vecfmt));

        RealVector_t rd({{-1, -1}});
        INFO("rd := ", rd.format(vecfmt));

        RealRay_t ray(ro, rd);

        checkTransformation(ray, centerOfRotation, RealVector_t({{-4, 0}}), RealVector_t({{1, 1}}),
                            rotation180);
    }
}

void checkBoundingBox(BoundingBox aabb, RealVector_t expectedMin, RealVector_t expectedMax)
{
    CAPTURE(aabb);

    THEN("_min is as expected")
    {
        INFO("_min := ", aabb._min.format(vecfmt));
        INFO("expected _min := ", expectedMin.format(vecfmt));
        CHECK_UNARY(aabb._min.isApprox(expectedMin));
    }

    THEN("_max is as expected")
    {
        INFO("_max := ", aabb._max.format(vecfmt));
        INFO("expected _max := ", expectedMax.format(vecfmt));
        CHECK_UNARY(aabb._max.isApprox(expectedMax));
    }
}

TEST_CASE("TraversalTransformation: Transform square bounding box")
{
    BoundingBox aabb(IndexVector_t{{5, 5}});
    CAPTURE(aabb);

    const RealVector_t expectedMin({{-2.5, -2.5}});
    const RealVector_t expectedMax({{2.5, 2.5}});

    GIVEN("A ray through the center of the bounding box point in +x direction")
    {
        const RealVector_t ro({{-1.5, 2.5}});
        const RealVector_t rd({{1, 0}});

        const RealRay_t ray(ro, rd);

        TransformToTraversal transformation(ray, aabb.center());

        WHEN("Transforming the bounding box")
        {
            auto rotatedAABB = transformation.toTraversalCoordinates(aabb);
            checkBoundingBox(rotatedAABB, expectedMin, expectedMax);
        }
    }

    GIVEN("A ray through the center of the bounding box point in -x direction")
    {
        const RealVector_t ro({{6.5, 2.5}});
        const RealVector_t rd({{-1, 0}});

        const RealRay_t ray(ro, rd);

        TransformToTraversal transformation(ray, aabb.center());

        WHEN("Transforming the bounding box")
        {
            auto rotatedAABB = transformation.toTraversalCoordinates(aabb);
            checkBoundingBox(rotatedAABB, expectedMin, expectedMax);
        }
    }

    GIVEN("A ray through the center of the bounding box point in +y direction")
    {
        const RealVector_t ro({{2.5, -1.5}});
        const RealVector_t rd({{0, 1}});

        const RealRay_t ray(ro, rd);

        TransformToTraversal transformation(ray, aabb.center());

        WHEN("Transforming the bounding box")
        {
            auto rotatedAABB = transformation.toTraversalCoordinates(aabb);
            checkBoundingBox(rotatedAABB, expectedMin, expectedMax);
        }
    }

    GIVEN("A ray through the center of the bounding box point in -y direction")
    {
        const RealVector_t ro({{2.5, 6.5}});
        const RealVector_t rd({{0, -1}});

        const RealRay_t ray(ro, rd);

        TransformToTraversal transformation(ray, aabb.center());

        WHEN("Transforming the bounding box")
        {
            auto rotatedAABB = transformation.toTraversalCoordinates(aabb);
            checkBoundingBox(rotatedAABB, expectedMin, expectedMax);
        }
    }
}

TEST_CASE("TraversalTransformation: Transform non-square bounding box")
{
    Eigen::IOFormat fmt(4, 0, ", ", "\n", "\t\t[", "]");

    BoundingBox aabb(IndexVector_t{{8, 5}});
    CAPTURE(aabb);

    GIVEN("A ray through the center of the bounding box point in +x direction")
    {
        const RealVector_t expectedMin({{-4, -2.5}});
        const RealVector_t expectedMax({{4, 2.5}});

        const RealVector_t ro({{-1.5, 2.5}});
        const RealVector_t rd({{1, 0}});

        const RealRay_t ray(ro, rd);

        TransformToTraversal transformation(ray, aabb.center());

        WHEN("Transforming the bounding box")
        {
            auto rotatedAABB = transformation.toTraversalCoordinates(aabb);
            checkBoundingBox(rotatedAABB, expectedMin, expectedMax);
        }
    }

    const RealVector_t expectedMin({{-2.5, -4}});
    const RealVector_t expectedMax({{2.5, 4}});

    GIVEN("A ray through the center of the bounding box point in +y direction")
    {

        const RealVector_t ro({{4, -1.5}});
        const RealVector_t rd({{0, 1}});

        const RealRay_t ray(ro, rd);

        TransformToTraversal transformation(ray, aabb.center());

        WHEN("Transforming the bounding box")
        {
            auto rotatedAABB = transformation.toTraversalCoordinates(aabb);
            checkBoundingBox(rotatedAABB, expectedMin, expectedMax);
        }
    }

    GIVEN("A ray through the center of the bounding box point in -y direction")
    {
        const RealVector_t ro({{4, 6.5}});
        const RealVector_t rd({{0, -1}});

        const RealRay_t ray(ro, rd);

        TransformToTraversal transformation(ray, aabb.center());

        WHEN("Transforming the bounding box")
        {
            auto rotatedAABB = transformation.toTraversalCoordinates(aabb);
            checkBoundingBox(rotatedAABB, expectedMin, expectedMax);
        }
    }
}

index_t checkTraversal(BoundingBox aabb, RealRay_t ray, std::deque<RealVector_t> visitedVoxels)
{
    Eigen::IOFormat fmt(10, 0, ", ", ", ", "", "", "[", "]");

    SliceTraversal traversal(aabb, ray);

    INFO("rd := ", ray.direction().format(vecfmt));
    INFO("ro := ", ray.origin().format(vecfmt));

    CAPTURE(traversal.startIndex());
    CAPTURE(traversal.endIndex());
    CHECK_EQ(traversal.endIndex(), visitedVoxels.size());
    // INFO("entryPoint := ", traversal.entryPoint_.format(fmt));
    // INFO("exitPoint := ", traversal.exitPoint_.format(fmt));

    index_t counter = 0;
    for (auto iter = traversal.begin(); iter != traversal.end(); ++iter) {
        auto value = *iter;

        CAPTURE(counter);
        REQUIRE_MESSAGE(!visitedVoxels.empty(), "Visiting more voxels than expected");

        RealVector_t expected = visitedVoxels.front();
        RealVector_t point = ray.pointAt(value.t_);

        CAPTURE(value.t_);
        INFO("RealRay_t hit: ", point.format(fmt));
        INFO("Should hit: ", expected.format(fmt));

        CHECK_UNARY(point.isApprox(expected));

        // Pop the first element, as we don't need it anymore
        visitedVoxels.pop_front();

        // increment counter
        ++counter;
    }

    INFO("Voxels left in list: ", visitedVoxels.size());

    REQUIRE_MESSAGE(visitedVoxels.empty(), "Voxel list is not empty, so we've visited too few");
    CHECK_NE(counter, 0);
    // CHECK(false);

    return counter;
}

TEST_CASE("SliceTraversal: Traversing a 2D grid parallel to x-axis, dir [1, 0]")
{

    IndexVector_t size({{3, 3}});

    BoundingBox aabb(size);

    const RealVector_t rd({{1, 0}});
    const real_t x = -1.5f;

    for (real_t i = 0; i < 3; ++i) {
        const real_t y = 0.5f + i;
        const RealVector_t ro({{x, y}});
        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{0.5, y}});
        visitedVoxels.emplace_back(RealVector_t{{1.5, y}});
        visitedVoxels.emplace_back(RealVector_t{{2.5, y}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly 3 voxels are visited") { CHECK_EQ(counter, 3); }
    }
}

TEST_CASE("SliceTraversal: Traversing a 2D grid parallel to x-axis, dir [-1, 0]")
{
    IndexVector_t size({{3, 3}});

    BoundingBox aabb(size);

    const RealVector_t rd({{-1, 0}});
    const real_t x = 5.5f;

    for (real_t i = 1; i < 3; ++i) {
        const real_t y = 0.5f + i;
        const RealVector_t ro({{x, y}});
        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{2.5, y}});
        visitedVoxels.emplace_back(RealVector_t{{1.5, y}});
        visitedVoxels.emplace_back(RealVector_t{{0.5, y}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly 3 voxels are visited") { CHECK_EQ(counter, 3); }
    }
}

TEST_CASE("SliceTraversal: Traversing a 2D grid parallel to y-axis, dir [0, 1]")
{
    IndexVector_t size({{3, 3}});

    BoundingBox aabb(size);

    const RealVector_t rd({{0, 1}});
    const real_t y = -1.5;

    for (real_t i = 1; i < 3; ++i) {
        const real_t x = 0.5f + i;
        const RealVector_t ro({{x, y}});
        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{x, 0.5}});
        visitedVoxels.emplace_back(RealVector_t{{x, 1.5}});
        visitedVoxels.emplace_back(RealVector_t{{x, 2.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly 3 voxels are visited") { CHECK_EQ(counter, 3); }
    }
}

TEST_CASE("SliceTraversal: Traversing a 2D grid parallel to y-axis, dir [0, -1]")
{
    IndexVector_t size({{3, 3}});

    BoundingBox aabb(size);

    const RealVector_t rd({{0, -1}});
    const real_t y = 5.5;

    for (real_t i = 1; i < 3; ++i) {
        const real_t x = 0.5f + i;
        const RealVector_t ro({{x, y}});
        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{x, 2.5}});
        visitedVoxels.emplace_back(RealVector_t{{x, 1.5}});
        visitedVoxels.emplace_back(RealVector_t{{x, 0.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly 3 voxels are visited") { CHECK_EQ(counter, 3); }
    }
}

TEST_CASE("SliceTraversal: Traversing a 2D grid diagonally, dir [1, 1]")
{
    IndexVector_t size({{3, 3}});

    BoundingBox aabb(size);

    RealVector_t rd({{1, 1}});
    rd.normalize();

    WHEN("Traversing the grid through the center")
    {
        const RealVector_t ro({{-1.5, -1.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{0.5, 0.5}});
        visitedVoxels.emplace_back(RealVector_t{{1.5, 1.5}});
        visitedVoxels.emplace_back(RealVector_t{{2.5, 2.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 3); }
    }

    WHEN("Traversing the grid through the left middle and top middle voxel")
    {
        const RealVector_t ro({{-0.5, 0.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{0.5, 1.5}});
        visitedVoxels.emplace_back(RealVector_t{{1.5, 2.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 2); }
    }

    WHEN("Traversing the grid through the right middle and bottom middle voxel")
    {
        Eigen::IOFormat fmtvec(10, 0, ", ", ", ", "", "", "[", "]");

        const RealVector_t ro({{0.5, -0.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.push_back(RealVector_t{{1.5, 0.5}});
        visitedVoxels.push_back(RealVector_t{{2.5, 1.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 2); }
    }

    WHEN("Traversing the grid through the bottom right corner")
    {
        const RealVector_t ro({{2.5 - 1, 0.5 - 1}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{2.5, 0.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 1); }
    }

    WHEN("Traversing the grid through the top left corner")
    {
        const RealVector_t ro({{-0.5, 1.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{0.5, 2.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 1); }
    }
}

TEST_CASE("SliceTraversal: Traversing a 2D grid diagonally, dir [1, -1]")
{
    IndexVector_t size({{3, 3}});

    BoundingBox aabb(size);

    RealVector_t rd({{1, -1}});
    rd.normalize();

    WHEN("Traversing the bottom left corner")
    {
        const RealVector_t ro({{-0.5, 1.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{0.5, 0.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly one step is taken") { CHECK_EQ(counter, 1); }
    }

    WHEN("Traversing the grid through the left middle and top middle voxel")
    {
        const RealVector_t ro({{-0.5, 2.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{0.5, 1.5}});
        visitedVoxels.emplace_back(RealVector_t{{1.5, 0.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 2); }
    }

    WHEN("Traversing the grid through the center diagonally")
    {
        const RealVector_t ro({{-0.5, 3.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{0.5, 2.5}});
        visitedVoxels.emplace_back(RealVector_t{{1.5, 1.5}});
        visitedVoxels.emplace_back(RealVector_t{{2.5, 0.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly three steps are taken") { CHECK_EQ(counter, 3); }
    }

    WHEN("Traversing the grid through the top middle and right middle voxel")
    {
        const RealVector_t ro({{0.5, 3.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{1.5, 2.5}});
        visitedVoxels.emplace_back(RealVector_t{{2.5, 1.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 2); }
    }

    WHEN("Traversing the grid through the top right voxel")
    {
        const RealVector_t ro({{1.5, 3.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{2.5, 2.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 1); }
    }
}

TEST_CASE("SliceTraversal: Traversing a 2D grid diagonally, dir [-1, 1]")
{
    IndexVector_t size({{3, 3}});

    BoundingBox aabb(size);

    RealVector_t rd({{-1, 1}});
    rd.normalize();

    WHEN("Traversing the top right corner")
    {
        const RealVector_t ro({{3.5, 1.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{2.5, 2.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly one step is taken") { CHECK_EQ(counter, 1); }
    }

    WHEN("Traversing the grid through the top center and right center voxel")
    {
        const RealVector_t ro({{3.5, 0.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{2.5, 1.5}});
        visitedVoxels.emplace_back(RealVector_t{{1.5, 2.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 2); }
    }

    WHEN("Traversing the grid through the center diagonally")
    {
        const RealVector_t ro({{3.5, -0.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{2.5, 0.5}});
        visitedVoxels.emplace_back(RealVector_t{{1.5, 1.5}});
        visitedVoxels.emplace_back(RealVector_t{{0.5, 2.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly three steps are taken") { CHECK_EQ(counter, 3); }
    }

    WHEN("Traversing the grid through the left center and bottom center voxel")
    {
        const RealVector_t ro({{2.5, -0.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{1.5, 0.5}});
        visitedVoxels.emplace_back(RealVector_t{{0.5, 1.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 2); }
    }

    WHEN("Traversing the grid through the bottom left voxel")
    {
        const RealVector_t ro({{1.5, -0.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{0.5, 0.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 1); }
    }
}

TEST_CASE("SliceTraversal: Traversing a 2D grid diagonally, dir [-1, -1]")
{
    IndexVector_t size({{3, 3}});

    BoundingBox aabb(size);

    RealVector_t rd({{-1, -1}});
    rd.normalize();

    CAPTURE(aabb);

    WHEN("Traversing the grid through the center of the top left corner")
    {
        const RealVector_t ro({{1.5, 3.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{0.5, 2.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly one step is taken") { CHECK_EQ(counter, 1); }
    }

    WHEN("Traversing the grid through the top left voxel, but not centered")
    {
        const RealVector_t ro({{1.75, 3.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{0.5, 2.25}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 1); }
    }

    WHEN("Traversing the grid through the left and top center voxel")
    {
        const RealVector_t ro({{2.5, 3.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{1.5, 2.5}});
        visitedVoxels.emplace_back(RealVector_t{{0.5, 1.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 2); }
    }

    WHEN("Traversing the grid through the left and top center voxel but not centered")
    {
        const RealVector_t ro({{2, 3.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{1.5, 3}});
        visitedVoxels.emplace_back(RealVector_t{{0.5, 2}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 2); }
    }

    WHEN("Traversing the grid slightly above the volume center diagonally")
    {
        const RealVector_t ro({{3.25, 3.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{2.5, 2.75}});
        visitedVoxels.emplace_back(RealVector_t{{1.5, 1.75}});
        visitedVoxels.emplace_back(RealVector_t{{0.5, 0.75}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly three steps are taken") { CHECK_EQ(counter, 3); }
    }

    WHEN("Traversing the grid through the volume center diagonally")
    {
        const RealVector_t ro({{3.5, 3.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{2.5, 2.5}});
        visitedVoxels.emplace_back(RealVector_t{{1.5, 1.5}});
        visitedVoxels.emplace_back(RealVector_t{{0.5, 0.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly three steps are taken") { CHECK_EQ(counter, 3); }
    }

    WHEN("Traversing the grid slightly below the volume center diagonally")
    {
        const RealVector_t ro({{3.75, 3.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{2.5, 2.25}});
        visitedVoxels.emplace_back(RealVector_t{{1.5, 1.25}});
        visitedVoxels.emplace_back(RealVector_t{{0.5, 0.25}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly three steps are taken") { CHECK_EQ(counter, 3); }
    }

    WHEN("Traversing the grid through the bottom and right center voxel")
    {
        const RealVector_t ro({{3.5, 2.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{2.5, 1.5}});
        visitedVoxels.emplace_back(RealVector_t{{1.5, 0.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 2); }
    }

    WHEN("Traversing the grid through the bottom left corner")
    {
        const RealVector_t ro({{3.5, 1.5}});

        const RealRay_t ray(ro, rd);

        // list of points we expect to visit
        std::deque<RealVector_t> visitedVoxels;
        visitedVoxels.emplace_back(RealVector_t{{2.5, 0.5}});

        auto counter = checkTraversal(aabb, ray, visitedVoxels);

        THEN("Exactly two steps are taken") { CHECK_EQ(counter, 1); }
    }
}

TEST_SUITE_END();
