/**
 * @file test_AB_GMRES.cpp
 *
 * @brief Tests for the AB_GMRES class
 *
 * @author Daniel Klitzner
 */

#include "doctest/doctest.h"

#include "AB_GMRES.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "JosephsMethod.h"
#include "CircleTrajectoryGenerator.h"
#include "Phantoms.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

template <template <typename> typename T, typename data_t>
constexpr data_t return_data_t(const T<data_t>&);

TYPE_TO_STRING(AB_GMRES<float>);
TYPE_TO_STRING(AB_GMRES<double>);

TEST_CASE("Clone and Comparison Operation")
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Phantom reconstruction problem")
    {

        IndexVector_t size(2);
        size << 16, 16;
        auto phantom = phantoms::modifiedSheppLogan(size);
        auto& volumeDescriptor = phantom.getDataDescriptor();

        index_t numAngles{30}, arc{180};
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, phantom.getDataDescriptor(), arc, static_cast<real_t>(size(0)) * 100.0f,
            static_cast<real_t>(size(0)));

        JosephsMethod projector(downcast<VolumeDescriptor>(volumeDescriptor), *sinoDescriptor);

        auto sinogram = projector.apply(phantom);

        // projector ABckprojector sinogram epsilon
        real_t epsilon = std::numeric_limits<real_t>::epsilon();

        WHEN("setting up a AB_GMRES solver")
        {
            AB_GMRES solver{projector, sinogram, epsilon};

            THEN("the clone works correctly")
            {
                auto gmresClone = solver.clone();

                REQUIRE_NE(gmresClone.get(), &solver);
                REQUIRE_EQ(*gmresClone, solver);
            }
        }
    }
}

TEST_CASE("JosephsMethod Phantom Reconstruction")
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Phantom reconstruction problem")
    {

        IndexVector_t size(2);
        size << 16, 16;
        auto phantom = phantoms::modifiedSheppLogan(size);
        auto& volumeDescriptor = phantom.getDataDescriptor();

        index_t numAngles{30}, arc{180};
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, phantom.getDataDescriptor(), arc, static_cast<real_t>(size(0)) * 100.0f,
            static_cast<real_t>(size(0)));

        JosephsMethod projector(downcast<VolumeDescriptor>(volumeDescriptor), *sinoDescriptor);

        auto sinogram = projector.apply(phantom);

        // projector ABckprojector sinogram epsilon
        real_t epsilon = std::numeric_limits<real_t>::epsilon();

        WHEN("setting up a AB_GMRES solver")
        {
            AB_GMRES solver{projector, sinogram, epsilon};

            THEN("Solving for 10 Iterations")
            {
                auto reconstruction = solver.solve(10);

                AND_THEN("Reconstruction works as expected")
                {

                    DataContainer resultsDifference = reconstruction - phantom;

                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * phantom.squaredL2Norm(), 0.1));
                }
            }
        }
    }
}

TEST_CASE("Comparison of restarted and non-restarted GMRES")
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Phantom reconstruction problem")
    {

        IndexVector_t size(2);
        size << 16, 16;
        auto phantom = phantoms::modifiedSheppLogan(size);
        auto& volumeDescriptor = phantom.getDataDescriptor();

        index_t numAngles{30}, arc{180};
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, phantom.getDataDescriptor(), arc, static_cast<real_t>(size(0)) * 100.0f,
            static_cast<real_t>(size(0)));

        JosephsMethod projector(downcast<VolumeDescriptor>(volumeDescriptor), *sinoDescriptor);

        auto sinogram = projector.apply(phantom);

        // projector ABckprojector sinogram epsilon
        real_t epsilon = std::numeric_limits<real_t>::epsilon();

        WHEN("setting up AB_GMRES solvers")
        {
            AB_GMRES solver{projector, sinogram, epsilon};

            THEN("Solving for 12 Iterations and 3 Iterations with 4 Restarts respectively")
            {
                auto reconstruction = solver.solve(12);
                auto reconstructionRestarted = solver.solveAndRestart(4, 3);

                AND_THEN("Reconstruction works as expected")
                {
                    DataContainer resultsDifference = reconstruction - phantom;

                    DataContainer resultsDifferenceRestarted = reconstructionRestarted - phantom;

                    // Checking the Difference between restarted/non-restarted version and phantom

                    REQUIRE_UNARY(checkApproxEq(resultsDifference.squaredL2Norm(),
                                                epsilon * epsilon * phantom.squaredL2Norm(), 0.1));

                    REQUIRE_UNARY(checkApproxEq(resultsDifferenceRestarted.squaredL2Norm(),
                                                epsilon * epsilon * phantom.squaredL2Norm(), 0.1));

                    // Checking difference between the two versions themselves

                    REQUIRE_UNARY(checkApproxEq(resultsDifferenceRestarted.squaredL2Norm(),
                                                resultsDifference.squaredL2Norm(), 0.1));
                }
            }
        }
    }
}

/// --- read these test cases if elsa unified-linalg is merged ---

/*TEST_CASE("Clone and Comparison Operation elsa::linalg")
{
   Logger::setLevel(Logger::LogLevel::OFF);

   GIVEN("a 2x2 Matrix A,a 2x1 Vector b and a 2x1 Vector x0")
   {

       linalg::Matrix<real_t> a(2, 2, {1, 1, 3, -4});

       VolumeDescriptor desc({{2}});
       MatrixOperator A(desc, desc, a);

       linalg::Vector<real_t> _b({3, 2});
       auto b = DataContainer<real_t>(desc);
       b = _b;

       real_t epsilon = std::numeric_limits<real_t>::epsilon();

       WHEN("setting up a AB_GMRES solver")
       {
           AB_GMRES solver{A, b, epsilon};

           THEN("the clone works correctly")
           {
               auto gmresClone = solver.clone();

               REQUIRE_NE(gmresClone.get(), &solver);
               REQUIRE_EQ(*gmresClone, solver);
           }
       }
   }
}

TEST_CASE("Simple Square Symmetric Matrix Reconstruction")
{

   Logger::setLevel(Logger::LogLevel::OFF);

   GIVEN("a 2x2 Matrix A,a 2x1 Vector b and a 2x1 Vector x0")
   {

       linalg::Matrix<real_t> a(2, 2, {1, 1, 3, -4});

       VolumeDescriptor desc({{2}});
       MatrixOperator A(desc, desc, a);

       linalg::Vector<real_t> _b({3, 2});
       auto b = DataContainer<real_t>(desc);
       b = _b;

       linalg::Vector<real_t> _x({1, 2});
       auto x = DataContainer<real_t>(desc);
       x = _x;

       real_t epsilon = std::numeric_limits<real_t>::epsilon();

       WHEN("setting up a AB_GMRES solver")
       {
           AB_GMRES solver{A, b, epsilon};

           THEN("Testing GMRES Solver for 2 iterations")
           {
               AND_THEN("Solving for 2 Iterations")
               {
                   // actual solution for the linear system
                   linalg::Vector<real_t> solution({2, 1});

                   // should have converged for the given number of iterations
                   auto reconstruction = flatten(solver.solve(2, x));

                   AND_THEN("Reconstruction worked as expected")
                   {
                       REQUIRE_UNARY(checkApproxEq(solution(0), reconstruction(0), 0.0001));

                       REQUIRE_UNARY(checkApproxEq(solution(1), reconstruction(1), 0.0001));
                   }
               }
           }
       }
   }
}

TEST_CASE("Non-Square, Non-Symmetric Matrix Reconstruction")
{

   Logger::setLevel(Logger::LogLevel::OFF);

   GIVEN("A 3x4 Matrix Operator A, a 3x1 Vector Datacontainer b and no x0")
   {

       linalg::Matrix<real_t> a(3, 4, {1, 2, 3, 4, 1, 2, 3, -4, 1, -2, 3, 4});

       VolumeDescriptor domain({{4}});
       VolumeDescriptor range({{3}});
       MatrixOperator A(domain, range, a);

       linalg::Vector<real_t> _b({3, 2, 1});
       auto b = DataContainer<real_t>(range, _b);

       real_t epsilon = std::numeric_limits<real_t>::epsilon();

       WHEN("setting up a AB_GMRES solver")
       {
           AB_GMRES solver{A, b, epsilon};

           THEN("Check if x0 is created properly solve for 3 Iterations")
           {
               // actual solution for the linear system
               linalg::Vector<real_t> solution({0.15f, 0.5f, 0.45f, 0.125f});

               auto reconstruction = flatten(solver.solve(3));

               AND_THEN("Reconstruction worked as expected")
               {
                   REQUIRE_UNARY(checkApproxEq(solution(0), reconstruction(0), 0.0001));

                   REQUIRE_UNARY(checkApproxEq(solution(1), reconstruction(1), 0.0001));

                   REQUIRE_UNARY(checkApproxEq(solution(2), reconstruction(2), 0.0001));

                   REQUIRE_UNARY(checkApproxEq(solution(3), reconstruction(3), 0.0001));
               }
           }
       }
   }
}

TEST_CASE("Non-Square, Non-Symmetric Unmatched AB Matrix Reconstruction")
{
   Logger::setLevel(Logger::LogLevel::OFF);

   GIVEN("A 2x3 Matrix Operator A, 3x2 Matrix Operator B, a 3x1 Vector Datacontainer b and no x0")
   {

       linalg::Matrix<real_t> a(3, 4, {1, 2, 3, 4, 1, 2, 3, -4, 1, -2, 3, 4});

       VolumeDescriptor domain({{4}});
       VolumeDescriptor range({{3}});
       MatrixOperator A(domain, range, a);

       // Creating a B Matrix that is A.T value wise and applying a imagined "filter" so it doesnt
       // match A.T
       real_t filter = 0.95f;
       linalg::Matrix<real_t> bMatrix(4, 3,
                                      {1 * filter, 1 * filter, 1 * filter, 2 * filter, 2 * filter,
                                       -2 * filter, 3 * filter, 3 * filter, 3 * filter, 4 * filter,
                                       -4 * filter, 4 * filter});

       MatrixOperator B(range, domain, bMatrix);

       linalg::Vector<real_t> _b({3, 2, 1});
       auto b = DataContainer<real_t>(range);
       b = _b;

       linalg::Vector<real_t> _x({1, 2, 3, 4});
       auto x = DataContainer<real_t>(domain);
       x = _x;

       real_t epsilon = std::numeric_limits<real_t>::epsilon();

       WHEN("Setting up a simple unmatched AB_GMRES solver")
       {
           AB_GMRES solver{A, B, b, epsilon};

           THEN("Solving for 3 Iterations")
           {
               // actual solution for the linear system
               linalg::Vector<real_t> solution({0.15f, 0.5f, 0.45f, 0.125f});

               auto reconstruction = flatten(solver.solve(3));

               AND_THEN("Reconstruction worked as expected")
               {
                   REQUIRE_UNARY(checkApproxEq(solution(0), reconstruction(0), 0.0001));

                   REQUIRE_UNARY(checkApproxEq(solution(1), reconstruction(1), 0.0001));

                   REQUIRE_UNARY(checkApproxEq(solution(2), reconstruction(2), 0.0001));
               }
           }
       }
   }
}*/

// TODO test JosephsMethodCUDA for unmatched Reconstruction

TEST_SUITE_END();