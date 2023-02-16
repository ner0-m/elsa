#include "doctest/doctest.h"

#include "Error.h"
#include "PGD.h"
#include "Identity.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "ProximalBoxConstraint.h"
#include "ProximalL1.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE_TEMPLATE("PDG: Solving a least squares problem", data_t, float, double)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    VolumeDescriptor volDescr({4, 7});

    Vector_t<data_t> bVec(volDescr.getNumberOfCoefficients());
    bVec.setRandom();
    DataContainer<data_t> b(volDescr, bVec);

    Identity<data_t> A(volDescr);

    PGD<data_t> solver(A, b, ProximalBoxConstraint<data_t>{});

    auto reco = solver.solve(1);

    CHECK_EQ(reco.squaredL2Norm(), doctest::Approx(b.squaredL2Norm()));

    THEN("Clone is equal to original one")
    {
        auto clone = solver.clone();

        CHECK_EQ(*clone, solver);
        CHECK_NE(clone.get(), &solver);
    }
}

TEST_CASE_TEMPLATE("PDG: Solving a least squares problem with non negativity constraint", data_t,
                   float, double)
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);

    VolumeDescriptor volDescr({4, 7});

    // Ensure data is actually non negative
    Vector_t<data_t> bVec(volDescr.getNumberOfCoefficients());
    bVec = bVec.setRandom().cwiseAbs();
    DataContainer<data_t> b(volDescr, bVec);

    Identity<data_t> A(volDescr);

    auto prox = ProximalBoxConstraint<data_t>{0};
    PGD<data_t> solver(A, b, prox);

    auto reco = solver.solve(2);

    CHECK_EQ(reco.squaredL2Norm(), doctest::Approx(b.squaredL2Norm()));

    THEN("Clone is equal to original one")
    {
        auto clone = solver.clone();

        CHECK_EQ(*clone, solver);
        CHECK_NE(clone.get(), &solver);
    }
}

TEST_SUITE_END();
