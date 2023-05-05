#include "doctest/doctest.h"

#include "IS_ADMML2.h"
#include "Identity.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "ProximalBoxConstraint.h"
#include "elsaDefines.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

template <class data_t>
struct ProximalConstFunction {
    DataContainer<data_t> apply(const DataContainer<data_t>& v, SelfType_t<data_t> t) const
    {
        return v * t;
    }

    void apply(const DataContainer<data_t>& v, SelfType_t<data_t> t,
               DataContainer<data_t>& prox) const
    {
        prox = v * t;
    }
};

TEST_SUITE_BEGIN("solvers");

TEST_CASE_TEMPLATE("ADMML2: Solving problems", data_t, float, double)
{
    Logger::setLevel(Logger::LogLevel::OFF);

    /// Set seed such that matrixes are always the same
    srand((unsigned int) 666);

    VolumeDescriptor volDescr({7, 11});

    GIVEN("Solve Least squares")
    {
        Vector_t<data_t> bVec(volDescr.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> b(volDescr, bVec);

        Identity<data_t> op(volDescr);

        Identity<data_t> id(volDescr);
        ProximalConstFunction<data_t> proxg;

        auto admm = IS_ADMML2<data_t>(op, b, id, proxg, 1, 1);

        auto reco = admm.solve(10);

        CAPTURE(reco);
        CAPTURE(b);

        THEN("Reconstruction is approximately equal to b")
        {
            CHECK_EQ((reco - b).squaredL2Norm(), doctest::Approx(0).epsilon(0.01));
        }

        THEN("Solver can be used twice")
        {
            auto reco2 = admm.solve(10);
            CHECK_EQ(reco, reco2);
        }

        // THEN("clone is equal")
        // {
        //     auto clone = admm.clone();
        //
        //     CHECK_NE(clone.get(), &admm);
        //     CHECK_EQ(*clone, admm);
        // }
    }

    // GIVEN("Solve Least squares with non-negativity constraint")
    // {
    //     Vector_t<data_t> bVec(volDescr.getNumberOfCoefficients());
    //     bVec.setRandom();
    //     DataContainer<data_t> b(volDescr, bVec);
    //
    //     Identity<data_t> op(volDescr);
    //
    //     Identity<data_t> A(volDescr);
    //     ProximalBoxConstraint<data_t> proxg{0};
    //
    //     auto admm = ADMML2<data_t>(op, b, A, proxg, data_t{10000});
    //
    //     auto reco = admm.solve(100);
    //
    //     CAPTURE(reco);
    //     CAPTURE(b);
    //
    //     THEN("All values are approximately larger than zero")
    //     {
    //         for (index_t i = 0; i < reco.getSize(); ++i) {
    //             INFO(i);
    //             CHECK_GE(reco[i], doctest::Approx(0));
    //         }
    //     }
    //
    //     THEN("clone is equal")
    //     {
    //         auto clone = admm.clone();
    //
    //         CHECK_NE(clone.get(), &admm);
    //         CHECK_EQ(*clone, admm);
    //     }
    // }
}

TEST_SUITE_END();
