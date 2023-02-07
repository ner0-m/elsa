#include <doctest/doctest.h>

#include "CombinedProximal.h"
#include "VolumeDescriptor.h"
#include "IdenticalBlocksDescriptor.h"
#include "ProximalL0.h"
#include "ProximalL1.h"
#include "ProximalL2Squared.h"
#include "StrongTypes.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("CombinedProximal: Testing with a single functional", data_t, float, double)
{
    ProximalL0<data_t> l1;

    CombinedProximal<data_t> prox(l1);

    VolumeDescriptor desc({8});
    IdenticalBlocksDescriptor blockdesc(1, desc);

    DataContainer<data_t> x(blockdesc, Vector_t<data_t>({{-2, 3, 4, -7, 7, 8, 8, 3}}));
    DataContainer<data_t> y(desc, Vector_t<data_t>({{-2, 3, 4, -7, 7, 8, 8, 3}}));

    auto res = prox.apply(x, 4);
    auto expected = l1.apply(y, 4);

    CHECK_UNARY(isApprox(res, expected));
}

TEST_CASE_TEMPLATE("CombinedProximal: Testing with a two functional", data_t, float, double)
{
    ProximalL0<data_t> prox1;
    ProximalL1<data_t> prox2;

    CombinedProximal<data_t> prox(prox1, prox2);

    VolumeDescriptor desc({8});
    IdenticalBlocksDescriptor blockdesc(2, desc);

    // clang-format off
    DataContainer<data_t> x(blockdesc, Vector_t<data_t>({{
            -2, 3, 4, -7,  7,  8,  8,   3,
             2, 9, 2, -1, 10, 23, 49, -23
    }}));
    // clang-format on
    DataContainer<data_t> y1(desc, Vector_t<data_t>({{-2, 3, 4, -7, 7, 8, 8, 3}}));
    DataContainer<data_t> y2(desc, Vector_t<data_t>({{2, 9, 2, -1, 10, 23, 49, -23}}));

    auto res = prox.apply(x, 4);
    DataContainer<data_t> expected(blockdesc);
    expected.getBlock(0) = prox1.apply(y1, 4);
    expected.getBlock(1) = prox2.apply(y2, 4);

    CHECK_UNARY(isApprox(res, expected));
}

TEST_CASE_TEMPLATE("CombinedProximal: Testing with a three functional", data_t, float, double)
{
    ProximalL0<data_t> prox1;
    ProximalL1<data_t> prox2;
    ProximalL2Squared<data_t> prox3;

    CombinedProximal<data_t> prox(prox1, prox2, prox3);

    VolumeDescriptor desc({8});
    IdenticalBlocksDescriptor blockdesc(3, desc);

    // clang-format off
    DataContainer<data_t> x(blockdesc, Vector_t<data_t>({{
            -2, 3, 4, -7, 7, 8, 8, 3,
            -2, 3, 4, -7, 7, 8, 8, 3,
            -2, 3, 4, -7, 7, 8, 8, 3,
    }}));
    // clang-format on
    DataContainer<data_t> y(desc, Vector_t<data_t>({{-2, 3, 4, -7, 7, 8, 8, 3}}));

    auto res = prox.apply(x, 4);
    DataContainer<data_t> expected(blockdesc);
    expected.getBlock(0) = prox1.apply(y, 4);
    expected.getBlock(1) = prox2.apply(y, 4);
    expected.getBlock(2) = prox3.apply(y, 4);

    CHECK_UNARY(isApprox(res, expected));
}

TEST_CASE_TEMPLATE("CombinedProximal: Testing with a four functional", data_t, float, double)
{
    VolumeDescriptor desc({8});
    DataContainer<data_t> b(desc, Vector_t<data_t>({{-2, 3, 4, -7, 7, 8, 8, 3}}));

    ProximalL0<data_t> prox1;
    ProximalL1<data_t> prox2;
    ProximalL2Squared<data_t> prox3;
    ProximalL2Squared<data_t> prox4(b);

    CombinedProximal<data_t> prox(prox1, prox2, prox3, prox4);

    IdenticalBlocksDescriptor blockdesc(4, desc);

    // clang-format off
    DataContainer<data_t> x(blockdesc, Vector_t<data_t>({{
            -2, 3, 4, -7, 7, 8, 8, 3,
            -2, 3, 4, -7, 7, 8, 8, 3,
            -2, 3, 4, -7, 7, 8, 8, 3,
            -2, 3, 4, -7, 7, 8, 8, 3,
    }}));
    // clang-format on
    DataContainer<data_t> y(desc, Vector_t<data_t>({{-2, 3, 4, -7, 7, 8, 8, 3}}));

    auto res = prox.apply(x, 4);
    DataContainer<data_t> expected(blockdesc);
    expected.getBlock(0) = prox1.apply(y, 4);
    expected.getBlock(1) = prox2.apply(y, 4);
    expected.getBlock(2) = prox3.apply(y, 4);
    expected.getBlock(3) = prox4.apply(y, 4);

    CAPTURE(expected);
    CAPTURE(res);
    CHECK_UNARY(isApprox(res, expected));
}
