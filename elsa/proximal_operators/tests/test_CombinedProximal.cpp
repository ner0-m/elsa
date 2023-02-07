#include <doctest/doctest.h>

#include "CombinedProximal.h"
#include "VolumeDescriptor.h"
#include "IdenticalBlocksDescriptor.h"
#include "ProximalL0.h"
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

    auto res = prox.apply(x, geometry::Threshold<data_t>{4});
    auto expected = l1.apply(y, geometry::Threshold<data_t>{4});

    CHECK_UNARY(isApprox(res, expected));
}

TEST_CASE_TEMPLATE("CombinedProximal: Testing with a two functional", data_t, float, double)
{
    ProximalL0<data_t> l11;
    ProximalL0<data_t> l12;

    CombinedProximal<data_t> prox(l11, l12);

    VolumeDescriptor desc({8});
    IdenticalBlocksDescriptor blockdesc(2, desc);

    // clang-format off
    DataContainer<data_t> x(blockdesc, Vector_t<data_t>({{
            -2, 3, 4, -7, 7, 8, 8, 3,
            -2, 3, 4, -7, 7, 8, 8, 3,
    }}));
    // clang-format on
    DataContainer<data_t> y(desc, Vector_t<data_t>({{-2, 3, 4, -7, 7, 8, 8, 3}}));

    auto res = prox.apply(x, geometry::Threshold<data_t>{4});
    DataContainer<data_t> expected(blockdesc);
    expected.getBlock(0) = l11.apply(y, geometry::Threshold<data_t>{4});
    expected.getBlock(1) = l12.apply(y, geometry::Threshold<data_t>{4});

    CHECK_UNARY(isApprox(res, expected));
}

TEST_CASE_TEMPLATE("CombinedProximal: Testing with a three functional", data_t, float, double)
{
    ProximalL0<data_t> l11;
    ProximalL0<data_t> l12;
    ProximalL0<data_t> l13;

    CombinedProximal<data_t> prox(l11, l12, l13);

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

    auto res = prox.apply(x, geometry::Threshold<data_t>{4});
    DataContainer<data_t> expected(blockdesc);
    expected.getBlock(0) = l11.apply(y, geometry::Threshold<data_t>{4});
    expected.getBlock(1) = l12.apply(y, geometry::Threshold<data_t>{4});
    expected.getBlock(2) = l13.apply(y, geometry::Threshold<data_t>{4});

    CHECK_UNARY(isApprox(res, expected));
}

TEST_CASE_TEMPLATE("CombinedProximal: Testing with a four functional", data_t, float, double)
{
    ProximalL0<data_t> l11;
    ProximalL0<data_t> l12;
    ProximalL0<data_t> l13;
    ProximalL0<data_t> l14;

    CombinedProximal<data_t> prox(l11, l12, l13, l14);

    VolumeDescriptor desc({8});
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

    auto res = prox.apply(x, geometry::Threshold<data_t>{4});
    DataContainer<data_t> expected(blockdesc);
    expected.getBlock(0) = l11.apply(y, geometry::Threshold<data_t>{4});
    expected.getBlock(1) = l12.apply(y, geometry::Threshold<data_t>{4});
    expected.getBlock(2) = l13.apply(y, geometry::Threshold<data_t>{4});
    expected.getBlock(3) = l14.apply(y, geometry::Threshold<data_t>{4});

    CHECK_UNARY(isApprox(res, expected));
}
