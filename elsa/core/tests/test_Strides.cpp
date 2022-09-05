#include "doctest/doctest.h"

#include "DataDescriptor.h"

TEST_CASE("Compute strides from 2D shape")
{
    elsa::IndexVector_t shape({{16, 16}});

    auto strides = elsa::computeStrides(shape);

    CHECK_EQ(strides[0], 1);
    CHECK_EQ(strides[1], 16);
}

TEST_CASE("Compute strides from 3D shape")
{
    elsa::IndexVector_t shape({{16, 16, 16}});

    auto strides = elsa::computeStrides(shape);

    CHECK_EQ(strides[0], 1);
    CHECK_EQ(strides[1], 16);
    CHECK_EQ(strides[2], 16 * 16);
}

TEST_CASE("Convert coordinate to memory location")
{
    elsa::IndexVector_t strides({{1, 16, 16 * 16}});

    THEN("Index of [0, 0, 0] = 0")
    {
        elsa::IndexVector_t coord({{0, 0, 0}});
        auto index = elsa::ravelIndex(coord, strides);

        CHECK_EQ(index, 0);
    }

    THEN("Index of [15, 15, 15] = 4095")
    {
        elsa::IndexVector_t coord({{15, 15, 15}});
        auto index = elsa::ravelIndex(coord, strides);

        CHECK_EQ(index, 16 * 16 * 16 - 1);
    }

    THEN("Index of [15, 0, 0] = 4095")
    {
        elsa::IndexVector_t coord({{15, 0, 0}});
        auto index = elsa::ravelIndex(coord, strides);

        CHECK_EQ(index, 15);
    }

    THEN("Index of [0, 15, 0] = 240")
    {
        elsa::IndexVector_t coord({{0, 15, 0}});
        auto index = elsa::ravelIndex(coord, strides);

        CHECK_EQ(index, 16 * 15);
    }

    THEN("Index of [0, 0, 15] = 3840")
    {
        elsa::IndexVector_t coord({{0, 0, 15}});
        auto index = elsa::ravelIndex(coord, strides);

        CHECK_EQ(index, 16 * 16 * 15);
    }
}

TEST_CASE("Convert memory location to coordinate")
{
    elsa::IndexVector_t strides({{1, 16, 16 * 16}});

    THEN("Index of [0, 0, 0] = 0")
    {
        auto index = 0;
        auto coord = elsa::unravelIndex(index, strides);

        CHECK_EQ(coord[0], 0);
        CHECK_EQ(coord[1], 0);
        CHECK_EQ(coord[2], 0);
    }

    THEN("Index of [15, 15, 15] = 4095")
    {
        auto index = 4095;
        auto coord = elsa::unravelIndex(index, strides);

        CHECK_EQ(coord[0], 15);
        CHECK_EQ(coord[1], 15);
        CHECK_EQ(coord[2], 15);
    }

    THEN("Index of [15, 0, 0] = 15")
    {
        auto index = 15;
        auto coord = elsa::unravelIndex(index, strides);

        CHECK_EQ(coord[0], 15);
        CHECK_EQ(coord[1], 0);
        CHECK_EQ(coord[2], 0);
    }

    THEN("Index of [0, 15, 0] = 240")
    {
        auto index = 240;
        auto coord = elsa::unravelIndex(index, strides);

        CHECK_EQ(coord[0], 0);
        CHECK_EQ(coord[1], 15);
        CHECK_EQ(coord[2], 0);
    }

    THEN("Index of [0, 0, 15] = 3840")
    {
        auto index = 3840;
        auto coord = elsa::unravelIndex(index, strides);

        CHECK_EQ(coord[0], 0);
        CHECK_EQ(coord[1], 0);
        CHECK_EQ(coord[2], 15);
    }
}
