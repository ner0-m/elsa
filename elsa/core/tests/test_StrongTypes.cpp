/**
 * @file test_StrongTypes.cpp
 *
 * @brief Test for Strong type classes
 *
 * @author David Frank - initial code
 */

#include "doctest/doctest.h"
#include "StrongTypes.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE("StrongTypes: Testing RotationAngles")
{

    using namespace geometry;
    using namespace geometry::detail;

    GIVEN("A 1D RotationAngles")
    {
        RotationAngles<1> angle{Degree{90}};

        THEN("The value and size are correct")
        {
            CHECK_EQ(angle[0], Radian{pi_t / 2});
        }
    }

    GIVEN("A 2D RotationAngles")
    {
        RotationAngles<2> angle{Degree{90}, Radian{pi_t / 4}};

        THEN("The value and size are correct")
        {
            CHECK_EQ(angle[0], Radian{pi_t / 2});
            CHECK_EQ(angle[1], Radian{pi_t / 4});
        }
    }

    GIVEN("A 3D RotationAngles")
    {
        RotationAngles<3> angle{Degree{90}, Radian{pi_t / 4}, Degree{180}};

        THEN("The value and size are correct")
        {
            CHECK_EQ(angle[0], Radian{pi_t / 2});
            CHECK_EQ(angle[1], Radian{pi_t / 4});
            CHECK_EQ(angle[2], Radian{pi_t});
        }
    }

    GIVEN("A RotationAngles3D")
    {
        RotationAngles3D angle{Degree{90}, Radian{pi_t / 4}, Degree{180}};

        THEN("The value and size are correct")
        {
            auto [g, b, a] = angle;

            CHECK_EQ(angle[0], Radian{pi_t / 2});
            CHECK_EQ(angle.gamma(), Radian{pi_t / 2});
            CHECK_EQ(g, Radian{pi_t / 2});

            CHECK_EQ(angle[1], Radian{pi_t / 4});
            CHECK_EQ(angle.beta(), Radian{pi_t / 4});
            CHECK_EQ(b, Radian{pi_t / 4});

            CHECK_EQ(angle[2], Radian{pi_t});
            CHECK_EQ(angle.alpha(), Radian{pi_t});
            CHECK_EQ(a, Radian{pi_t});
        }
    }
}

TEST_CASE("StrongTypes: Testing StaticRealVector")
{
    using namespace geometry;
    using namespace geometry::detail;

    GIVEN("A default constructed StaticRealVector")
    {
        StaticRealVector<0> vec;

        THEN("The Eigen Vector is of size 0")
        {
            auto eigenVec = vec.get();
            REQUIRE_EQ(eigenVec.size(), 0);
        }
    }

    GIVEN("A 1D StaticRealVector")
    {
        StaticRealVector<1> vec{1};

        THEN("The value and size are correct")
        {
            auto eigenVec = vec.get();
            CHECK_EQ(eigenVec.size(), 1);

            CHECK_EQ(vec[0], Approx(1));
        }
    }

    GIVEN("A 2D StaticRealVector")
    {
        StaticRealVector<2> vec{1, 2};

        THEN("The value and size are correct")
        {
            auto eigenVec = vec.get();
            CHECK_EQ(eigenVec.size(), 2);

            CHECK_EQ(vec[0], Approx(1));
            CHECK_EQ(vec[1], Approx(2));
        }
    }

    GIVEN("A 3D StaticRealVector")
    {
        StaticRealVector<3> vec{1, 2, 3};

        THEN("The value and size are correct")
        {
            auto eigenVec = vec.get();
            CHECK_EQ(eigenVec.size(), 3);

            CHECK_EQ(vec[0], Approx(1));
            CHECK_EQ(vec[1], Approx(2));
            CHECK_EQ(vec[2], Approx(3));
        }
    }

    GIVEN("A 4D StaticRealVector")
    {
        StaticRealVector<4> vec{1, 2, 3, 6};

        THEN("The value and size are correct")
        {
            auto eigenVec = vec.get();
            CHECK_EQ(eigenVec.size(), 4);

            CHECK_EQ(vec[0], Approx(1));
            CHECK_EQ(vec[1], Approx(2));
            CHECK_EQ(vec[2], Approx(3));
            CHECK_EQ(vec[3], Approx(6));
        }
    }
}

TEST_CASE("StrongTypes: Testing GeometryData")
{
    using namespace geometry;
    using namespace geometry::detail;

    GIVEN("A default constructed GeometryData")
    {
        GeometryData<0> data;

        THEN("The Eigen Vector is of size 0")
        {
            CHECK_EQ(data.getSpacing().size(), 0);
            CHECK_EQ(data.getLocationOfOrigin().size(), 0);
        }
    }

    GIVEN("A GeometryData for 1D data")
    {
        GeometryData data{Spacing1D{1}, OriginShift1D{0}};

        THEN("Spacing and Origin is of correct size and correct values")
        {
            CHECK_EQ(data.getSpacing().size(), 1);
            CHECK_EQ(data.getSpacing()[0], Approx(1));

            CHECK_EQ(data.getLocationOfOrigin().size(), 1);
            CHECK_EQ(data.getLocationOfOrigin()[0], Approx(0));
        }

        THEN("We can construct it from coefficients")
        {
            auto coeffs = IndexVector_t::Constant(1, 5);

            GeometryData<1> data2{Size1D{coeffs}};

            CHECK_EQ(data2.getSpacing().size(), 1);
            CHECK_EQ(data2.getSpacing()[0], Approx(1));

            CHECK_EQ(data2.getLocationOfOrigin().size(), 1);
            CHECK_EQ(data2.getLocationOfOrigin()[0], Approx(2.5));
        }
    }

    GIVEN("A GeometryData for 2D data")
    {
        GeometryData data{Spacing2D{1, 0.5}, OriginShift2D{0, 0.2}};

        THEN("Spacing and Origin is of correct size and correct values")
        {
            CHECK_EQ(data.getSpacing().size(), 2);
            CHECK_EQ(data.getSpacing()[0], Approx(1));
            CHECK_EQ(data.getSpacing()[1], Approx(0.5));

            CHECK_EQ(data.getLocationOfOrigin().size(), 2);
            CHECK_EQ(data.getLocationOfOrigin()[0], Approx(0));
            CHECK_EQ(data.getLocationOfOrigin()[1], Approx(0.2));
        }

        THEN("We can construct it from coefficients")
        {
            auto coeffs = IndexVector_t::Constant(2, 5);

            GeometryData<2> data2{Size2D{coeffs}, Spacing2D{2, 2}};

            CHECK_EQ(data2.getSpacing().size(), 2);
            CHECK_EQ(data2.getSpacing()[0], Approx(2));
            CHECK_EQ(data2.getSpacing()[1], Approx(2));

            CHECK_EQ(data2.getLocationOfOrigin().size(), 2);
            CHECK_EQ(data2.getLocationOfOrigin()[0], Approx(5));
            CHECK_EQ(data2.getLocationOfOrigin()[1], Approx(5));
        }
    }
}

TEST_CASE("StrongTypes: Testing VolumeData")
{
    using namespace geometry;

    GIVEN("Size coefficients for 2D")
    {
        IndexVector_t size(2);
        size << 10, 10;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            VolumeData2D volData{Size2D{size}};

            CHECK_EQ(volData.getSpacing().size(), 2);
            CHECK_EQ(volData.getSpacing()[0], Approx(1));
            CHECK_EQ(volData.getSpacing()[1], Approx(1));

            CHECK_EQ(volData.getLocationOfOrigin().size(), 2);
            CHECK_EQ(volData.getLocationOfOrigin()[0], Approx(5));
            CHECK_EQ(volData.getLocationOfOrigin()[1], Approx(5));
        }
    }

    GIVEN("Size coefficients and Spacing for 2D")
    {
        IndexVector_t size(2);
        size << 10, 10;

        RealVector_t spacing(2);
        spacing << 0.5, 2;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            VolumeData2D volData{Size2D{size}, Spacing2D{spacing}};

            CHECK_EQ(volData.getSpacing().size(), 2);
            CHECK_EQ(volData.getSpacing()[0], Approx(0.5));
            CHECK_EQ(volData.getSpacing()[1], Approx(2));

            CHECK_EQ(volData.getLocationOfOrigin().size(), 2);
            CHECK_EQ(volData.getLocationOfOrigin()[0], Approx(2.5));
            CHECK_EQ(volData.getLocationOfOrigin()[1], Approx(10));
        }
        THEN("Structured bindings produce correct results")
        {
            auto [sp, o] = VolumeData2D{Size2D{size}, Spacing2D{spacing}};

            CHECK_EQ(sp.size(), 2);
            CHECK_EQ(sp[0], Approx(0.5));
            CHECK_EQ(sp[1], Approx(2));

            CHECK_EQ(o.size(), 2);
            CHECK_EQ(o[0], Approx(2.5));
            CHECK_EQ(o[1], Approx(10));
        }
    }

    GIVEN("Size coefficients for 3D")
    {
        IndexVector_t size(3);
        size << 10, 10, 10;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            VolumeData3D volData{Size3D{size}};

            CHECK_EQ(volData.getSpacing().size(), 3);
            CHECK_EQ(volData.getSpacing()[0], Approx(1));
            CHECK_EQ(volData.getSpacing()[1], Approx(1));
            CHECK_EQ(volData.getSpacing()[2], Approx(1));

            CHECK_EQ(volData.getLocationOfOrigin().size(), 3);
            CHECK_EQ(volData.getLocationOfOrigin()[0], Approx(5));
            CHECK_EQ(volData.getLocationOfOrigin()[1], Approx(5));
            CHECK_EQ(volData.getLocationOfOrigin()[2], Approx(5));
        }
    }

    GIVEN("Size coefficients and Spacing for 2D")
    {
        IndexVector_t size(3);
        size << 10, 10, 10;

        RealVector_t spacing(3);
        spacing << 0.5, 2, 1;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            VolumeData3D volData{Size3D{size}, Spacing3D{spacing}};

            CHECK_EQ(volData.getSpacing().size(), 3);
            CHECK_EQ(volData.getSpacing()[0], Approx(0.5));
            CHECK_EQ(volData.getSpacing()[1], Approx(2));
            CHECK_EQ(volData.getSpacing()[2], Approx(1));

            CHECK_EQ(volData.getLocationOfOrigin().size(), 3);
            CHECK_EQ(volData.getLocationOfOrigin()[0], Approx(2.5));
            CHECK_EQ(volData.getLocationOfOrigin()[1], Approx(10));
            CHECK_EQ(volData.getLocationOfOrigin()[2], Approx(5));
        }
    }
}

TEST_CASE("StrongTypes: Testing SinogramData")
{
    using namespace geometry;

    GIVEN("Size coefficients for 2D")
    {
        IndexVector_t size(2);
        size << 10, 10;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            SinogramData2D volData{Size2D{size}};

            CHECK_EQ(volData.getSpacing().size(), 2);
            CHECK_EQ(volData.getSpacing()[0], Approx(1));
            CHECK_EQ(volData.getSpacing()[1], Approx(1));

            CHECK_EQ(volData.getLocationOfOrigin().size(), 2);
            CHECK_EQ(volData.getLocationOfOrigin()[0], Approx(5));
            CHECK_EQ(volData.getLocationOfOrigin()[1], Approx(5));
        }
    }

    GIVEN("Size coefficients and Spacing for 2D")
    {
        IndexVector_t size(2);
        size << 10, 10;

        RealVector_t spacing(2);
        spacing << 0.5, 2;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            SinogramData2D sinoData{Size2D{size}, Spacing2D{spacing}};

            CHECK_EQ(sinoData.getSpacing().size(), 2);
            CHECK_EQ(sinoData.getSpacing()[0], Approx(0.5));
            CHECK_EQ(sinoData.getSpacing()[1], Approx(2));

            CHECK_EQ(sinoData.getLocationOfOrigin().size(), 2);
            CHECK_EQ(sinoData.getLocationOfOrigin()[0], Approx(2.5));
            CHECK_EQ(sinoData.getLocationOfOrigin()[1], Approx(10));

            CHECK_THROWS(SinogramData2D{Size2D{size}, Spacing2D{RealVector_t(3)}});
        }
    }

    GIVEN("Spacing and Origin shift in 2D")
    {
        RealVector_t spacing(2);
        spacing << 1, 1;

        RealVector_t shift(2);
        shift << 1, 1;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            auto [s, o] = SinogramData2D{Spacing2D{spacing}, OriginShift2D{shift}};

            CHECK_EQ(s.size(), 2);
            CHECK_EQ(s[0], Approx(1));
            CHECK_EQ(s[1], Approx(1));

            CHECK_EQ(o.size(), 2);
            CHECK_EQ(o[0], Approx(1));
            CHECK_EQ(o[1], Approx(1));

            // Test that exceptions are thrown
            CHECK_THROWS(SinogramData2D{Spacing2D{spacing}, RealVector_t(3)});
            CHECK_THROWS(SinogramData2D{RealVector_t(3), OriginShift2D{shift}});
            CHECK_THROWS(SinogramData2D{RealVector_t(3), RealVector_t(3)});
        }
    }

    GIVEN("Size coefficients for 3D")
    {
        IndexVector_t size(3);
        size << 10, 10, 10;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            SinogramData3D sinoData{Size3D{size}};

            CHECK_EQ(sinoData.getSpacing().size(), 3);
            CHECK_EQ(sinoData.getSpacing()[0], Approx(1));
            CHECK_EQ(sinoData.getSpacing()[1], Approx(1));
            CHECK_EQ(sinoData.getSpacing()[2], Approx(1));

            CHECK_EQ(sinoData.getLocationOfOrigin().size(), 3);
            CHECK_EQ(sinoData.getLocationOfOrigin()[0], Approx(5));
            CHECK_EQ(sinoData.getLocationOfOrigin()[1], Approx(5));
            CHECK_EQ(sinoData.getLocationOfOrigin()[2], Approx(5));
        }
    }

    GIVEN("Size coefficients and Spacing for 2D")
    {
        IndexVector_t size(3);
        size << 10, 10, 10;

        RealVector_t spacing(3);
        spacing << 0.5, 2, 1;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            auto [s, o] = SinogramData3D{Size3D{size}, Spacing3D{spacing}};

            CHECK_EQ(s.size(), 3);
            CHECK_EQ(s[0], Approx(0.5));
            CHECK_EQ(s[1], Approx(2));
            CHECK_EQ(s[2], Approx(1));

            CHECK_EQ(o.size(), 3);
            CHECK_EQ(o[0], Approx(2.5));
            CHECK_EQ(o[1], Approx(10));
            CHECK_EQ(o[2], Approx(5));
        }
    }

    GIVEN("Spacing and Origin shift in 3D")
    {
        RealVector_t spacing(3);
        spacing << 1, 1, 1;

        RealVector_t shift(3);
        shift << 1, 1, 1;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            SinogramData3D volData{Spacing3D{spacing}, OriginShift3D{shift}};

            CHECK_EQ(volData.getSpacing().size(), 3);
            CHECK_EQ(volData.getSpacing()[0], Approx(1));
            CHECK_EQ(volData.getSpacing()[1], Approx(1));
            CHECK_EQ(volData.getSpacing()[2], Approx(1));

            CHECK_EQ(volData.getLocationOfOrigin().size(), 3);
            CHECK_EQ(volData.getLocationOfOrigin()[0], Approx(1));
            CHECK_EQ(volData.getLocationOfOrigin()[1], Approx(1));
            CHECK_EQ(volData.getLocationOfOrigin()[2], Approx(1));

            // Test that exceptions are thrown
            CHECK_THROWS(SinogramData3D{Spacing3D{spacing}, RealVector_t(2)});
            CHECK_THROWS(SinogramData3D{RealVector_t(4), OriginShift3D{shift}});
            CHECK_THROWS(SinogramData3D{RealVector_t(1), RealVector_t(3)});
        }
    }
}

TEST_CASE("StrongTypes: Testing Threshold")
{
    using namespace geometry;

    GIVEN("Valid arguments for Thresholds")
    {
        real_t one = 1;
        real_t half = 1.0 / 2;
        real_t nine = 9;

        THEN("Overloaded relational operators are implemented correctly")
        {
            Threshold<real_t> tOne{one};
            Threshold<real_t> tHalf{half};
            Threshold<real_t> tNine{nine};

            CHECK_EQ(tOne, one);
            CHECK_GT((nine - tOne), tOne);
            CHECK_GE(nine, tHalf);
            CHECK_NE(tNine, half);
            CHECK_LT(tHalf, (one + tNine));
            CHECK_LE((tHalf + half), (one + tNine));
        }
    }

    GIVEN("Invalid arguments for Thresholds")
    {
        real_t zero = 0;
        real_t neg1 = -1;

        THEN("An exception is thrown as such Thresholds cannot be constructed")
        {
            CHECK_THROWS(Threshold<real_t>{zero});
            CHECK_THROWS(Threshold<real_t>{neg1});
        }
    }
}

TEST_SUITE_END();
