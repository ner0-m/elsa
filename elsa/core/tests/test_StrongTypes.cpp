/**
 * @file test_StrongTypes.cpp
 *
 * @brief Test for Strong type classes
 *
 * @author David Frank - initial code
 */

#include <catch2/catch.hpp>
#include "StrongTypes.h"

using namespace elsa;

SCENARIO("Testing RotationAngles")
{

    using namespace geometry;
    using namespace geometry::detail;

    GIVEN("A 1D RotationAngles")
    {
        RotationAngles<1> angle{Degree{90}};

        THEN("The value and size are correct") { CHECK(angle[0] == Radian{pi_t / 2}); }
    }

    GIVEN("A 2D RotationAngles")
    {
        RotationAngles<2> angle{Degree{90}, Radian{pi_t / 4}};

        THEN("The value and size are correct")
        {
            CHECK(angle[0] == Radian{pi_t / 2});
            CHECK(angle[1] == Radian{pi_t / 4});
        }
    }

    GIVEN("A 3D RotationAngles")
    {
        RotationAngles<3> angle{Degree{90}, Radian{pi_t / 4}, Degree{180}};

        THEN("The value and size are correct")
        {
            CHECK(angle[0] == Radian{pi_t / 2});
            CHECK(angle[1] == Radian{pi_t / 4});
            CHECK(angle[2] == Radian{pi_t});
        }
    }

    GIVEN("A RotationAngles3D")
    {
        RotationAngles3D angle{Degree{90}, Radian{pi_t / 4}, Degree{180}};

        THEN("The value and size are correct")
        {
            auto [g, b, a] = angle;

            CHECK(angle[0] == Radian{pi_t / 2});
            CHECK(angle.gamma() == Radian{pi_t / 2});
            CHECK(g == Radian{pi_t / 2});

            CHECK(angle[1] == Radian{pi_t / 4});
            CHECK(angle.beta() == Radian{pi_t / 4});
            CHECK(b == Radian{pi_t / 4});

            CHECK(angle[2] == Radian{pi_t});
            CHECK(angle.alpha() == Radian{pi_t});
            CHECK(a == Radian{pi_t});
        }
    }
}

SCENARIO("Testing StaticRealVector")
{
    using namespace geometry;
    using namespace geometry::detail;

    GIVEN("A default constructed StaticRealVector")
    {
        StaticRealVector<0> vec;

        THEN("The Eigen Vector is of size 0")
        {
            auto eigenVec = vec.get();
            REQUIRE(eigenVec.size() == 0);
        }
    }

    GIVEN("A 1D StaticRealVector")
    {
        StaticRealVector<1> vec{1};

        THEN("The value and size are correct")
        {
            auto eigenVec = vec.get();
            CHECK(eigenVec.size() == 1);

            CHECK(vec[0] == Approx(1));
        }
    }

    GIVEN("A 2D StaticRealVector")
    {
        StaticRealVector<2> vec{1, 2};

        THEN("The value and size are correct")
        {
            auto eigenVec = vec.get();
            CHECK(eigenVec.size() == 2);

            CHECK(vec[0] == Approx(1));
            CHECK(vec[1] == Approx(2));
        }
    }

    GIVEN("A 3D StaticRealVector")
    {
        StaticRealVector<3> vec{1, 2, 3};

        THEN("The value and size are correct")
        {
            auto eigenVec = vec.get();
            CHECK(eigenVec.size() == 3);

            CHECK(vec[0] == Approx(1));
            CHECK(vec[1] == Approx(2));
            CHECK(vec[2] == Approx(3));
        }
    }

    GIVEN("A 4D StaticRealVector")
    {
        StaticRealVector<4> vec{1, 2, 3, 6};

        THEN("The value and size are correct")
        {
            auto eigenVec = vec.get();
            CHECK(eigenVec.size() == 4);

            CHECK(vec[0] == Approx(1));
            CHECK(vec[1] == Approx(2));
            CHECK(vec[2] == Approx(3));
            CHECK(vec[3] == Approx(6));
        }
    }
}

SCENARIO("Testing GeometryData")
{
    using namespace geometry;
    using namespace geometry::detail;

    GIVEN("A default constructed GeometryData")
    {
        GeometryData<0> data;

        THEN("The Eigen Vector is of size 0")
        {
            CHECK(data.getSpacing().size() == 0);
            CHECK(data.getLocationOfOrigin().size() == 0);
        }
    }

    GIVEN("A GeometryData for 1D data")
    {
        GeometryData data{Spacing1D{1}, OriginShift1D{0}};

        // static_assert(2 == std::tuple_size<decltype(data)>::value);

        THEN("Spacing and Origin is of correct size and correct values")
        {
            // auto [spacing, origin] = data;

            CHECK(data.getSpacing().size() == 1);
            CHECK(data.getSpacing()[0] == Approx(1));

            CHECK(data.getLocationOfOrigin().size() == 1);
            CHECK(data.getLocationOfOrigin()[0] == Approx(0));
        }

        THEN("We can construct it from coefficients")
        {
            auto coeffs = IndexVector_t::Constant(1, 5);

            GeometryData<1> data2{Size1D{coeffs}};

            CHECK(data2.getSpacing().size() == 1);
            CHECK(data2.getSpacing()[0] == Approx(1));

            CHECK(data2.getLocationOfOrigin().size() == 1);
            CHECK(data2.getLocationOfOrigin()[0] == Approx(2.5));
        }
    }

    GIVEN("A GeometryData for 2D data")
    {
        GeometryData data{Spacing2D{1, 0.5}, OriginShift2D{0, 0.2}};

        THEN("Spacing and Origin is of correct size and correct values")
        {
            CHECK(data.getSpacing().size() == 2);
            CHECK(data.getSpacing()[0] == Approx(1));
            CHECK(data.getSpacing()[1] == Approx(0.5));

            CHECK(data.getLocationOfOrigin().size() == 2);
            CHECK(data.getLocationOfOrigin()[0] == Approx(0));
            CHECK(data.getLocationOfOrigin()[1] == Approx(0.2));
        }

        THEN("We can construct it from coefficients")
        {
            auto coeffs = IndexVector_t::Constant(2, 5);

            GeometryData<2> data2{Size2D{coeffs}, Spacing2D{2, 2}};

            CHECK(data2.getSpacing().size() == 2);
            CHECK(data2.getSpacing()[0] == Approx(2));
            CHECK(data2.getSpacing()[1] == Approx(2));

            CHECK(data2.getLocationOfOrigin().size() == 2);
            CHECK(data2.getLocationOfOrigin()[0] == Approx(5));
            CHECK(data2.getLocationOfOrigin()[1] == Approx(5));
        }
    }
}

SCENARIO("Testing VolumeData")
{
    using namespace geometry;

    GIVEN("Size coefficients for 2D")
    {
        IndexVector_t size(2);
        size << 10, 10;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            VolumeData2D volData{Size2D{size}};

            CHECK(volData.getSpacing().size() == 2);
            CHECK(volData.getSpacing()[0] == Approx(1));
            CHECK(volData.getSpacing()[1] == Approx(1));

            CHECK(volData.getLocationOfOrigin().size() == 2);
            CHECK(volData.getLocationOfOrigin()[0] == Approx(5));
            CHECK(volData.getLocationOfOrigin()[1] == Approx(5));
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

            CHECK(volData.getSpacing().size() == 2);
            CHECK(volData.getSpacing()[0] == Approx(0.5));
            CHECK(volData.getSpacing()[1] == Approx(2));

            CHECK(volData.getLocationOfOrigin().size() == 2);
            CHECK(volData.getLocationOfOrigin()[0] == Approx(2.5));
            CHECK(volData.getLocationOfOrigin()[1] == Approx(10));
        }
        THEN("Structured bindings produce correct results")
        {
            auto [sp, o] = VolumeData2D{Size2D{size}, Spacing2D{spacing}};

            CHECK(sp.size() == 2);
            CHECK(sp[0] == Approx(0.5));
            CHECK(sp[1] == Approx(2));

            CHECK(o.size() == 2);
            CHECK(o[0] == Approx(2.5));
            CHECK(o[1] == Approx(10));
        }
    }

    GIVEN("Size coefficients for 3D")
    {
        IndexVector_t size(3);
        size << 10, 10, 10;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            VolumeData3D volData{Size3D{size}};

            CHECK(volData.getSpacing().size() == 3);
            CHECK(volData.getSpacing()[0] == Approx(1));
            CHECK(volData.getSpacing()[1] == Approx(1));
            CHECK(volData.getSpacing()[2] == Approx(1));

            CHECK(volData.getLocationOfOrigin().size() == 3);
            CHECK(volData.getLocationOfOrigin()[0] == Approx(5));
            CHECK(volData.getLocationOfOrigin()[1] == Approx(5));
            CHECK(volData.getLocationOfOrigin()[2] == Approx(5));
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

            CHECK(volData.getSpacing().size() == 3);
            CHECK(volData.getSpacing()[0] == Approx(0.5));
            CHECK(volData.getSpacing()[1] == Approx(2));
            CHECK(volData.getSpacing()[2] == Approx(1));

            CHECK(volData.getLocationOfOrigin().size() == 3);
            CHECK(volData.getLocationOfOrigin()[0] == Approx(2.5));
            CHECK(volData.getLocationOfOrigin()[1] == Approx(10));
            CHECK(volData.getLocationOfOrigin()[2] == Approx(5));
        }
    }
}

SCENARIO("Testing SinogramData")
{
    using namespace geometry;

    GIVEN("Size coefficients for 2D")
    {
        IndexVector_t size(2);
        size << 10, 10;

        THEN("Then Spacing and location of origin is calculated correctly")
        {
            SinogramData2D volData{Size2D{size}};

            CHECK(volData.getSpacing().size() == 2);
            CHECK(volData.getSpacing()[0] == Approx(1));
            CHECK(volData.getSpacing()[1] == Approx(1));

            CHECK(volData.getLocationOfOrigin().size() == 2);
            CHECK(volData.getLocationOfOrigin()[0] == Approx(5));
            CHECK(volData.getLocationOfOrigin()[1] == Approx(5));
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

            CHECK(sinoData.getSpacing().size() == 2);
            CHECK(sinoData.getSpacing()[0] == Approx(0.5));
            CHECK(sinoData.getSpacing()[1] == Approx(2));

            CHECK(sinoData.getLocationOfOrigin().size() == 2);
            CHECK(sinoData.getLocationOfOrigin()[0] == Approx(2.5));
            CHECK(sinoData.getLocationOfOrigin()[1] == Approx(10));

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

            CHECK(s.size() == 2);
            CHECK(s[0] == Approx(1));
            CHECK(s[1] == Approx(1));

            CHECK(o.size() == 2);
            CHECK(o[0] == Approx(1));
            CHECK(o[1] == Approx(1));

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

            CHECK(sinoData.getSpacing().size() == 3);
            CHECK(sinoData.getSpacing()[0] == Approx(1));
            CHECK(sinoData.getSpacing()[1] == Approx(1));
            CHECK(sinoData.getSpacing()[2] == Approx(1));

            CHECK(sinoData.getLocationOfOrigin().size() == 3);
            CHECK(sinoData.getLocationOfOrigin()[0] == Approx(5));
            CHECK(sinoData.getLocationOfOrigin()[1] == Approx(5));
            CHECK(sinoData.getLocationOfOrigin()[2] == Approx(5));
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

            CHECK(s.size() == 3);
            CHECK(s[0] == Approx(0.5));
            CHECK(s[1] == Approx(2));
            CHECK(s[2] == Approx(1));

            CHECK(o.size() == 3);
            CHECK(o[0] == Approx(2.5));
            CHECK(o[1] == Approx(10));
            CHECK(o[2] == Approx(5));
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

            CHECK(volData.getSpacing().size() == 3);
            CHECK(volData.getSpacing()[0] == Approx(1));
            CHECK(volData.getSpacing()[1] == Approx(1));
            CHECK(volData.getSpacing()[2] == Approx(1));

            CHECK(volData.getLocationOfOrigin().size() == 3);
            CHECK(volData.getLocationOfOrigin()[0] == Approx(1));
            CHECK(volData.getLocationOfOrigin()[1] == Approx(1));
            CHECK(volData.getLocationOfOrigin()[2] == Approx(1));

            // Test that exceptions are thrown
            CHECK_THROWS(SinogramData3D{Spacing3D{spacing}, RealVector_t(2)});
            CHECK_THROWS(SinogramData3D{RealVector_t(4), OriginShift3D{shift}});
            CHECK_THROWS(SinogramData3D{RealVector_t(1), RealVector_t(3)});
        }
    }
}

SCENARIO("Testing Threshold")
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

            CHECK(tOne == one);
            CHECK((nine - tOne) > tOne);
            CHECK(nine >= tHalf);
            CHECK(tNine != half);
            CHECK(tHalf < (one + tNine));
            CHECK((tHalf + half) <= (one + tNine));
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
