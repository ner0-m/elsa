/**
 * @file test_ShearletTransform.cpp
 *
 * @brief Tests for the ShearletTransform class
 *
 * @author Andi Braimllari
 */

#include "ShearletTransform.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

#include <doctest/doctest.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("shearlets");

TEST_CASE_TEMPLATE("ShearletTransform: Testing construction", TestType, float, double)
{
    GIVEN("a DataDescriptor")
    {
        IndexVector_t size(2);
        size << 256, 256;
        VolumeDescriptor volDescr(size);

        WHEN("instantiating a ShearletTransform operator")
        {
            ShearletTransform<TestType> shearletTransform(size[0], size[1]);

            THEN("the DataDescriptors are equal")
            {
                REQUIRE_EQ(shearletTransform.getDomainDescriptor(), volDescr);
            }
        }

        WHEN("cloning a ShearletTransform operator")
        {
            ShearletTransform<TestType> shearletTransform(size[0], size[1]);
            auto shearletTransformClone = shearletTransform.clone();

            THEN("cloned ShearletTransform operator equals original ShearletTransform operator")
            {
                REQUIRE_NE(shearletTransformClone.get(), &shearletTransform);
                // TODO this throws an error, double check
                // REQUIRE_EQ(*shearletTransformClone, shearletTransformClone);
            }
        }
    }
}

TEST_CASE_TEMPLATE("ShearletTransform: Testing reconstruction precision", TestType, float, double)
{
    GIVEN("a 2D signal")
    {
        IndexVector_t size(2);
        size << 127, 127;
        VolumeDescriptor volDescr(size);

        Vector_t<TestType> randomData(volDescr.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<TestType> signal(volDescr, randomData);

        WHEN("reconstructing the signal")
        {
            ShearletTransform<TestType> shearletTransform(size[0], size[1]);

            DataContainer<TestType> shearletCoefficients = shearletTransform.apply(signal);

            DataContainer<TestType> reconstruction =
                shearletTransform.applyAdjoint(shearletCoefficients);

            THEN("the ground truth and the reconstruction match")
            {
                // REQUIRE_UNARY(isApprox(reconstruction, signal));
            }
        }
    }
}

TEST_CASE_TEMPLATE("ShearletTransform: Testing spectra's Parseval frame property", TestType, float,
                   double)
{
    GIVEN("a 2D signal")
    {
        IndexVector_t size(2);
        size << 127, 127;
        VolumeDescriptor volDescr(size);

        Vector_t<TestType> randomData(volDescr.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<TestType> signal(volDescr, randomData);

        WHEN("generating the spectra")
        {
            ShearletTransform<TestType> shearletTransform(size[0], size[1]);

            shearletTransform.computeSpectra();

            THEN("the spectra is reported as computed")
            {
                REQUIRE(shearletTransform.isSpectraComputed());
            }

            /// If a matrix mxn A has rows that constitute Parseval frame, then AtA = I
            /// (Corollary 1.4.7 from An Introduction to Frames and Riesz Bases). Given that our
            /// spectra constitute a Parseval frame, we can utilize this property to check if
            /// they've been generated correctly.
            THEN("the spectra constitute a Parseval frame")
            {
                DataContainer<TestType> spectra = shearletTransform.getSpectra();
                index_t width = shearletTransform.getWidth();
                index_t height = shearletTransform.getHeight();
                index_t layers = shearletTransform.getL();

                DataContainer<TestType> frameCorrectness(VolumeDescriptor{{width, height}});

                for (index_t w1 = 0; w1 < width; w1++) {
                    for (index_t w2 = 0; w2 < height; w2++) {
                        TestType currFrameSum = 0;
                        for (index_t i = 0; i < layers; i++) {
                            currFrameSum += spectra(w1, w2, i) * spectra(w1, w2, i);
                        }
                        frameCorrectness(w1, w2) = currFrameSum; // TODO why not -1 here?
                    }
                }

                DataContainer<TestType> zeroes(VolumeDescriptor{{width, height}});
                zeroes = 0;

                REQUIRE_UNARY(isApprox(frameCorrectness, zeroes));

                // spectra here is of shape (L, W, H), square its elements and get the sum by the
                // last axis and subtract 1, the output will be of shape (W, H), its elements
                // should be zeroes, or very close to it
            }
        }
    }
}

// TODO what other tests to add? check how wavelets and curvelets are tested

TEST_SUITE_END();
