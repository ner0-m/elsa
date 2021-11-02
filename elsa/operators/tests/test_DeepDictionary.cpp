/**
 * @file test_Dictionary.cpp
 *
 * @brief Tests for DeepDictionary class
 *
 * @author Jonas Buerger - main code
 */

#include "doctest/doctest.h"
#include "DeepDictionary.h"
#include "IdenticalBlocksDescriptor.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("Constructing a DeepDictionary operator ", data_t, float, double)
{
    GIVEN("a descriptor for a signal, the number of atoms for each level and activation functions "
          "between the levels")
    {
        VolumeDescriptor dd({5});
        std::vector<index_t> nAtoms{10, 12, 5};
        std::vector<activation::ActivationFunction<data_t>> activations;
        activations.push_back(activation::IdentityActivation<data_t>());
        activations.push_back(activation::IdentityActivation<data_t>());

        WHEN("instantiating a DeepDictionary operator")
        {
            DeepDictionary deepDictOp(dd, nAtoms, activations);

            THEN("the DataDescriptors are as expected")
            {
                VolumeDescriptor representationDescriptor({nAtoms.back()});
                REQUIRE_EQ(deepDictOp.getDomainDescriptor(), representationDescriptor);
                REQUIRE_EQ(deepDictOp.getRangeDescriptor(), dd);
            }
        }
        /*
                WHEN("cloning a Dictionary operator")
                {
                    Dictionary dictOp(dd, nAtoms);
                    auto dictOpClone = dictOp.clone();

                    THEN("everything matches")
                    {
                        REQUIRE_NE(dictOpClone.get(), &dictOp);
                        REQUIRE_EQ(*dictOpClone, dictOp);
                    }
                }
        */
    }
    /*
        GIVEN("some initial data")
        {
            VolumeDescriptor dd({5});
            index_t nAtoms(10);
            IdenticalBlocksDescriptor ibd(nAtoms, dd);
            auto randomDictionary = generateRandomMatrix<data_t>(ibd.getNumberOfCoefficients());
            DataContainer<data_t> dict(ibd, randomDictionary);

            WHEN("instantiating an Dictionary operator")
            {
                Dictionary dictOp(dict);

                THEN("the DataDescriptors are as expected")
                {
                    VolumeDescriptor representationDescriptor({nAtoms});
                    REQUIRE_EQ(dictOp.getDomainDescriptor(), representationDescriptor);
                    REQUIRE_EQ(dictOp.getRangeDescriptor(), dd);
                }

                AND_THEN("the atoms are normalized")
                {
                    for (int i = 0; i < dictOp.getNumberOfAtoms(); ++i) {
                        REQUIRE_EQ(dictOp.getAtom(i).l2Norm(), Approx(1));
                    }
                }
            }
        }

        GIVEN("some invalid initial data")
        {
            VolumeDescriptor dd({5});
            DataContainer<data_t> invalidDict(dd);

            WHEN("instantiating an Dictionary operator")
            {
                THEN("an exception is thrown")
                {
                    REQUIRE_THROWS_AS(Dictionary{invalidDict}, InvalidArgumentError);
                }
            }
        }
    */
}

/*
TEST_CASE_TEMPLATE("Accessing Dictionary atoms ", data_t, float, double)
{
    GIVEN("some dictionary operator")
    {
        VolumeDescriptor dd({5});
        index_t nAtoms(10);
        IdenticalBlocksDescriptor ibd(nAtoms, dd);
        auto randomDictionary = generateRandomMatrix<data_t>(ibd.getNumberOfCoefficients());
        DataContainer<data_t> dict(ibd, randomDictionary);

        // normalize the atoms beforehand so we can compare
        for (int i = 0; i < nAtoms; ++i) {
            auto block = dict.getBlock(i);
            block /= block.l2Norm();
        }
        Dictionary dictOp(dict);

        WHEN("accessing an atom")
        {
            index_t i = 4;
            DataContainer<data_t> atom = dictOp.getAtom(i);
            THEN("the data is correct") { REQUIRE_EQ(atom, dict.getBlock(i)); }
        }

        WHEN("accessing an atom from a const dictionary reference")
        {
            index_t i = 4;
            const Dictionary<data_t>& constDictOp(dictOp);
            auto atom = constDictOp.getAtom(i);
            THEN("the data is correct") { REQUIRE_EQ(atom, dict.getBlock(i)); }
        }

        WHEN("accessing an atom with an invalid index")
        {
            index_t i = 42;
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(dictOp.getAtom(i), InvalidArgumentError);
            }
        }

        WHEN("updating an atom")
        {
            index_t i = 4;
            auto randomData = generateRandomMatrix<data_t>(dd.getNumberOfCoefficients());
            DataContainer<data_t> newAtom(dd, randomData);

            dictOp.updateAtom(i, newAtom);
            THEN("the data is correct (and normalized)")
            {
                REQUIRE_EQ(dictOp.getAtom(i), (newAtom / newAtom.l2Norm()));
            }
        }

        WHEN("updating an atom with itself")
        {
            // this test makes sure that a normalized atom doesn't get normalized again

            index_t i = 4;
            DataContainer<data_t> newAtom = dictOp.getAtom(i);

            dictOp.updateAtom(i, newAtom);
            THEN("the data is identical") { REQUIRE_EQ(dictOp.getAtom(i), newAtom); }
        }

        WHEN("updating an atom with an invalid index")
        {
            index_t i = 42;
            DataContainer<data_t> dummyAtom(dd);
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(dictOp.updateAtom(i, dummyAtom), InvalidArgumentError);
            }
        }

        WHEN("updating an atom with invalid data")
        {
            index_t i = 4;
            VolumeDescriptor invalidDesc({dd.getNumberOfCoefficients() + 1});
            DataContainer<data_t> invalid(invalidDesc);
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(dictOp.updateAtom(i, invalid), InvalidArgumentError);
            }
        }
    }
}

TEST_CASE_TEMPLATE("Getting the support of a dictionary ", data_t, float, double)
{

    GIVEN("some dictionary and a support vector")
    {
        VolumeDescriptor dd({5});
        const index_t nAtoms(10);
        IdenticalBlocksDescriptor ibd(nAtoms, dd);
        auto randomDictionary = generateRandomMatrix<data_t>(ibd.getNumberOfCoefficients());
        DataContainer<data_t> dict(ibd, randomDictionary);
        Dictionary dictOp(dict);

        IndexVector_t support(3);
        support << 1, 5, 7;

        WHEN("getting the support of the dictionary")
        {
            auto purgedDict = dictOp.getSupportedDictionary(support);

            THEN("the data is correct")
            {
                for (index_t i = 0; i < purgedDict.getNumberOfAtoms(); ++i) {
                    REQUIRE_EQ(purgedDict.getAtom(i), dictOp.getAtom(support[i]));
                }
            }
        }
    }
}
*/

TEST_CASE_TEMPLATE("Using the DeepDictionary ", data_t, float, double)
{

    GIVEN("some deep dictionary")
    {
        VolumeDescriptor dd({10});
        std::vector<index_t> nAtoms{4, 3, 4};
        std::vector<activation::ActivationFunction<data_t>> activations;
        activations.push_back(activation::IdentityActivation<data_t>());
        activations.push_back(activation::IdentityActivation<data_t>());

        DeepDictionary deepDictOp(dd, nAtoms, activations);

        WHEN("applying the dictionary to a representation vector")
        {
            VolumeDescriptor representationDescriptor({nAtoms.back()});
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> inputData(
                representationDescriptor.getNumberOfCoefficients());
            inputData.setRandom();
            DataContainer input(representationDescriptor, inputData);
            auto output = deepDictOp.apply(input);

            THEN("the result is as expected")
            {
                // we are happy if we dont have exceptions
                REQUIRE_UNARY(true);
            }
        }

        /*
        WHEN("applying the dictionary to a non-matching vector")
        {
            VolumeDescriptor invalidDesc({nAtoms + 1});
            DataContainer<data_t> invalid(invalidDesc);
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(dictOp.apply(invalid), InvalidArgumentError);
            }
        }
        */
    }
}
TEST_SUITE_END();
