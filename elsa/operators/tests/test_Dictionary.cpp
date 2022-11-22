/**
 * @file test_Dictionary.cpp
 *
 * @brief Tests for Dictionary class
 *
 * @author Jonas Buerger - main code
 */

#include "doctest/doctest.h"
#include "Dictionary.h"
#include "IdenticalBlocksDescriptor.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("Constructing a Dictionary operator ", data_t, float, double)
{
    GIVEN("a descriptor for a signal and the number of atoms")
    {
        VolumeDescriptor dd({5});
        index_t nAtoms(10);

        WHEN("instantiating an Dictionary operator")
        {
            Dictionary dictOp(dd, nAtoms);

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
    }

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
}

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
            THEN("the data is correct")
            {
                REQUIRE_EQ(atom, dict.getBlock(i));
            }
        }

        WHEN("accessing an atom from a const dictionary reference")
        {
            index_t i = 4;
            const Dictionary<data_t>& constDictOp(dictOp);
            auto atom = constDictOp.getAtom(i);
            THEN("the data is correct")
            {
                REQUIRE_EQ(atom, dict.getBlock(i));
            }
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
            THEN("the data is identical")
            {
                REQUIRE_EQ(dictOp.getAtom(i), newAtom);
            }
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

TEST_CASE_TEMPLATE("Using the Dictionary ", data_t, float, double)
{

    GIVEN("some dictionary")
    {
        VolumeDescriptor dd({2});
        const index_t nAtoms(4);
        IdenticalBlocksDescriptor ibd(nAtoms, dd);
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> dictData(ibd.getNumberOfCoefficients());
        dictData << 1, 2, 3, 4, 5, 6, 7, 8;
        DataContainer<data_t> dict(ibd, dictData);
        /*  1,3,5,7
            2,4,6,8 */
        Dictionary dictOp(dict);

        // construct a eigen matrix corresponding to the dictionary so we can check the results
        Eigen::Map<Eigen::Matrix<data_t, Eigen::Dynamic, nAtoms>> matDictData(
            dictData.data(), dd.getNumberOfCoefficients(), nAtoms);
        for (index_t i = 0; i < matDictData.cols(); ++i) {
            matDictData.col(i).normalize();
        }

        WHEN("applying the dictionary to a representation vector")
        {
            VolumeDescriptor representationDescriptor({nAtoms});
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> inputData(
                representationDescriptor.getNumberOfCoefficients());
            inputData << 2, 3, 4, 5;
            DataContainer input(representationDescriptor, inputData);
            auto output = dictOp.apply(input);

            THEN("the result is as expected")
            {
                Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(dd.getNumberOfCoefficients());
                expected = matDictData * inputData;
                DataContainer expectedOutput(dd, expected);
                REQUIRE_UNARY(isApprox(expectedOutput, output));
            }
        }

        WHEN("applying the adjoint dictionary to a matching vector")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> inputData(dd.getNumberOfCoefficients());
            inputData << 2, 3;
            DataContainer input(dd, inputData);
            auto output = dictOp.applyAdjoint(input);

            THEN("the result is as expected")
            {
                VolumeDescriptor expectedDescriptor({nAtoms});
                Eigen::Matrix<data_t, Eigen::Dynamic, 1> expected(
                    expectedDescriptor.getNumberOfCoefficients());
                expected = matDictData.transpose() * inputData;
                DataContainer expectedOutput(expectedDescriptor, expected);
                REQUIRE_UNARY(isApprox(expectedOutput, output));
            }
        }

        WHEN("applying the dictionary to a non-matching vector")
        {
            VolumeDescriptor invalidDesc({nAtoms + 1});
            DataContainer<data_t> invalid(invalidDesc);
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(dictOp.apply(invalid), InvalidArgumentError);
            }
        }

        WHEN("applying the adjoint dictionary to a non-matching vector")
        {
            VolumeDescriptor invalidDesc({dd.getNumberOfCoefficients() + 1});
            DataContainer<data_t> invalid(invalidDesc);
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(dictOp.applyAdjoint(invalid), InvalidArgumentError);
            }
        }
    }
}
TEST_SUITE_END();
