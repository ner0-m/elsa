#pragma once

//#include "Solver.h"
#include "RepresentationProblem.h"
#include "DictionaryLearningProblem.h"
#include "OrthogonalMatchingPursuit.h"

#include <Eigen/SVD>

namespace elsa
{
    /**
     * @brief Class representing the K-Singular Value Decomposition
     *
     * @author Jonas Buerger - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * Given a DictionaryLearningProblem, KSVD finds a dictionary and representations.
     * This is done by iteratively updating representations and dictionary in the following manner:
     * 1. Find representations using Orthogonal Matching Pursuit
     * 2. Update atom by atom: Calculate the error only with respect to representations that use the
     * atom, neglecting the effect of all other atoms
     * 3. A Singular value Decomposition of the modified error matrix directly give the new atom and
     * corresponding representation
     * 4. Repeat
     */
    template <typename data_t = real_t>
    class KSVD
    {
    public:
        /**
         * @brief Constructor for KSVD, accepting a dictionary representation problem, the desired
         * sparsity level of the representations and, optionally, a value for epsilon
         *
         * @param[in] problem the dictionary learning problem that is supposed to be solved
         * @param[in] sparsityLevel The number of non-zero entries in the representations
         * @param[in] epsilon affects the stopping condition
         */
        KSVD(DictionaryLearningProblem<data_t>& problem, index_t sparsityLevel,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        KSVD(const KSVD<data_t>&) = delete;

        /**
         * @brief Solve the dictionary learning problem, i.e. apply iterations number of iterations
         * of KSVD
         *
         * @param[in] iterations number of iterations to execute.
         *
         * @returns the found representations
         */
        DataContainer<data_t> solve(index_t iterations);

        /**
         * @brief Get the learned dictionary
         *
         * @returns a reference to the dictionary as learned in solve()
         */
        const Dictionary<data_t>& getLearnedDictionary();

    private:
        /// keep a reference to the DictionaryLearningProblem
        DictionaryLearningProblem<data_t>& _problem;

        /// number of samples
        index_t _nSamples;

        /// number of atoms used for each representation
        index_t _sparsityLevel;

        /// variable affecting the stopping condition
        data_t _epsilon;

        /// calculates the SVD to a given matrix
        Eigen::JacobiSVD<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>>
            calculateSVD(const DataContainer<data_t>& data);

        /// given a SVD, generate the new atom
        DataContainer<data_t>
            getNextAtom(Eigen::JacobiSVD<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> svd,
                        const DataDescriptor& atomDescriptor);

        /// given a SVD, generate the new row vector for the representation matrix that corresponds
        /// to the current atom
        DataContainer<data_t> getNextRepresentation(
            Eigen::JacobiSVD<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> svd);

        /// helper function for obtaining the number of samples
        static index_t getNumberOfSamples(const DataContainer<data_t>& signals);
    };
} // namespace elsa
