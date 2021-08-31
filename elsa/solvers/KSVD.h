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
     * TODO add meaningful description
     */
    template <typename data_t = real_t>
    class KSVD
    {
    public:
        /**
         * @brief Constructor for KSVD, accepting a dictionary representation problem, the desired
         * sparsity level of the representations and, optionally, a value for epsilon
         *
         * @param[in] problem the representation problem that is supposed to be solved
         * @param[in] sparsityLevel The number of non-zero entries in the representations
         * @param[in] epsilon affects the stopping condition
         */
        KSVD(DictionaryLearningProblem<data_t>& problem, index_t sparsityLevel,
             data_t epsilon = std::numeric_limits<data_t>::epsilon());

        /// make copy constructor deletion explicit
        KSVD(const KSVD<data_t>&) = delete;

        /**
         * @brief Solve the representation problem, i.e. apply iterations number of iterations of
         * matching pursuit
         *
         * @param[in] iterations number of iterations to execute. As OMP is a greedy algorithm, this
         * corresponds to the desired sparsity level
         *
         * @returns a reference to the current solution
         */
        DataContainer<data_t> solve(index_t iterations);

        const Dictionary<data_t>& getLearnedDictionary();

    private:
        /// variable affecting the stopping condition
        data_t _epsilon;

        /// number of samples
        index_t _nSamples;

        /// number of atoms used for each representation
        index_t _sparsityLevel;

        /// keep a reference to the DictionaryLearningProblem
        DictionaryLearningProblem<data_t>& _problem;

        auto calculateSVD(DataContainer<data_t> data);

        DataContainer<data_t>
            getNextAtom(Eigen::JacobiSVD<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> svd,
                        const DataDescriptor& atomDescriptor);

        DataContainer<data_t> getNextRepresentation(
            Eigen::JacobiSVD<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> svd);

        static index_t getNumberOfSamples(const DataContainer<data_t>& signals);
    };
} // namespace elsa
