#pragma once

#include "Dictionary.h"

namespace elsa
{
    /**
     * @brief Class representing a sparse representation problem.
     *
     * @author Jonas Buerger - initial code
     *
     * @tparam data_t data type for the domain and range of the problem, defaulting to real_t
     *
     * This class represents a sparse representation problem, i.e.
     * \f$ \min_x \|\| x \_\|_0 \f$, s.t. \f$ \| y - Dx \| < \epsilon \f$, where \f$ D \f$ is a
     * dictionary operator, \f$ y \f$ is the signal that should be represented, \f$ x \f$ is the
     * sparse representation vector and \f$ \epsilon \f$ is some error bound.
     */
    template <typename data_t = real_t>
    class DictionaryLearningProblem
    {
    public:
        /**
         * @brief Constructor for the representation problem, accepting D, x, and an initial guess
         * x0
         *
         * @param[in] D dictionary operator
         * @param[in] y signal that should be sparsely represented
         * @param[in] x0 initial value for the current estimated solution
         */
        /* we probably don't need a constructor with an initial guess, at least not for OMP...
        RepresentationProblem(const Dictionary<data_t>& D, const DataContainer<data_t>& y,
                              const DataContainer<data_t>& x0);
        */

        /**
         * @brief Constructor for the representation problem, accepting D and y
         *
         * @param[in] D dictionary operator
         * @param[in] y signal that should be sparsely represented
         */
        DictionaryLearningProblem(const DataContainer<data_t>& signals, index_t nAtoms);

        /// default destructor
        //~DictionaryLearningProblem() override = default;

        Dictionary<data_t>& getCurrentDictionary();

        DataContainer<data_t>& getCurrentRepresentations();

        DataContainer<data_t> getSignals();

        DataContainer<data_t> getGlobalError();

        DataContainer<data_t> getRestrictedError(index_t atom);

        void updateError();

        /*
            protected:
                /// implement the polymorphic clone operation
                RepresentationProblem<data_t>* cloneImpl() const override;

                /// override getGradient and throw exception if called
                void getGradientImpl(DataContainer<data_t>& result) override;
        */

    private:
        Dictionary<data_t> _dictionary;
        DataContainer<data_t> _representations;
        const DataContainer<data_t> _signals;
        DataContainer<data_t> _residual;
    };
} // namespace elsa
