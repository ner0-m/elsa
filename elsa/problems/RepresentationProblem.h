#pragma once

#include "Dictionary.h"
#include "Problem.h"
#include "LinearResidual.h"
#include "L2NormPow2.h"
#include <memory>

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
     * \f$ \min_x \| x \|_0 \f$, s.t. \f$ \| y - Dx \|_2^2 < \epsilon \f$, where \f$ D \f$ is a
     * dictionary operator, \f$ y \f$ is the signal that should be represented, \f$ x \f$ is the
     * sparse representation vector and \f$ \epsilon \f$ is some error bound.
     * The sparsity condition \f$ \min_x \|\| x \|\|_0 \f$ is not inforced by the class but handled
     * implicitly, either by using a greedy algorithm that starts with the 0-vector as a
     * representation or by creating another problem that explicitly takes sparsity into account and
     * relaxes the L0-Norm to the L1-Norm.
     */
    template <typename data_t = real_t>
    class RepresentationProblem : public Problem<data_t>
    {
    public:
        /**
         * @brief Constructor for the representation problem, accepting D and y
         *
         * @param[in] D dictionary operator
         * @param[in] y signal that should be sparsely represented
         */
        RepresentationProblem(const Dictionary<data_t>& D, const DataContainer<data_t>& y);

        /// default destructor
        ~RepresentationProblem() override = default;

        const Dictionary<data_t>& getDictionary() const;

        const DataContainer<data_t>& getSignal() const;

    protected:
        /// implement the polymorphic clone operation
        RepresentationProblem<data_t>* cloneImpl() const override;

        /// override getGradient and throw exception if called because L0-Norm, even though only
        /// enforced implicitly, is not differtiable
        void getGradientImpl(DataContainer<data_t>& result) override;

    private:
        std::unique_ptr<const Dictionary<data_t>> _dict;
        DataContainer<data_t> _signal;
    };
} // namespace elsa
