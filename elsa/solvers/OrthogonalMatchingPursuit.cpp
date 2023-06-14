#include "OrthogonalMatchingPursuit.h"
#include "TypeCasts.hpp"
#include "CGLS.h"

#include <iostream>

namespace elsa
{
    template <typename data_t>
    OrthogonalMatchingPursuit<data_t>::OrthogonalMatchingPursuit(const Dictionary<data_t>& D,
                                                                 const DataContainer<data_t>& y,
                                                                 data_t epsilon)
        : Solver<data_t>(), dict_(D), signal_(y), epsilon_{epsilon}
    {
    }

    template <typename data_t>
    DataContainer<data_t>
        OrthogonalMatchingPursuit<data_t>::solve(index_t iterations,
                                                 std::optional<DataContainer<data_t>> x0)
    {
        auto eval = [&](auto x) { return 0.5 * (dict_.apply(x) - signal_).squaredL2Norm(); };

        auto x = DataContainer<data_t>(dict_.getDomainDescriptor());
        if (x0.has_value()) {
            x = *x0;
        } else {
            x = 0;
        }

        IndexVector_t support(0); // the atoms used for the representation

        index_t i = 0;
        while (i < iterations && eval(x) > epsilon_) {
            auto residual = dict_.apply(x) - signal_;
            index_t k = mostCorrelatedAtom(dict_, residual);

            support.conservativeResize(support.size() + 1);
            support[support.size() - 1] = k;
            Dictionary<data_t> purgedDict = dict_.getSupportedDictionary(support);

            CGLS<data_t> cgSolver(purgedDict, signal_);
            const auto wlsSolution = cgSolver.solve(10);

            // wlsSolution has only non-zero coefficients, copy those to the full solution with zero
            // coefficients
            index_t j = 0;
            for (const auto& atomIndex : support) {
                x[atomIndex] = wlsSolution[j];
                ++j;
            }

            ++i;
        }

        return x;
    }

    template <typename data_t>
    index_t OrthogonalMatchingPursuit<data_t>::mostCorrelatedAtom(
        const Dictionary<data_t>& dict, const DataContainer<data_t>& evaluatedResidual)
    {
        // for this to work atom has to be L2-normalized
        data_t maxCorrelation = 0;
        index_t argMaxCorrelation = 0;

        for (index_t j = 0; j < dict.getNumberOfAtoms(); ++j) {
            const auto& atom = dict.getAtom(j);
            data_t correlation_j = std::abs(atom.dot(evaluatedResidual));

            if (correlation_j > maxCorrelation) {
                maxCorrelation = correlation_j;
                argMaxCorrelation = j;
            }
        }

        return argMaxCorrelation;
    }

    template <typename data_t>
    OrthogonalMatchingPursuit<data_t>* OrthogonalMatchingPursuit<data_t>::cloneImpl() const
    {
        return new OrthogonalMatchingPursuit(dict_, signal_, epsilon_);
    }

    template <typename data_t>
    bool OrthogonalMatchingPursuit<data_t>::isEqual(const Solver<data_t>& other) const
    {
        auto otherOMP = downcast_safe<OrthogonalMatchingPursuit>(&other);
        if (!otherOMP)
            return false;

        if (epsilon_ != otherOMP->epsilon_)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class OrthogonalMatchingPursuit<float>;
    template class OrthogonalMatchingPursuit<double>;

} // namespace elsa
