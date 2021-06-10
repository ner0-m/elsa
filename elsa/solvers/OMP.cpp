#include "OMP.h"

namespace elsa
{
    template <typename data_t>
    OMP<data_t>::OMP(const RepresentationProblem<data_t>& problem, data_t epsilon)
        : Solver<data_t>(problem), _epsilon{epsilon}
    {
    }

    template <typename data_t>
    DataContainer<data_t>& OMP<data_t>::solveImpl(index_t iterations)
    {
        const auto& reprProblem = dynamic_cast<RepresentationProblem<data_t>&>(*_problem);

        const auto& dict = reprProblem.getDictionary();
        const auto& residual = _problem->getDataTerm().getResidual();
        auto& currentRepresentation = _problem->getCurrentSolution();

        IndexVector_t support(0); // the atoms used for the representation
        currentRepresentation = 0;

        index_t i = 0;
        while (i < iterations && _problem->evaluate() > _epsilon) {
            index_t k = mostCorrelatedAtom(dict, residual.evaluate(currentRepresentation));

            support.conservativeResize(support.size() + 1);
            support[support.size() - 1] = k;
            Dictionary<data_t> purgedDict = dict.getSupportedDictionary(support);

            WLSProblem<data_t> wls(purgedDict, reprProblem.getSignal());

            CG cgSolver(wls);
            const auto& wlsSolution = cgSolver.solve(10);

            // wlsSolution has only non-zero coefficients, copy those to the full solution with zero
            // coefficients
            index_t j = 0;
            for (const auto& atomIndex : support) {
                currentRepresentation[atomIndex] = wlsSolution[j];
                ++j;
            }

            ++i;
        }

        return getCurrentSolution();
    }

    template <typename data_t>
    index_t OMP<data_t>::mostCorrelatedAtom(const Dictionary<data_t>& dict,
                                            const DataContainer<data_t>& evaluatedResidual)
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
    OMP<data_t>* OMP<data_t>::cloneImpl() const
    {
        const auto& reprProblem = dynamic_cast<RepresentationProblem<data_t>&>(*_problem);
        return new OMP(reprProblem, _epsilon);
    }

    template <typename data_t>
    bool OMP<data_t>::isEqual(const Solver<data_t>& other) const
    {
        if (!Solver<data_t>::isEqual(other))
            return false;

        auto otherOMP = dynamic_cast<const OMP*>(&other);
        if (!otherOMP)
            return false;

        if (_epsilon != otherOMP->_epsilon)
            return false;

        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class OMP<float>;
    template class OMP<double>;

} // namespace elsa
