#include "KSVD.h"

namespace elsa
{
    template <typename data_t>
    KSVD<data_t>::KSVD(/*const*/ DictionaryLearningProblem<data_t>& problem, data_t epsilon)
        : /*Solver<data_t>(problem),*/ _problem{problem},
          _epsilon{epsilon},
          _nSamples{getNumberOfSamples(problem.getSignals())},
          _firstLeftSingular(VolumeDescriptor({problem.getCurrentDictionary().getNumberOfAtoms()})),
          _firstRightSingular(VolumeDescriptor{1})
    {
    }

    template <typename data_t>
    static index_t getNumberOfSamples(const DataContainer<data_t>& signals)
    {
        const auto& signalsDescriptor =
            dynamic_cast<const IdenticalBlocksDescriptor&>(signals.getDataDescriptor());
        return signalsDescriptor.getNumberOfBlocks();
    }

    template <typename data_t>
    DataContainer<data_t>& KSVD<data_t>::solveImpl(index_t iterations)
    {
        auto& dict = _problem.getCurrentDictionary();
        auto& representations = _problem.getCurrentRepresentations();
        auto signals = _problem.getSignals();

        index_t i = 0;
        while (i < iterations && _problem.getGlobalError().l2Norm() >= _epsilon) {
            // first find a sparse representation
            for (index_t j = 0; j < _nSamples; ++j) {
                RepresentationProblem reprProblem(dict, signals.getBlock(i));
                OMP omp(reprProblem);
                representations.getBlock(j) = omp.solve(3); // sparsity level needs to be set here
            }

            // then optimize atom by atom
            for (index_t k = 0; k < dict.getNumberOfAtoms(); ++k) {
                auto affectedSignals = getAffectedSignals(representations, k);
                auto modifiedError = _problem.getRestrictedError(affectedSignals, k);
                calculateSVD(modifiedError);
                dict.updateAtom(k, _firstLeftSingular);
                updateRepresentations(representations, affectedSignals, k);
                _problem.updateError();
                if (_problem.getGlobalError().l2Norm() < _epsilon)
                    break;
            }

            ++i;
        }
        return representations;
    }

    template <typename data_t>
    IndexVector_t KSVD<data_t>::getAffectedSignals(const DataContainer<data_t>& representations,
                                                   index_t atom)
    {
        IndexVector_t affectedSignals(0);

        for (index_t i = 0; i < _nSamples; ++i) {
            if (representations.getBlock(i)[atom] != 0) {
                affectedSignals.conservativeResize(affectedSignals.size() + 1);
                affectedSignals[affectedSignals.size() - 1] = i;
            }
        }
        return affectedSignals;
    }

    template <typename data_t>
    void KSVD<data_t>::calculateSVD(DataContainer<data_t> error)
    {
        const auto& errorDescriptor =
            dynamic_cast<const IdenticalBlocksDescriptor&>(error.getDataDescriptor());
        const index_t nBlocks = errorDescriptor.getNumberOfBlocks();
        const index_t nCoeffs = errorDescriptor.getDescriptorOfBlock(0).getNumberOfCoefficients();
        Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> errorMatrix(nCoeffs, nBlocks);

        for (index_t i = 0; i < nBlocks; ++i) {
            for (index_t j = 0; j < nCoeffs; ++j) {
                errorMatrix(j, i) = error.getBlock(i)[j];
            }
        }

        Eigen::JacobiSVD svd(errorMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        _firstLeftSingular =
            DataContainer<data_t>(VolumeDescriptor({nCoeffs}), svd.matrixU().col(0));
        _firstSingularValue = svd.singularValues()[0];
        _firstRightSingular =
            DataContainer<data_t>(VolumeDescriptor({nBlocks}), svd.matrixV().col(0));
    }

    template <typename data_t>
    void KSVD<data_t>::updateRepresentations(DataContainer<data_t>& representations,
                                             IndexVector_t affectedSignals, index_t atom)
    {
        DataContainer<data_t> nextRepresentation = _firstSingularValue * _firstRightSingular;

        index_t i = 0;
        for (auto idx : affectedSignals) {
            representations.getBlock(idx)[atom] = nextRepresentation[i];
            ++i;
        }
    }
    // ------------------------------------------
    // explicit template instantiation
    template class KSVD<float>;
    template class KSVD<double>;

} // namespace elsa
