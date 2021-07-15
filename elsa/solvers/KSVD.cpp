#include "KSVD.h"

namespace elsa
{
    template <typename data_t>
    KSVD<data_t>::KSVD(DictionaryLearningProblem<data_t>& problem, index_t sparsityLevel,
                       data_t epsilon)
        : /*Solver<data_t>(problem),*/ _problem{problem},
          _sparsityLevel{sparsityLevel},
          _epsilon{epsilon},
          _nSamples{getNumberOfSamples(problem.getSignals())}
    {
    }

    template <typename data_t>
    index_t KSVD<data_t>::getNumberOfSamples(const DataContainer<data_t>& signals)
    {
        const auto& signalsDescriptor =
            dynamic_cast<const IdenticalBlocksDescriptor&>(signals.getDataDescriptor());
        return signalsDescriptor.getNumberOfBlocks();
    }

    template <typename data_t>
    DataContainer<data_t>& KSVD<data_t>::solve(index_t iterations)
    {
        return solveImpl(iterations);
    }

    template <typename data_t>
    Dictionary<data_t>& KSVD<data_t>::getLearnedDictionary()
    {
        return _problem.getCurrentDictionary();
    }

    template <typename data_t>
    DataContainer<data_t>& KSVD<data_t>::solveImpl(index_t iterations)
    {
        auto& dict = _problem.getCurrentDictionary();
        auto& representations = _problem.getCurrentRepresentations();
        auto signals = _problem.getSignals();

        Logger::get("KSVD")->info("Started for {} iterations, with {} signals and {} atoms. "
                                  "Stopping error: {}, initial error: {}",
                                  iterations, _nSamples, dict.getNumberOfAtoms(), _epsilon,
                                  _problem.getGlobalError().l2Norm());

        index_t i = 0;
        while (i < iterations && _problem.getGlobalError().l2Norm() >= _epsilon) {
            // first find a sparse representation
            for (index_t j = 0; j < _nSamples; ++j) {
                RepresentationProblem reprProblem(dict, signals.getBlock(j));
                OrthogonalMatchingPursuit omp(reprProblem);
                representations.getBlock(j) = omp.solve(_sparsityLevel);
            }
            _problem.updateError();

            // then optimize atom by atom
            for (index_t k = 0; k < dict.getNumberOfAtoms(); ++k) {
                auto affectedSignals = getAffectedSignals(representations, k);
                if (affectedSignals.size() == 0)
                    continue;
                auto modifiedError = _problem.getRestrictedError(affectedSignals, k);
                auto svd = calculateSVD(modifiedError);
                dict.updateAtom(k, getNextAtom(svd));
                updateRepresentations(representations, getNextRepresentation(svd), affectedSignals,
                                      k);
                _problem.updateError();
                if (_problem.getGlobalError().l2Norm() < _epsilon)
                    break;
            }

            Logger::get("KSVD")->info("Error after iteration {}: {}", i,
                                      _problem.getGlobalError().l2Norm());
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
    auto KSVD<data_t>::calculateSVD(DataContainer<data_t> error)
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

        return errorMatrix.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    }

    template <typename data_t>
    DataContainer<data_t> KSVD<data_t>::getNextAtom(
        Eigen::JacobiSVD<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> svd)
    {
        auto firstLeft = svd.matrixU().col(0);
        DataContainer<data_t> firstLeftSingular(VolumeDescriptor({firstLeft.rows()}), firstLeft);
        return firstLeftSingular;
    }

    template <typename data_t>
    DataContainer<data_t> KSVD<data_t>::getNextRepresentation(
        Eigen::JacobiSVD<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> svd)
    {
        auto firstRight = svd.matrixV().col(0);
        DataContainer<data_t> firstRightSingular(VolumeDescriptor({firstRight.rows()}), firstRight);
        return firstRightSingular * svd.singularValues()[0];
    }

    template <typename data_t>
    void KSVD<data_t>::updateRepresentations(DataContainer<data_t>& representations,
                                             DataContainer<data_t> nextRepresentation,
                                             IndexVector_t affectedSignals, index_t atom)
    {
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
