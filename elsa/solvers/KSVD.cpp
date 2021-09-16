#include "KSVD.h"

namespace elsa
{
    template <typename data_t>
    KSVD<data_t>::KSVD(DictionaryLearningProblem<data_t>& problem, index_t sparsityLevel,
                       data_t epsilon)
        : _problem{problem},
          _nSamples{getNumberOfSamples(problem.getSignals())},
          _sparsityLevel{sparsityLevel},
          _epsilon{epsilon}
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
    const Dictionary<data_t>& KSVD<data_t>::getLearnedDictionary()
    {
        return _problem.getDictionary();
    }

    template <typename data_t>
    DataContainer<data_t> KSVD<data_t>::solve(index_t iterations)
    {
        auto& dict = _problem.getDictionary();
        auto& representations = _problem.getRepresentations();
        auto signals = _problem.getSignals();

        Logger::get("KSVD")->info("Started for {} iterations, with {} signals and {} atoms. "
                                  "Stopping error: {}, initial error: {}",
                                  iterations, _nSamples, dict.getNumberOfAtoms(), _epsilon,
                                  _problem.getGlobalError().l2Norm());

        index_t i = 0;
        while (i < iterations && _problem.getGlobalError().l2Norm() >= _epsilon) {
            DataContainer<data_t> nextRepresentations(representations.getDataDescriptor());
            // first find a sparse representation
            for (index_t j = 0; j < _nSamples; ++j) {
                RepresentationProblem reprProblem(dict, signals.getBlock(j));
                OrthogonalMatchingPursuit omp(reprProblem);
                nextRepresentations.getBlock(j) = omp.solve(_sparsityLevel);
            }
            _problem.updateRepresentations(nextRepresentations);

            // then optimize atom by atom
            for (index_t k = 0; k < dict.getNumberOfAtoms(); ++k) {
                const auto atom = dict.getAtom(k);
                const auto& atomDescriptor =
                    atom.getDataDescriptor(); // needed for constructing a new atom
                try {
                    auto modifiedError = _problem.getRestrictedError(k);
                    auto svd = calculateSVD(modifiedError);
                    auto nextAtom = getNextAtom(svd, atomDescriptor);
                    auto nextRepresentation = getNextRepresentation(svd);
                    _problem.updateAtom(k, nextAtom, nextRepresentation);
                    if (_problem.getGlobalError().l2Norm() < _epsilon)
                        break;
                } catch (LogicError& e) {
                    // nothing bad, this atom is not used
                    continue;
                }
            }

            Logger::get("KSVD")->info("Error after iteration {}: {}", i,
                                      _problem.getGlobalError().l2Norm());
            ++i;
        }
        return representations;
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
        Eigen::JacobiSVD<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> svd,
        const DataDescriptor& atomDescriptor)
    {
        auto firstLeft = svd.matrixU().col(0);
        DataContainer<data_t> firstLeftSingular(atomDescriptor, firstLeft);
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

    // ------------------------------------------
    // explicit template instantiation
    template class KSVD<float>;
    template class KSVD<double>;

} // namespace elsa
