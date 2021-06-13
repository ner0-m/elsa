#include "KSVD.h"
#include <Eigen/src/SVD/JacobiSVD.h>

namespace elsa
{
    template <typename data_t>
    KSVD<data_t>::KSVD(const DictionaryLearningProblem<data_t>& problem, data_t epsilon)
        : Solver<data_t>(problem), _epsilon{epsilon}
    {
    }

    template <typename data_t>
    DataContainer<data_t>& KSVD<data_t>::solveImpl(index_t iterations)
    {
        auto& dict = problem.getCurrentDictionary();
        auto& representations = problem.getCurrentRepresentations();
        auto signals = problem.getSignals();

        index_t i = 0;
        while (i < iterations && problem.getGlobalError().l2Norm() > _epsilon) {
            index_t idx = 0;
            for (auto& representation : representations) {
                RepresentationProblem reprProblem(dict, signals.getBlock(idx));
                OMP omp(reprProblem);
                representation = omp.solve(3); // sparsity level needs to be set here
                ++idx;
            }

            for (index_t j = 0; j < dict.getNumberOfAtoms(); ++j) {
                auto modifiedError = problem.getRestrictedError(j);
                calculateSVD(modifiedError);
                dict.updateAtom(j /*, U[0]*/);
                // should also update representation here
            }

            problem.updateError();
            ++i;
        }
    }

    template <typename data_t>
    void KSVD<data_t>::calculateSVD(DataContainer<data_t> error)
    {
        auto errorDescriptor = error.getDataDescriptor();
        Eigen::Map<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> errorMatrix(
            &error, errorDescriptor.getDescriptorOfBlock(0).getNumberOfCoefficients(),
            errorDescriptor.getNumberOfBlocks());

        Eigen::JacobiSVD svd(errorMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    }
    // ------------------------------------------
    // explicit template instantiation
    template class OMP<float>;
    template class OMP<double>;

} // namespace elsa
