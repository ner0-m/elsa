#pragma once

#include "DeepDictionary.h"
#include "elsaDefines.h"
#include "DataContainer.h"
#include "Patchifier.h"
#include "DictionaryLearningProblem.h"
#include "DeepDictionaryLearningProblem.h"
#include "KSVD.h"
#include "DeepDictSolver.h"

#include <limits>

namespace elsa
{
    /**
     * @brief Class representing a image denoising task
     * regularization term(s).
     *
     * @author Jonas Buerger - initial code
     *
     * @tparam data_t data type for the domain and range of the image denoising task, defaulting to
     * real_t
     *
     */
    template <typename data_t = real_t>
    class ImageDenoisingTask
    {
    public:
        /**
         * @brief Constructor for a image denoising task
         *
         * @param[in] dataTerm functional expressing the data term
         */
        ImageDenoisingTask(index_t patchSize, index_t stride, index_t sparsityLevel,
                           index_t nIterations, index_t nAtoms,
                           data_t epsilon = std::numeric_limits<data_t>::epsilon());

        ImageDenoisingTask(index_t patchSize, index_t stride, index_t sparsityLevel,
                           index_t nIterations, std::vector<index_t> nAtoms,
                           std::vector<ActivationFunction<data_t>> activations,
                           data_t epsilon = std::numeric_limits<data_t>::epsilon());

        ImageDenoisingTask(index_t patchSize, index_t stride, index_t sparsityLevel,
                           index_t nIterations, std::vector<index_t> nAtoms,
                           data_t epsilon = std::numeric_limits<data_t>::epsilon());

        DataContainer<data_t> denoise(const DataContainer<data_t>& image);

        DataContainer<data_t> train(const DataContainer<data_t>& image, float downSampleFactor = 1);

        // void train(const std::vector<DataContainer<data_t>>& images);

    private:
        enum class Strategy { DictLearning, DeepDictLearning };

        const Strategy _strat;

        std::unique_ptr<DictionaryLearningProblem<data_t>> _dictProblem;
        std::unique_ptr<DeepDictionaryLearningProblem<data_t>> _deepDictProblem;

        index_t _patchSize;
        index_t _stride;
        index_t _sparsityLevel;
        index_t _nAtoms;
        std::vector<index_t> _nAtomsPerLevel;
        std::vector<ActivationFunction<data_t>> _activations;
        index_t _nIterations;
        data_t _epsilon;

        DataContainer<data_t> denoiseDict(const DataContainer<data_t>& signals);
        DataContainer<data_t> denoiseDeepDict(const DataContainer<data_t>& signals);

        DataContainer<data_t> trainDict(const DataContainer<data_t>& signals);
        DataContainer<data_t> trainDeepDict(const DataContainer<data_t>& signals);

        static DataContainer<data_t> downSample(const DataContainer<data_t>& patches,
                                                float downSampleFactor);
    };
} // namespace elsa
