#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "Patchifier.h"
#include "DictionaryLearningProblem.h"
#include "KSVD.h"

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

        DataContainer<data_t> denoise(const DataContainer<data_t>& image);

        DataContainer<data_t> train(const DataContainer<data_t>& image, float downSampleFactor = 1);

        // void train(const std::vector<DataContainer<data_t>>& images);

    private:
        std::unique_ptr<DictionaryLearningProblem<data_t>> _problem;
        index_t _patchSize;
        index_t _stride;
        index_t _sparsityLevel;
        index_t _nAtoms;
        index_t _nIterations;
        data_t _epsilon;

        static DataContainer<data_t> downSample(const DataContainer<data_t>& patches,
                                                float downSampleFactor);
    };
} // namespace elsa
