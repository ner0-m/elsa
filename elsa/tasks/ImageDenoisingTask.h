#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"

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
        ImageDenoisingTask(const DataContainer<data_t>& image, index_t blockSize, index_t stride,
                           index_t sparsityLevel, index_t nIterations,
                           data_t epsilon = std::numeric_limits<data_t>::epsilon());

        DataContainer<data_t> denoise();

    private:
        DataContainer<data_t> _image;
        index_t _blockSize;
        index_t _stride;
        index_t _sparsityLevel;
        index_t _nIterations;
        data_t _epsilon;
    };
} // namespace elsa
