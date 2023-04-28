#pragma once

#include <memory>

#include "DataContainer.h"
#include "LinearOperator.h"
#include "VolumeDescriptor.h"
#include "elsaDefines.h"
#include "FourierTransform.h"
#include "Filter.h"

namespace elsa
{
    template <typename data_t = float> // template <std::floating_point data_t = float>
    class FBP
    {
        /**
         * @brief Solve the inverse radon transform analytically using the filtered backprojection.
         *
         * @author Cederik Höfs - initial code
         *
         * @tparam data_t floating point type for the domain and range of the transformation,
         *                defaulting to float
         *
         * Implements the two-dimensional FBP
         * Uses Eigen::FFT with FFTW.
         */

    public:
        /**
         * @brief Construct a new FBP object
         *
         * @param P Projector from image to sinogram
         * @param g Filter, normalized to range [0,1]
         */
        explicit FBP(const LinearOperator<data_t>& P, const Filter<data_t>& g);

        /**
         * @brief perform the filtered backprojection
         * @param x inputData (sinogram)
         * @param Ax outputData (image matrix)
         */
        DataContainer<data_t> apply(const DataContainer<data_t>& sinogram) const;

    private:
        std::unique_ptr<LinearOperator<data_t>> projector_;
        std::unique_ptr<Filter<data_t>> filter_;
    };

} // namespace elsa
