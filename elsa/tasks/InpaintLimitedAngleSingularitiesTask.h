#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "StrongTypes.h"
#include "ShearletTransform.h"
#include "PhantomNet.h"

namespace elsa
{
    /**
     * @brief Class representing a task of inpainting missing singularities (coefficients) in
     * limited-angle scans
     *
     * @author Andi Braimllari - initial code
     *
     * @tparam data_t data type for the domain and range of the SDLX task, defaulting to real_t
     */
    template <typename data_t = real_t>
    class InpaintLimitedAngleSingularitiesTask // TODO what's a better name here?
    {
    public:
        InpaintLimitedAngleSingularitiesTask() = default;

        // generate LA sinogram for a given image
        // reconstruct using ADMM based on a LA sinogram
        // train PhantomNet on the ground truths and reconstructions
        // combine visible coeffs. to the inpainted invis. ones

        void generateLimitedAngleSinogram(
            DataContainer<data_t> image,
            std::pair<elsa::geometry::Degree, elsa::geometry::Degree> missingWedgeAngles,
            index_t numOfAngles = 180, index_t arc = 360);

        void reconstructOnLimitedAngleSinogram(DataContainer<data_t> limitedAngleSinogram,
                                               index_t numberOfScales,
                                               index_t solverIterations = 50, data_t rho1 = 1 / 2,
                                               data_t rho2 = 1);

        void trainPhantomNet(const std::vector<DataContainer<data_t>>& x,
                             const std::vector<DataContainer<data_t>>& y, index_t epochs,
                             index_t batchSize = 16);

        DataContainer<data_t>
            combineVisCoeffsToInpaintedInvisCoeffs(DataContainer<data_t> visCoeffs,
                                                   DataContainer<data_t> invisCoeffs);

    private:
        ml::PhantomNet _phantomNet;
    };
} // namespace elsa
