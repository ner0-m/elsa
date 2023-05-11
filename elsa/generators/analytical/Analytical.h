#pragma once
#include "DataContainer.h"
#include "DetectorDescriptor.h"
#include "Empty.h"
#include "Error.h"
#include "Image.h"
#include "Sum.h"
#include "Scaling.h"
#include "Ellipse.h"

#include "Phantoms.h"
#include "VolumeDescriptor.h"
#include "elsaDefines.h"
#include <Eigen/src/Core/DiagonalMatrix.h>

namespace elsa::phantoms
{

    template <typename data_t>
    Position<data_t> getPosition(const VolumeDescriptor& imageDescriptor, data_t x0, data_t y0)
    {
        const auto& coeffs = imageDescriptor.getNumberOfCoefficientsPerDimension();
        return imageDescriptor.getLocationOfOrigin() + Position<data_t>{x0, y0} * coeffs[0] / 2;
    }

    template <typename data_t>
    DataContainer<data_t> analyticalSheppLogan(const VolumeDescriptor& imageDescriptor,
                                               const DetectorDescriptor& sinogramDescriptor)
    {
        if (imageDescriptor.getNumberOfDimensions() != 2
            || sinogramDescriptor.getNumberOfDimensions() != 2) {
            throw InvalidArgumentError("only 2d shepplogan supported (yet)");
        }

        const auto& coeffs = imageDescriptor.getNumberOfCoefficientsPerDimension();

        if (coeffs[0] != coeffs[1]) {
            throw InvalidArgumentError("only square shepplogan supported!");
        }

        Canvas<float> sheppLogan;

        for (auto [A, a, b, c, x0, y0, z0, phi, theta, psi] :
             modifiedSheppLoganParameters<data_t>) {
            sheppLogan +=
                100
                * Ellipse<data_t>{A, getPosition(imageDescriptor, x0, y0), a * coeffs[0] / 2,
                                  b * coeffs[1] / 2, -phi / 180 * pi<data_t>};
        }

        return sheppLogan.makeSinogram(sinogramDescriptor);
    }

}; // namespace elsa::phantoms