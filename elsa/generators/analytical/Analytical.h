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
    std::tuple<Position<data_t>, data_t, data_t, data_t>
        rescale(const VolumeDescriptor& imageDescriptor, data_t x0, data_t y0, data_t a, data_t b,
                data_t phi)
    {
        auto px = imageDescriptor.getNumberOfCoefficientsPerDimension()[0];

        return std::tuple{getPosition(imageDescriptor, x0, y0), a * px / 2.0, b * px / 2.0,
                          -phi / 180.0
                              * pi<data_t>}; // Invert angle due to parity mistake somewhere
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

        using std::get;

        for (auto [A, a, b, c, x0, y0, z0, phi, theta, psi] :
             modifiedSheppLoganParameters<data_t>) {
            auto S = rescale(imageDescriptor, x0, y0, a, b, phi);
            sheppLogan += 100 * Ellipse<data_t>{A, get<0>(S), get<1>(S), get<2>(S), get<3>(S)};
        }

        return sheppLogan.makeSinogram(sinogramDescriptor);
    }

}; // namespace elsa::phantoms