#pragma once
#include "CUDAEllipses.h"
#include "DataContainer.h"
#include "DetectorDescriptor.h"
#include "Canvas.h"
#include "Error.h"
#include "Image.h"
#include "Sum.h"
#include "Scaling.h"
#include "Ellipse.h"

#include "StrongTypes.h"
#include "Phantoms.h"
#include "PhantomDefines.h"
#include "VolumeDescriptor.h"
#include "elsaDefines.h"
#include <Eigen/src/Core/Diagonal.h>
#include <Eigen/src/Core/DiagonalMatrix.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Geometry/AngleAxis.h>
#include <Eigen/src/Geometry/Scaling.h>
#include <Eigen/src/Geometry/Transform.h>

namespace elsa::phantoms
{

    template <typename data_t, bool useCUDA = false>
    DataContainer<data_t> analyticalSheppLogan(const VolumeDescriptor& imageDescriptor,
                                               const DetectorDescriptor& sinogramDescriptor)
    {
        auto imgDim = imageDescriptor.getNumberOfDimensions();
        auto sinoDim = sinogramDescriptor.getNumberOfDimensions();
        if (!((imgDim == 2 && sinoDim == 2) || (imgDim == 3 && sinoDim == 3))) {
            throw InvalidArgumentError("only 2/3d shepplogan supported (yet)");
        }

        const auto& coeffs = imageDescriptor.getNumberOfCoefficientsPerDimension();

        if (coeffs[0] != coeffs[1]) {
            throw InvalidArgumentError("only square shepplogan supported!");
        }

        if (imgDim == 3) {

            auto sheppLogan =
                std::conditional_t<useCUDA, CUDAEllipses<data_t, 3>, Canvas<data_t>>{};

            Eigen::Transform<data_t, 3, Eigen::Affine> scale;

            scale.linear() = Eigen::Matrix3<data_t>::Identity() * (coeffs[0] / 2.0);
            scale.translation() = imageDescriptor.getLocationOfOrigin();

            for (auto [A, a, b, c, x0, y0, z0, phi, theta, psi] :
                 modifiedSheppLoganParameters<data_t>) {

                Eigen::Vector3<data_t> center{x0, y0, z0};
                Eigen::Vector3<data_t> axes{a, b, c};

                Eigen::Matrix3<data_t> rotation;
                fillRotationMatrix({phi, theta, psi}, rotation);

                center = scale * center;
                axes = axes * (coeffs[0] / 2.0);

                sheppLogan += Ellipsoid<3, data_t>{100 * A, center, axes, rotation};
            }
            return sheppLogan.makeSinogram(sinogramDescriptor);
        } else {

            auto sheppLogan =
                std::conditional_t<useCUDA, CUDAEllipses<data_t, 2>, Canvas<data_t>>{};

            Eigen::Transform<data_t, 2, Eigen::Affine> scale;

            scale.linear() = Eigen::Matrix2<data_t>::Identity() * (coeffs[0] / 2.0);
            scale.translation() = imageDescriptor.getLocationOfOrigin();

            for (auto [A, a, b, c, x0, y0, z0, phi, theta, psi] :
                 modifiedSheppLoganParameters<data_t>) {

                Eigen::Vector2<data_t> center{x0, y0};
                Eigen::Vector2<data_t> axes{a, b};

                Eigen::Rotation2D<data_t> rotation{-phi * pi<data_t> / 180};

                center = scale * center;
                axes = axes * (coeffs[0] / 2.0);

                sheppLogan += Ellipsoid<2, data_t>{
                    100 * A, center, axes, rotation.toRotationMatrix()}; // Why do I need *100?
            }
            return sheppLogan.makeSinogram(sinogramDescriptor);
        }
    }
}; // namespace elsa::phantoms