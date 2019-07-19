#include "EllipseGenerator.h"
#include "Timer.h"
#include "Logger.h"

#include <cmath>
#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    void EllipseGenerator<data_t>::drawFilledEllipse2d(DataContainer<data_t>& dc, data_t amplitude, Vec2 center, Vec2 sizes, data_t angle)
    {
        // sanity check
        if (dc.getDataDescriptor().getNumberOfDimensions() != 2)
            throw std::invalid_argument("EllipseGenerator::drawFilledEllipse2d: can only work on 2d DataContainers");

        // convert to radians
        auto angleRad = angle * pi / 180.0;

        // special case: circle or no rotation
        if (sizes[0] == sizes[1] || std::fmod(angle, 180.0) == 0) {
            drawShearedFilledEllipse2d(dc, amplitude, center, sizes, {1, 0});
            return;
        }

        // convert rotation by angle into shearing
        data_t theta = std::atan2(-sizes[1] * std::tan(angleRad), sizes[0]);

        Vec2 shear;
        shear[0] = std::floor((sizes[0] * std::cos(theta) * std::cos(angleRad)) - (sizes[1] * std::sin(theta) * std::sin(angleRad)));
        shear[1] = std::floor((sizes[0] * std::cos(theta) * std::sin(angleRad)) + (sizes[1] * std::sin(theta) * std::cos(angleRad)));

        Vec2 shearedSizes;
        shearedSizes[0] = std::abs(shear[0]);
        shearedSizes[1] = sizes[1] * sizes[0] / shearedSizes[0];

        drawShearedFilledEllipse2d(dc, amplitude, center, shearedSizes, shear);
    }


    template <typename data_t>
    void EllipseGenerator<data_t>::drawFilledEllipsoid3d(DataContainer<data_t>& dc, data_t amplitude, Vec3 center, Vec3 sizes,
            data_t phi, data_t theta, data_t psi)
    {
        // sanity check
        if (dc.getDataDescriptor().getNumberOfDimensions() != 3)
            throw std::invalid_argument("EllipseGenerator::drawFilledEllipsoid3d: can only work on 3d DataContainers");


        // enables small optimizations
        bool hasRotation = ( std::abs(phi) + std::abs(theta) + std::abs(psi) ) > 0;

        // convert to radians
        auto phiRad = phi * pi / 180.0;
        auto thetaRad = theta * pi / 180.0;
        auto psiRad = psi * pi / 180.0;

        auto cosPhi   = std::cos(phiRad);   auto sinPhi   = std::sin(phiRad);
        auto cosTheta = std::cos(thetaRad); auto sinTheta = std::sin(thetaRad);
        auto cosPsi   = std::cos(psiRad);   auto sinPsi   = std::sin(psiRad);

        // setup ZXZ Euler rotation matrix
        Eigen::Matrix<data_t, 3, 3> R;
        R(0,0) = cosPhi * cosPsi - cosTheta * sinPhi * sinPsi;
        R(0,1) = cosPsi * sinPhi + cosPhi * cosTheta * sinPsi;
        R(0,2) = sinTheta * sinPsi;

        R(1,0) = -cosPhi * sinPsi - cosTheta * cosPsi * sinPhi;
        R(1,1) = cosPhi * cosTheta * cosPsi - sinPhi * sinPsi;
        R(1,2) = cosPsi * sinTheta;

        R(2,0) = sinPhi * sinTheta;
        R(2,1) = -cosPhi * sinTheta;
        R(2,2) = cosTheta;

        // enables safe early abort
        index_t maxSize = sizes.maxCoeff();

        // precomputations
        index_t asq = sizes[0] * sizes[0];
        index_t bsq = sizes[1] * sizes[1];
        index_t csq = sizes[2] * sizes[2];

        IndexVector_t idx(3);

        // loop over everything... (very inefficient!)
        for (index_t x = 0; x < dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0]; ++x) {
            if (x < center[0] - maxSize || x > center[0] + maxSize) continue;
            idx[0] = x;

            for (index_t y = 0; y < dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1]; ++y) {
                if (y < center[1] - maxSize || y > center[1] + maxSize) continue;
                idx[1] = dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1] - 1 - y; // flip y axis

                for (index_t z = 0; z < dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[2]; ++z) {
                    if (z < center[2] - maxSize || z > center[2] + maxSize) continue;
                    idx[2] = z;

                    // center it
                    index_t xc = x - center[0];
                    index_t yc = y - center[1];
                    index_t zc = z - center[2];

                    // check ellipsoid equation
                    data_t aPart = (hasRotation) ? xc * R(0, 0) + yc * R(0, 1) + zc * R(0, 2) : xc;
                    aPart *= aPart / asq;

                    data_t bPart = (hasRotation) ? xc * R(1, 0) + yc * R(1, 1) + zc * R(1, 2) : yc;
                    bPart *= bPart / bsq;

                    data_t cPart = (hasRotation) ? xc * R(2, 0) + yc * R(2, 1) + zc * R(2, 2) : zc;
                    cPart *= cPart / csq;

                    if (aPart + bPart + cPart <= 1.0)
                        dc(idx) += amplitude;
                }
            }
        }
    }



    template <typename data_t>
    void EllipseGenerator<data_t>::drawShearedFilledEllipse2d(DataContainer<data_t>& dc, data_t amplitude, Vec2 center,
            Vec2 sizes, Vec2 shear)
    {
        auto twoSizeXSquared = 2 * sizes[0] * sizes[0];
        auto twoSizeYSquared = 2 * sizes[1] * sizes[1];

        // setup first ellipse part where major axis of "advance" is the y axis
        auto x = sizes[0];
        auto y = 0;

        auto xChange = sizes[1] * sizes[1] * (1 - 2 * sizes[0]);
        auto yChange = sizes[0] * sizes[0];

        index_t ellipseError = 0;
        auto xStop = twoSizeYSquared * sizes[0];
        index_t yStop = 0;

        // draw the first ellipse part
        while (xStop >= yStop) {
            drawShearedLinePairs2d(dc, amplitude, center, x, y, shear);
            y += 1;
            yStop += twoSizeXSquared;
            ellipseError += yChange;
            yChange += twoSizeXSquared;

            // check if x update is necessary
            if ( (2 * ellipseError + xChange) > 0) {
                x -= 1;
                xStop -= twoSizeYSquared;
                ellipseError += xChange;
                xChange += twoSizeYSquared;
            }
        }

        // setup second ellipse part where major axis of "advance" is the x axis
        x = 0;
        y = sizes[1];

        xChange = sizes[1] * sizes[1];
        yChange = sizes[0] * sizes[0] * (1 - 2 * sizes[1]);

        ellipseError = 0;
        xStop = 0;
        yStop = twoSizeXSquared * sizes[1];

        // draw the second ellipse part
        while (xStop < yStop) {
            x += 1;
            xStop += twoSizeYSquared;
            ellipseError += xChange;
            xChange += twoSizeYSquared;

            // check if y update is necessary
            if ( (2 * ellipseError + yChange) > 0) {
                // we only draw once the y axis is updated, to avoid line overlays (since we draw lines along x axis),
                // else we would have multiple lines stacking up the amplitude (which is additive)
                drawShearedLinePairs2d(dc, amplitude, center, x-1, y, shear);

                y -= 1;
                yStop -= twoSizeXSquared;
                ellipseError += yChange;
                yChange += twoSizeXSquared;
            }
        }
    }

    template <typename data_t>
    void EllipseGenerator<data_t>::drawShearedLinePairs2d(DataContainer<data_t>& dc, data_t amplitude, Vec2 center,
            index_t xOffset, index_t yOffset, Vec2 shear)
    {
        IndexVector_t coord(2);

        // draw the line along the x axis
        for (index_t x = center[0] - xOffset; x <= center[0] + xOffset; ++x) {
            auto shearTerm = (x - center[0]) * shear[1] / shear[0];
            coord[0] = x;
            coord[1] = center[1] + yOffset + shearTerm;
            // flip y axis
            coord[1] = dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1] - coord[1];

            // bounds check coord just to be sure (we're not performance critical here anyway)
            if (coord[0] < 0 || coord[0] >= dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0])
                throw std::invalid_argument("EllipseGenerator::drawShearedLinePairs2d: drawing coordinate (x) out of bounds");
            if (coord[1] < 0 || coord[1] >= dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1])
                throw std::invalid_argument("EllipseGenerator::drawShearedLinePairs2d: drawing coordinate (y) out of bounds");

            dc(coord) += amplitude;

            if (yOffset != 0) {
                coord[1] = center[1] - yOffset + shearTerm;
                // flip y axis
                coord[1] = dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1] - coord[1];

                if (coord[1] < 0 || coord[1] >= dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1])
                    throw std::invalid_argument("EllipseGenerator::drawShearedLinePairs2d: drawing coordinate (y) out of bounds");

                dc(coord) += amplitude;
            }
        }
    }


    // ------------------------------------------
    // explicit template instantiation
    template class EllipseGenerator<float>;
    template class EllipseGenerator<double>;

} // namespace elsa
