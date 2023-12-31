#include "EllipseGenerator.h"
#include "Timer.h"
#include "Logger.h"

#include <cmath>
#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    void EllipseGenerator<data_t>::drawFilledEllipse2d(DataContainer<data_t>& dc, data_t amplitude,
                                                       Vec2 const& center, Vec2 sizes, data_t angle)
    {
        // sanity check
        if (dc.getDataDescriptor().getNumberOfDimensions() != 2)
            throw InvalidArgumentError(
                "EllipseGenerator::drawFilledEllipse2d: can only work on 2d DataContainers");

        // don't draw anything if size is 0
        if (sizes[0] == 0 && sizes[1] == 0)
            return;

        // convert to radians
        auto angleRad = angle * pi<double> / 180.0f;

        // special case: circle or no rotation
        if (sizes[0] == sizes[1] || std::fmod(angle, 180.0) == 0) {
            drawShearedFilledEllipse2d(dc, amplitude, center, sizes, {1, 0});
            return;
        }

        // convert rotation by angle into shearing
        auto theta = std::atan2(static_cast<real_t>(-sizes[1]) * std::tan(angleRad), sizes[0]);

        Vec2 shear;
        shear[0] = static_cast<index_t>(
            std::floor((static_cast<real_t>(sizes[0]) * std::cos(theta) * std::cos(angleRad))
                       - (static_cast<real_t>(sizes[1]) * std::sin(theta) * std::sin(angleRad))));
        shear[1] = static_cast<index_t>(
            std::floor((static_cast<real_t>(sizes[0]) * std::cos(theta) * std::sin(angleRad))
                       + (static_cast<real_t>(sizes[1]) * std::sin(theta) * std::cos(angleRad))));

        Vec2 shearedSizes;
        shearedSizes[0] = std::abs(shear[0]);
        shearedSizes[1] = sizes[1] * sizes[0] / shearedSizes[0];

        drawShearedFilledEllipse2d(dc, amplitude, center, shearedSizes, shear);
    }

    template <typename data_t>
    void EllipseGenerator<data_t>::drawShearedFilledEllipse2d(DataContainer<data_t>& dc,
                                                              data_t amplitude, Vec2 const& center,
                                                              Vec2 sizes, Vec2 const& shear)
    {
        auto twoSizeXSquared = 2 * sizes[0] * sizes[0];
        auto twoSizeYSquared = 2 * sizes[1] * sizes[1];

        // setup first ellipse part where major axis of "advance" is the y axis
        auto x = sizes[0];
        index_t y = 0;

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
            if ((2 * ellipseError + xChange) > 0) {
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
            if ((2 * ellipseError + yChange) > 0) {
                // we only draw once the y axis is updated, to avoid line overlays (since we draw
                // lines along x axis), else we would have multiple lines stacking up the amplitude
                // (which is additive)
                drawShearedLinePairs2d(dc, amplitude, center, x - 1, y, shear);

                y -= 1;
                yStop -= twoSizeXSquared;
                ellipseError += yChange;
                yChange += twoSizeXSquared;
            }
        }
    }

    template <typename data_t>
    void EllipseGenerator<data_t>::drawShearedLinePairs2d(DataContainer<data_t>& dc,
                                                          data_t amplitude, Vec2 center,
                                                          index_t xOffset, index_t yOffset,
                                                          Vec2 shear)
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
            if (coord[0] < 0
                || coord[0] >= dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0])
                throw InvalidArgumentError("EllipseGenerator::drawShearedLinePairs2d: drawing "
                                           "coordinate (x) out of bounds");
            if (coord[1] < 0
                || coord[1] >= dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1])
                throw InvalidArgumentError("EllipseGenerator::drawShearedLinePairs2d: drawing "
                                           "coordinate (y) out of bounds");

            dc(coord) += amplitude;

            if (yOffset != 0) {
                coord[1] = center[1] - yOffset + shearTerm;
                // flip y axis
                coord[1] =
                    dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1] - coord[1];

                if (coord[1] < 0
                    || coord[1] >= dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1])
                    throw InvalidArgumentError("EllipseGenerator::drawShearedLinePairs2d: drawing "
                                               "coordinate (y) out of bounds");

                dc(coord) += amplitude;
            }
        }
    }

    // ------------------------------------------
    // explicit template instantiation
    template class EllipseGenerator<float>;
    template class EllipseGenerator<double>;

} // namespace elsa
