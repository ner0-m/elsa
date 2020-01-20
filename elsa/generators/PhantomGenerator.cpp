#include "PhantomGenerator.h"
#include "EllipseGenerator.h"
#include "Logger.h"

#include <cmath>
#include <stdexcept>

namespace elsa
{
    template <typename data_t>
    DataContainer<data_t> PhantomGenerator<data_t>::createModifiedSheppLogan(IndexVector_t sizes)
    {
        // sanity check
        if (sizes.size() < 2 || sizes.size() > 3)
            throw std::invalid_argument(
                "PhantomGenerator::createModifiedSheppLogan: only 2d or 3d supported");
        if (sizes.size() == 2 && sizes[0] != sizes[1])
            throw std::invalid_argument(
                "PhantomGenerator::createModifiedSheppLogan: 2d size has to be square");
        if (sizes.size() == 3 && (sizes[0] != sizes[1] || sizes[0] != sizes[2]))
            throw std::invalid_argument(
                "PhantomGenerator::createModifiedSheppLogan: 3d size has to be cubed");

        Logger::get("PhantomGenerator")
            ->info("creating modified Shepp Logan phantom of size {}^{}", sizes[0], sizes.size());

        DataDescriptor dd(sizes);
        DataContainer<data_t> dc(dd);

        if (sizes.size() == 2) {
            EllipseGenerator<data_t>::drawFilledEllipse2d(dc, 1.0,
                                                          {scaleShift(dd, 0), scaleShift(dd, 0)},
                                                          {scale(dd, 0.69f), scale(dd, 0.92f)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, -0.8f, {scaleShift(dd, 0), scaleShift(dd, -0.0184f)},
                {scale(dd, 0.6624f), scale(dd, 0.8740f)}, 0);

            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, -0.2f, {scaleShift(dd, 0.22f), scaleShift(dd, 0)},
                {scale(dd, 0.11f), scale(dd, 0.31f)}, -18);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, -0.2f, {scaleShift(dd, -0.22f), scaleShift(dd, 0)},
                {scale(dd, 0.16f), scale(dd, 0.41f)}, 18);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, 0.1f, {scaleShift(dd, 0), scaleShift(dd, 0.35f)},
                {scale(dd, 0.21f), scale(dd, 0.25)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, 0.1f, {scaleShift(dd, 0), scaleShift(dd, 0.1f)},
                {scale(dd, 0.046f), scale(dd, 0.046f)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, 0.1f, {scaleShift(dd, 0), scaleShift(dd, -0.1f)},
                {scale(dd, 0.046f), scale(dd, 0.046f)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, 0.1f, {scaleShift(dd, -0.08f), scaleShift(dd, -0.605f)},
                {scale(dd, 0.046f), scale(dd, 0.023f)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, 0.1f, {scaleShift(dd, 0), scaleShift(dd, -0.606f)},
                {scale(dd, 0.023f), scale(dd, 0.023f)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, 0.1f, {scaleShift(dd, 0.06f), scaleShift(dd, -0.605f)},
                {scale(dd, 0.023f), scale(dd, 0.046f)}, 0);
        }

        if (sizes.size() == 3) {
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 1.0, {scaleShift(dd, 0), scaleShift(dd, 0), scaleShift(dd, 0)},
                {scale(dd, 0.69f), scale(dd, 0.92f), scale(dd, 0.81f)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, -0.8f, {scaleShift(dd, 0), scaleShift(dd, -0.0184f), scaleShift(dd, 0)},
                {scale(dd, 0.6624f), scale(dd, 0.874f), scale(dd, 0.78f)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, -0.2f, {scaleShift(dd, 0.22f), scaleShift(dd, 0), scaleShift(dd, 0)},
                {scale(dd, 0.11f), scale(dd, 0.31f), scale(dd, 0.22f)}, -18, 0, 10);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, -0.2f, {scaleShift(dd, -0.22f), scaleShift(dd, 0), scaleShift(dd, 0)},
                {scale(dd, 0.16f), scale(dd, 0.41f), scale(dd, 0.28f)}, 18, 0, 10);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 0.1f, {scaleShift(dd, 0), scaleShift(dd, 0.35f), scaleShift(dd, -0.15f)},
                {scale(dd, 0.21f), scale(dd, 0.25f), scale(dd, 0.41f)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 0.1f, {scaleShift(dd, 0), scaleShift(dd, 0.1f), scaleShift(dd, 0.25f)},
                {scale(dd, 0.046f), scale(dd, 0.046f), scale(dd, 0.05f)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 0.1f, {scaleShift(dd, 0), scaleShift(dd, -0.1f), scaleShift(dd, 0.25f)},
                {scale(dd, 0.046f), scale(dd, 0.046f), scale(dd, 0.05f)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 0.1f, {scaleShift(dd, -0.08f), scaleShift(dd, -0.605f), scaleShift(dd, 0)},
                {scale(dd, 0.046f), scale(dd, 0.023f), scale(dd, 0.05f)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 0.1f, {scaleShift(dd, 0), scaleShift(dd, -0.606f), scaleShift(dd, 0)},
                {scale(dd, 0.023f), scale(dd, 0.023f), scale(dd, 0.02f)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 0.1f, {scaleShift(dd, 0.06f), scaleShift(dd, -0.605f), scaleShift(dd, 0)},
                {scale(dd, 0.023f), scale(dd, 0.046f), scale(dd, 0.02f)}, 0, 0, 0);
        }

        return dc;
    }

    template <typename data_t>
    DataContainer<data_t>
        PhantomGenerator<data_t>::createRandomizedSheppLogan(IndexVector_t sizes,
                                                             std::mt19937_64& mersenneTwisterEngine)
    {
        // sanity check
        if (sizes.size() != 2)
            throw std::invalid_argument(
                "PhantomGenerator::createRandomizedSheppLogan: only 2d supported");
        if (sizes.size() == 2 && sizes[0] != sizes[1])
            throw std::invalid_argument(
                "PhantomGenerator::createModifiedSheppLogan: 2d size has to be square");

        // Create randomized Shepp-Logan-like phantom
        DataDescriptor dd(sizes);
        DataContainer<data_t> dc(dd);

        // Big ellipsis are not altered
        EllipseGenerator<data_t>::drawFilledEllipse2d(dc, 1.0,
                                                      {scaleShift(dd, 0), scaleShift(dd, 0)},
                                                      {scale(dd, 0.69f), scale(dd, 0.92f)}, 0);
        EllipseGenerator<data_t>::drawFilledEllipse2d(dc, -0.8f,
                                                      {scaleShift(dd, 0), scaleShift(dd, -0.0184f)},
                                                      {scale(dd, 0.6624f), scale(dd, 0.8740f)}, 0);

        // We want all small ellipsis to lay inside the bigger black ellipses E. To achieve this we
        // position all inner ellipsis inside a circle around the center of E with radius equal to
        // the smaller radius of E.

        // Construct random number generator
        std::uniform_real_distribution<data_t> dist(0, 1);

        auto randScaling =
            std::lround(1 - dist(mersenneTwisterEngine) * static_cast<data_t>(2) / 9);
        auto randAngle = std::lround(2 * 45 * (dist(mersenneTwisterEngine) - 0.5f));
        auto randTranslation1 = std::lround(0.2f * dist(mersenneTwisterEngine));
        auto randTranslation2 = std::lround(0.2f * dist(mersenneTwisterEngine));

        EllipseGenerator<data_t>::drawFilledEllipse2d(
            dc, -0.2f,
            {scaleShift(dd, randTranslation1 + 0.22f), scaleShift(dd, randTranslation2 + 0)},
            {randScaling * scale(dd, 0.11f), randScaling * scale(dd, 0.31f)}, randAngle - 18);

        randScaling = std::lround(1 - dist(mersenneTwisterEngine) * static_cast<data_t>(2) / 9);
        randAngle = std::lround(2 * 45 * (dist(mersenneTwisterEngine) - 0.5f));
        randTranslation1 = std::lround(0.2f * dist(mersenneTwisterEngine));
        randTranslation2 = std::lround(0.2f * dist(mersenneTwisterEngine));

        EllipseGenerator<data_t>::drawFilledEllipse2d(
            dc, -0.2f,
            {scaleShift(dd, randTranslation1 - 0.22f), scaleShift(dd, randTranslation2 + 0)},
            {randScaling * scale(dd, 0.16f), randScaling * scale(dd, 0.41f)}, randAngle + 18);

        randScaling = std::lround(1 - dist(mersenneTwisterEngine) * static_cast<data_t>(2) / 9);
        randAngle = std::lround(2 * 45 * (dist(mersenneTwisterEngine) - 0.5f));
        randTranslation1 = std::lround(0.2f * dist(mersenneTwisterEngine));
        randTranslation2 = std::lround(0.2f * dist(mersenneTwisterEngine));

        EllipseGenerator<data_t>::drawFilledEllipse2d(
            dc, 0.1f,
            {scaleShift(dd, randTranslation1 + 0), scaleShift(dd, randTranslation2 + 0.35f)},
            {randScaling * scale(dd, 0.21f), randScaling * scale(dd, 0.25)}, randAngle + 0);
        randScaling = std::lround(1 - dist(mersenneTwisterEngine) * static_cast<data_t>(2) / 9);
        randAngle = std::lround(2 * 45 * (dist(mersenneTwisterEngine) - 0.5f));
        randTranslation1 = std::lround(0.2f * dist(mersenneTwisterEngine));
        randTranslation2 = std::lround(0.2f * dist(mersenneTwisterEngine));

        EllipseGenerator<data_t>::drawFilledEllipse2d(
            dc, 0.1f,
            {scaleShift(dd, randTranslation1 + 0), scaleShift(dd, randTranslation2 + 0.1f)},
            {randScaling * scale(dd, 0.046f), randScaling * scale(dd, 0.046f)}, randAngle + 0);
        randScaling = std::lround(1 - dist(mersenneTwisterEngine) * static_cast<data_t>(2) / 9);
        randAngle = std::lround(2 * 45 * (dist(mersenneTwisterEngine) - 0.5f));
        randTranslation1 = std::lround(0.2f * dist(mersenneTwisterEngine));
        randTranslation2 = std::lround(0.2f * dist(mersenneTwisterEngine));

        EllipseGenerator<data_t>::drawFilledEllipse2d(
            dc, 0.1f,
            {scaleShift(dd, randTranslation1 + 0), scaleShift(dd, randTranslation2 - 0.1f)},
            {randScaling * scale(dd, 0.046f), randScaling * scale(dd, 0.046f)}, randAngle + 0);

        randScaling = std::lround(1 - dist(mersenneTwisterEngine) * static_cast<data_t>(2) / 9);
        randAngle = std::lround(2 * 45 * (dist(mersenneTwisterEngine) - 0.5f));
        randTranslation1 = std::lround(0.2f * dist(mersenneTwisterEngine));
        randTranslation2 = std::lround(0.2f * dist(mersenneTwisterEngine));

        EllipseGenerator<data_t>::drawFilledEllipse2d(
            dc, 0.1f,
            {scaleShift(dd, randTranslation1 - 0.08f), scaleShift(dd, randTranslation2 - 0.605f)},
            {randScaling * scale(dd, 0.046f), randScaling * scale(dd, 0.023f)}, randAngle + 0);

        randScaling = std::lround(1 - dist(mersenneTwisterEngine) * static_cast<data_t>(2) / 9);
        randAngle = std::lround(2 * 45 * (dist(mersenneTwisterEngine) - 0.5f));
        randTranslation1 = std::lround(0.2f * dist(mersenneTwisterEngine));
        randTranslation2 = std::lround(0.2f * dist(mersenneTwisterEngine));

        EllipseGenerator<data_t>::drawFilledEllipse2d(
            dc, 0.1f,
            {scaleShift(dd, randTranslation1 + 0), scaleShift(dd, randTranslation2 - 0.606f)},
            {randScaling * scale(dd, 0.023f), randScaling * scale(dd, 0.023f)}, randAngle + 0);

        randScaling = std::lround(1 - dist(mersenneTwisterEngine) * static_cast<data_t>(2) / 9);
        randAngle = std::lround(2 * 45 * (dist(mersenneTwisterEngine) - 0.5f));
        randTranslation1 = std::lround(0.2f * dist(mersenneTwisterEngine));
        randTranslation2 = std::lround(0.2f * dist(mersenneTwisterEngine));

        EllipseGenerator<data_t>::drawFilledEllipse2d(
            dc, 0.1f,
            {scaleShift(dd, randTranslation1 + 0.06f), scaleShift(dd, randTranslation2 - 0.605f)},
            {randScaling * scale(dd, 0.023f), randScaling * scale(dd, 0.046f)}, randAngle + 0);

        return dc;
    }

    template <typename data_t>
    index_t PhantomGenerator<data_t>::scale(const DataDescriptor& dd, data_t value)
    {
        return std::lround(
            value * static_cast<data_t>(dd.getNumberOfCoefficientsPerDimension()[0] - 1) / 2.0f);
    }

    template <typename data_t>
    index_t PhantomGenerator<data_t>::scaleShift(const DataDescriptor& dd, data_t value)
    {
        return std::lround(value
                           * static_cast<data_t>(dd.getNumberOfCoefficientsPerDimension()[0] - 1)
                           / 2.0f)
               + (dd.getNumberOfCoefficientsPerDimension()[0] / 2);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class PhantomGenerator<float>;
    template class PhantomGenerator<double>;

} // namespace elsa