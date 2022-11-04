#include "Phantoms.h"
#include "EllipseGenerator.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "CartesianIndices.h"

#include <cmath>
#include <stdexcept>

namespace elsa::phantoms
{

    // scale sizes from [0,1] to the (square) phantom size, producing indices (integers)
    template <typename data_t>
    index_t scale(const DataDescriptor& dd, data_t value)
    {
        return std::lround(
            value * static_cast<data_t>(dd.getNumberOfCoefficientsPerDimension()[0] - 1) / 2.0f);
    }

    // scale and shift center coordinates to the (square) phantom size, producing indices
    // (integers)
    template <typename data_t>
    index_t scaleShift(const DataDescriptor& dd, data_t value)
    {
        return std::lround(value
                           * static_cast<data_t>(dd.getNumberOfCoefficientsPerDimension()[0] - 1)
                           / 2.0f)
               + (dd.getNumberOfCoefficientsPerDimension()[0] / 2);
    }

    template <typename data_t>
    DataContainer<data_t> circular(IndexVector_t volumesize, data_t radius)
    {
        VolumeDescriptor dd(volumesize);
        DataContainer<data_t> dc(dd);
        dc = 0;

        const Vector_t<data_t> sizef = volumesize.template cast<data_t>();
        const auto center = (sizef.array() / 2).matrix();

        for (auto pos : CartesianIndices(volumesize)) {
            const Vector_t<data_t> p = pos.template cast<data_t>();
            if ((p - center).norm() <= radius) {
                dc(pos) = 1;
            }
        }

        return dc;
    }

    template <typename data_t>
    DataContainer<data_t> rectangle(IndexVector_t volumesize, IndexVector_t lower,
                                    IndexVector_t upper)
    {
        VolumeDescriptor dd(volumesize);
        DataContainer<data_t> dc(dd);
        dc = 0;

        for (auto pos : CartesianIndices(lower, upper)) {
            dc(pos) = 1;
        }

        return dc;
    }

    template <typename data_t>
    DataContainer<data_t> modifiedSheppLogan(IndexVector_t sizes)
    {
        // sanity check
        if (sizes.size() < 2 || sizes.size() > 3)
            throw InvalidArgumentError("phantom::modifiedSheppLogan: only 2d or 3d supported");
        if (sizes.size() == 2 && sizes[0] != sizes[1])
            throw InvalidArgumentError("phantom::modifiedSheppLogan: 2d size has to be square");
        if (sizes.size() == 3 && (sizes[0] != sizes[1] || sizes[0] != sizes[2]))
            throw InvalidArgumentError("phantom::modifiedSheppLogan: 3d size has to be cubed");

        Logger::get("phantom::modifiedSheppLogan")
            ->info("creating modified Shepp Logan phantom of size {}^{}", sizes[0], sizes.size());

        VolumeDescriptor dd(sizes);
        DataContainer<data_t> dc(dd);
        dc = 0;

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

    // ------------------------------------------
    // explicit template instantiation
    template DataContainer<float> circular<float>(IndexVector_t volumesize, float radius);
    template DataContainer<double> circular<double>(IndexVector_t volumesize, double radius);
    template DataContainer<float> modifiedSheppLogan<float>(IndexVector_t sizes);
    template DataContainer<double> modifiedSheppLogan<double>(IndexVector_t sizes);
    template DataContainer<float> rectangle<float>(IndexVector_t volumesize, IndexVector_t lower,
                                                   IndexVector_t upper);
    template DataContainer<double> rectangle<double>(IndexVector_t volumesize, IndexVector_t lower,
                                                     IndexVector_t upper);

} // namespace elsa::phantoms