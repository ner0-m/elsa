#include "PhantomGenerator.h"
#include "EllipseGenerator.h"
#include "Logger.h"
#include "VolumeDescriptor.h"

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

        VolumeDescriptor dd(sizes);
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