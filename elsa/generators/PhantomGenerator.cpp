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
                                                          {scale(dd, 0.69), scale(dd, 0.92)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, -0.8, {scaleShift(dd, 0), scaleShift(dd, -0.0184)},
                {scale(dd, 0.6624), scale(dd, 0.8740)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(dc, -0.2,
                                                          {scaleShift(dd, 0.22), scaleShift(dd, 0)},
                                                          {scale(dd, 0.11), scale(dd, 0.31)}, -18);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, -0.2, {scaleShift(dd, -0.22), scaleShift(dd, 0)},
                {scale(dd, 0.16), scale(dd, 0.41)}, 18);
            EllipseGenerator<data_t>::drawFilledEllipse2d(dc, 0.1,
                                                          {scaleShift(dd, 0), scaleShift(dd, 0.35)},
                                                          {scale(dd, 0.21), scale(dd, 0.25)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(dc, 0.1,
                                                          {scaleShift(dd, 0), scaleShift(dd, 0.1)},
                                                          {scale(dd, 0.046), scale(dd, 0.046)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(dc, 0.1,
                                                          {scaleShift(dd, 0), scaleShift(dd, -0.1)},
                                                          {scale(dd, 0.046), scale(dd, 0.046)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, 0.1, {scaleShift(dd, -0.08), scaleShift(dd, -0.605)},
                {scale(dd, 0.046), scale(dd, 0.023)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, 0.1, {scaleShift(dd, 0), scaleShift(dd, -0.606)},
                {scale(dd, 0.023), scale(dd, 0.023)}, 0);
            EllipseGenerator<data_t>::drawFilledEllipse2d(
                dc, 0.1, {scaleShift(dd, 0.06), scaleShift(dd, -0.605)},
                {scale(dd, 0.023), scale(dd, 0.046)}, 0);
        }

        if (sizes.size() == 3) {
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 1.0, {scaleShift(dd, 0), scaleShift(dd, 0), scaleShift(dd, 0)},
                {scale(dd, 0.69), scale(dd, 0.92), scale(dd, 0.81)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, -0.8, {scaleShift(dd, 0), scaleShift(dd, -0.0184), scaleShift(dd, 0)},
                {scale(dd, 0.6624), scale(dd, 0.874), scale(dd, 0.78)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, -0.2, {scaleShift(dd, 0.22), scaleShift(dd, 0), scaleShift(dd, 0)},
                {scale(dd, 0.11), scale(dd, 0.31), scale(dd, 0.22)}, -18, 0, 10);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, -0.2, {scaleShift(dd, -0.22), scaleShift(dd, 0), scaleShift(dd, 0)},
                {scale(dd, 0.16), scale(dd, 0.41), scale(dd, 0.28)}, 18, 0, 10);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 0.1, {scaleShift(dd, 0), scaleShift(dd, 0.35), scaleShift(dd, -0.15)},
                {scale(dd, 0.21), scale(dd, 0.25), scale(dd, 0.41)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 0.1, {scaleShift(dd, 0), scaleShift(dd, 0.1), scaleShift(dd, 0.25)},
                {scale(dd, 0.046), scale(dd, 0.046), scale(dd, 0.05)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 0.1, {scaleShift(dd, 0), scaleShift(dd, -0.1), scaleShift(dd, 0.25)},
                {scale(dd, 0.046), scale(dd, 0.046), scale(dd, 0.05)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 0.1, {scaleShift(dd, -0.08), scaleShift(dd, -0.605), scaleShift(dd, 0)},
                {scale(dd, 0.046), scale(dd, 0.023), scale(dd, 0.05)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 0.1, {scaleShift(dd, 0), scaleShift(dd, -0.606), scaleShift(dd, 0)},
                {scale(dd, 0.023), scale(dd, 0.023), scale(dd, 0.02)}, 0, 0, 0);
            EllipseGenerator<data_t>::drawFilledEllipsoid3d(
                dc, 0.1, {scaleShift(dd, 0.06), scaleShift(dd, -0.605), scaleShift(dd, 0)},
                {scale(dd, 0.023), scale(dd, 0.046), scale(dd, 0.02)}, 0, 0, 0);
        }

        return dc;
    }

    template <typename data_t>
    index_t PhantomGenerator<data_t>::scale(const DataDescriptor& dd, data_t value)
    {
        return std::lround(value * (dd.getNumberOfCoefficientsPerDimension()[0] - 1) / 2.0f);
    }

    template <typename data_t>
    index_t PhantomGenerator<data_t>::scaleShift(const DataDescriptor& dd, data_t value)
    {
        return std::lround(value * (dd.getNumberOfCoefficientsPerDimension()[0] - 1) / 2.0f)
               + (dd.getNumberOfCoefficientsPerDimension()[0] / 2);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class PhantomGenerator<float>;
    template class PhantomGenerator<double>;

} // namespace elsa