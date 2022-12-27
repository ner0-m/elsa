#include "Phantoms.h"
#include "EllipseGenerator.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "CartesianIndices.h"
#include "ForbildPhantom.h"
#include "ForbildData.h"

#include <cmath>
#include <stdexcept>

#include <spdlog/fmt/ostr.h>

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
            for (std::array<data_t, 10> e : modifiedSheppLoganParameters<data_t>) {

                Vec3X<data_t> halfAxis{data_t(scale(dd, e[1])), data_t(scale(dd, e[2])),
                                       data_t(scale(dd, e[3]))};

                if (halfAxis[0] < 1 || halfAxis[1] < 1 || halfAxis[2] < 1 || e[0] == data_t(0)) {
                    Logger::get("phantom::modifiedSheppLogan")
                        ->warn(
                            "Ellipsoid will not be rendered, because of amplitude=0 or an invalid "
                            "half axis! amplitude {}, half axis ({},{},{}) ",
                            e[0], halfAxis[0], halfAxis[1], halfAxis[2]);
                    continue;
                }

                Ellipsoid<data_t> ellipsoid{
                    e[0],
                    {scaleShift(dd, e[4]), scaleShift(dd, e[5]), scaleShift(dd, e[6])},
                    halfAxis,
                    {e[7], e[8], e[9]}};
                Logger::get("phantom::modifiedSheppLogan")->info("rasterize {}", ellipsoid);
                rasterize<Blending::ADDITION>(ellipsoid, dd, dc);
            }
        }

        return dc;
    }

    template <typename data_t>
    DataContainer<data_t> forbild(IndexVector_t sizes, ForbildPhantom<data_t>& fp)
    {
        // sanity checks
        if (sizes.size() != 3 || sizes[0] != sizes[1] || sizes[0] != sizes[2]) {
            throw InvalidArgumentError("phantom::forbild: 3d size has to be cubed");
        }

        VolumeDescriptor dd(sizes);
        DataContainer<data_t> dc(dd);
        dc = 0;

        for (int order = 0; order < fp.getMaxOrderIndex(); order++) {
            if (fp.getEllipsoids().count(order)) {
                // Ellipsoids
                auto e = fp.getEllipsoids().at(order);
                Logger::get("phantoms::forbild")->info("rasterize {}", e);
                rasterize<Blending::OVERWRITE>(e, dd, dc);
            } else if (fp.getEllipsoidsClippedX().count(order)) {
                // Ellipsoids clipped at x axis
                auto [el, clipping] = fp.getEllipsoidsClippedX().at(order);
                Logger::get("phantoms::forbild")->info("rasterize {} with clipping", el);
                rasterizeWithClipping<Blending::OVERWRITE>(el, dd, dc, clipping);
            } else if (fp.getSpheres().count(order)) {
                // Spheres
                auto s = fp.getSpheres().at(order);
                Logger::get("phantoms::forbild")->info("rasterize {}", s);
                rasterize<Blending::OVERWRITE>(s, dd, dc);
            } else if (fp.getEllipCylinders().count(order)) {
                // EllipCylinders
                auto e = fp.getEllipCylinders().at(order);
                Logger::get("phantoms::forbild")->info("rasterize {}", e);
                rasterize<Blending::OVERWRITE>(e, dd, dc);
            } else if (fp.getEllipCylindersFree().count(order)) {
                // EllipCylinders free
                auto e = fp.getEllipCylindersFree().at(order);
                Logger::get("phantoms::forbild")->info("rasterize {}", e);
                rasterize<Blending::OVERWRITE>(e, dd, dc);
            } else if (fp.getCylinders().count(order)) {
                // Cylinders
                auto c = fp.getCylinders().at(order);
                Logger::get("phantoms::forbild")->info("rasterize {}", c);
                rasterize<Blending::OVERWRITE>(c, dd, dc);
            } else if (fp.getCylindersFree().count(order)) {
                // Cylinders free
                auto cf = fp.getCylindersFree().at(order);
                Logger::get("phantoms::forbild")->info("rasterize {}", cf);
                rasterize<Blending::OVERWRITE>(cf, dd, dc);
            } else if (fp.getBoxes().count(order)) {
                // Boxes
                auto b = fp.getBoxes().at(order);
                Logger::get("phantoms::forbild")->info("rasterize {}", b);
                rasterize<Blending::OVERWRITE>(b, dd, dc);
            } else {
                Logger::get("phantoms::forbild")->warn("No object found for order index {}", order);
            }
        }

        return dc;
    }

    template <typename data_t>
    DataContainer<data_t> forbild_head(IndexVector_t sizes)
    {

        Logger::get("phantom::forbild_head")
            ->info("creating forbild of size {}^{}", sizes[0], sizes.size());
        // field view from http://ftp.imp.uni-erlangen.de/phantoms/head/head.html
        ForbildPhantom<data_t> fp{sizes[0] - 1, data_t(25.6f), forbilddata::maxOrderIndex_head};
        fp.addEllipsoids(forbilddata::ellipsoids_head<data_t>);
        fp.addEllipsoidsClippedX(forbilddata::ellipsoidsClippingX_head<data_t>);
        fp.addSpheres(forbilddata::spheres_head<data_t>);
        fp.addEllipCylinders(forbilddata::ellipCyls_head<data_t>);
        fp.addEllipCylindersFree(forbilddata::ellipCylsFree_head<data_t>);
        fp.addCylinders(forbilddata::cyls_head<data_t>);
        fp.addCylindersFree(forbilddata::cylsFree_head<data_t>);
        fp.addBoxes(forbilddata::boxes_head<data_t>);
        return std::move(forbild<data_t>(sizes, fp));
    }

    template <typename data_t>
    DataContainer<data_t> forbild_thorax(IndexVector_t sizes)
    {

        Logger::get("phantom::forbild_thorax")
            ->info("creating forbild of size {}^{}", sizes[0], sizes.size());
        // field view from http://ftp.imp.uni-erlangen.de/phantoms/thorax/thorax.htm
        ForbildPhantom<data_t> fp{sizes[0] - 1, data_t(50.f), forbilddata::maxOrderIndex_thorax};
        fp.addEllipsoids(forbilddata::ellipsoids_thorax<data_t>);
        fp.addSpheres(forbilddata::spheres_thorax<data_t>);
        fp.addEllipCylinders(forbilddata::ellipCyls_thorax<data_t>);
        fp.addEllipCylindersFree(forbilddata::ellipCylsFree_thorax<data_t>);
        fp.addCylinders(forbilddata::cyls_thorax<data_t>);
        fp.addCylindersFree(forbilddata::cylsFree_thorax<data_t>);
        fp.addBoxes(forbilddata::boxes_thorax<data_t>);

        return std::move(forbild<data_t>(sizes, fp));
    }

    template <typename data_t>
    DataContainer<data_t> forbild_abdomen(IndexVector_t sizes)
    {

        Logger::get("phantom::forbild_abdomen")
            ->info("creating forbild of size {}^{}", sizes[0], sizes.size());
        // field view from http://ftp.imp.uni-erlangen.de/phantoms/abdomen/abdomen.html
        ForbildPhantom<data_t> fp{sizes[0] - 1, data_t(40.f), forbilddata::maxOrderIndex_abdomen};
        fp.addEllipsoids(forbilddata::ellipsoids_abdomen<data_t>);
        fp.addSpheres(forbilddata::spheres_abdomen<data_t>);
        fp.addEllipCylinders(forbilddata::ellipCyls_abdomen<data_t>);
        fp.addEllipCylindersFree(forbilddata::ellipCylsFree_abdomen<data_t>);
        fp.addCylinders(forbilddata::cyls_abdomen<data_t>);
        fp.addCylindersFree(forbilddata::cylsFree_abdomen<data_t>);
        fp.addBoxes(forbilddata::boxes_abdomen<data_t>);
        return std::move(forbild<data_t>(sizes, fp));
    }

    // ------------------------------------------
    // explicit template instantiation
    template DataContainer<float> circular<float>(IndexVector_t volumesize, float radius);
    template DataContainer<double> circular<double>(IndexVector_t volumesize, double radius);

    template DataContainer<float> modifiedSheppLogan<float>(IndexVector_t sizes);
    template DataContainer<double> modifiedSheppLogan<double>(IndexVector_t sizes);

    template DataContainer<float> forbild_head<float>(IndexVector_t sizes);
    template DataContainer<double> forbild_head<double>(IndexVector_t sizes);

    template DataContainer<float> forbild_abdomen<float>(IndexVector_t sizes);
    template DataContainer<double> forbild_abdomen<double>(IndexVector_t sizes);

    template DataContainer<float> forbild_thorax<float>(IndexVector_t sizes);
    template DataContainer<double> forbild_thorax<double>(IndexVector_t sizes);

    template DataContainer<float> rectangle<float>(IndexVector_t volumesize, IndexVector_t lower,
                                                   IndexVector_t upper);
    template DataContainer<double> rectangle<double>(IndexVector_t volumesize, IndexVector_t lower,
                                                     IndexVector_t upper);

    namespace old
    {

        template <typename data_t>
        void drawFilledEllipsoid3d(DataContainer<data_t>& dc, data_t amplitude, Vec3 center,
                                   Vec3 sizes, data_t phi, data_t theta, data_t psi)
        {
            // sanity check
            if (dc.getDataDescriptor().getNumberOfDimensions() != 3)
                throw InvalidArgumentError(
                    "EllipseGenerator::drawFilledEllipsoid3d: can only work on 3d "
                    "DataContainers");

            // enables small optimizations
            bool hasRotation = (std::abs(phi) + std::abs(theta) + std::abs(psi)) > 0;

            // convert to radians
            auto phiRad = phi * pi<double> / 180.0;
            auto thetaRad = theta * pi<double> / 180.0;
            auto psiRad = psi * pi<double> / 180.0;

            auto cosPhi = std::cos(phiRad);
            auto sinPhi = std::sin(phiRad);
            auto cosTheta = std::cos(thetaRad);
            auto sinTheta = std::sin(thetaRad);
            auto cosPsi = std::cos(psiRad);
            auto sinPsi = std::sin(psiRad);

            // setup ZXZ Euler rotation matrix
            Eigen::Matrix<data_t, 3, 3> rot;
            rot(0, 0) = static_cast<real_t>(cosPhi * cosPsi - cosTheta * sinPhi * sinPsi);
            rot(0, 1) = static_cast<real_t>(cosPsi * sinPhi + cosPhi * cosTheta * sinPsi);
            rot(0, 2) = static_cast<real_t>(sinTheta * sinPsi);

            rot(1, 0) = static_cast<real_t>(-cosPhi * sinPsi - cosTheta * cosPsi * sinPhi);
            rot(1, 1) = static_cast<real_t>(cosPhi * cosTheta * cosPsi - sinPhi * sinPsi);
            rot(1, 2) = static_cast<real_t>(cosPsi * sinTheta);

            rot(2, 0) = static_cast<real_t>(sinPhi * sinTheta);
            rot(2, 1) = static_cast<real_t>(-cosPhi * sinTheta);
            rot(2, 2) = static_cast<real_t>(cosTheta);

            // enables safe early abort
            index_t maxSize = sizes.maxCoeff();

            // precomputations
            index_t asq = sizes[0] * sizes[0];
            index_t bsq = sizes[1] * sizes[1];
            index_t csq = sizes[2] * sizes[2];

            IndexVector_t idx(3);

            // loop over everything... (very inefficient!)
            for (index_t x = 0; x < dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[0];
                 ++x) {
                if (x < center[0] - maxSize || x > center[0] + maxSize)
                    continue;
                idx[0] = x;

                for (index_t y = 0;
                     y < dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1]; ++y) {
                    if (y < center[1] - maxSize || y > center[1] + maxSize)
                        continue;
                    idx[1] = dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[1] - 1
                             - y; // flip y axis

                    for (index_t z = 0;
                         z < dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()[2]; ++z) {
                        if (z < center[2] - maxSize || z > center[2] + maxSize)
                            continue;
                        idx[2] = z;

                        // center it
                        index_t xc = x - center[0];
                        index_t yc = y - center[1];
                        index_t zc = z - center[2];

                        // check ellipsoid equation
                        data_t aPart = (hasRotation) ? static_cast<data_t>(xc) * rot(0, 0)
                                                           + static_cast<data_t>(yc) * rot(0, 1)
                                                           + static_cast<data_t>(zc) * rot(0, 2)
                                                     : static_cast<data_t>(xc);
                        aPart *= aPart / static_cast<data_t>(asq);

                        data_t bPart = (hasRotation) ? static_cast<data_t>(xc) * rot(1, 0)
                                                           + static_cast<data_t>(yc) * rot(1, 1)
                                                           + static_cast<data_t>(zc) * rot(1, 2)
                                                     : static_cast<data_t>(yc);
                        bPart *= bPart / static_cast<data_t>(bsq);

                        data_t cPart = (hasRotation) ? static_cast<data_t>(xc) * rot(2, 0)
                                                           + static_cast<data_t>(yc) * rot(2, 1)
                                                           + static_cast<data_t>(zc) * rot(2, 2)
                                                     : static_cast<data_t>(zc);
                        cPart *= cPart / static_cast<data_t>(csq);

                        if (aPart + bPart + cPart <= 1.0)
                            dc(idx) += amplitude;
                    }
                }
            }
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
                ->info("creating modified Shepp Logan phantom of size {}^{}", sizes[0],
                       sizes.size());

            VolumeDescriptor dd(sizes);
            DataContainer<data_t> dc(dd);
            dc = 0;

            if (sizes.size() == 2) {
                EllipseGenerator<data_t>::drawFilledEllipse2d(
                    dc, 1.0, {scaleShift(dd, 0), scaleShift(dd, 0)},
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
                for (std::array<data_t, 10> e : modifiedSheppLoganParameters<data_t>) {
                    drawFilledEllipsoid3d(
                        dc, e[0],
                        {scaleShift(dd, e[4]), scaleShift(dd, e[5]), scaleShift(dd, e[6])},
                        {scale(dd, e[1]), scale(dd, e[2]), scale(dd, e[3])}, e[7], e[8], e[9]);
                }
            }
            return dc;
        };
        template void drawFilledEllipsoid3d<double>(DataContainer<double>& dc, double amplitude,
                                                    Vec3 center, Vec3 sizes, double phi,
                                                    double theta, double psi);

        template void drawFilledEllipsoid3d<float>(DataContainer<float>& dc, float amplitude,
                                                   Vec3 center, Vec3 sizes, float phi, float theta,
                                                   float psi);
        template DataContainer<double> modifiedSheppLogan(IndexVector_t sizes);

    } // namespace old
} // namespace elsa::phantoms
