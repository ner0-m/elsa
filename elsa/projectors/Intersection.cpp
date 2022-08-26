#include "Intersection.h"

namespace elsa
{
    std::tuple<real_t, real_t, real_t, real_t> intersect(const BoundingBox& aabb,
                                                         const RealRay_t& r)
    {
        real_t invDir = 1 / r.direction()(0);

        real_t t1 = (aabb.min()(0) - r.origin()(0)) * invDir;
        real_t t2 = (aabb.max()(0) - r.origin()(0)) * invDir;

        real_t tmin = invDir >= 0 ? t1 : t2;
        real_t tmax = invDir >= 0 ? t2 : t1;
        const auto txmin = tmin;
        const auto txmax = tmax;

        for (int i = 1; i < aabb.min().rows(); ++i) {
            invDir = 1 / r.direction()(i);

            t1 = (aabb.min()(i) - r.origin()(i)) * invDir;
            t2 = (aabb.max()(i) - r.origin()(i)) * invDir;

            tmin = maxNum(tmin, invDir >= 0 ? t1 : t2);
            tmax = minNum(tmax, invDir >= 0 ? t2 : t1);
        }

        return {tmin, tmax, txmin, txmax};
    }
    std::optional<IntersectionResult> Intersection::withRay(const BoundingBox& aabb,
                                                            const RealRay_t& r)
    {
        const auto [tmin, tmax, txmin, txmax] = intersect(aabb, r);

        if (tmax == 0.0 && tmin == 0.0)
            return std::nullopt;
        if (tmax >= maxNum(tmin, 0.0f)) // hit
            return std::make_optional<IntersectionResult>(tmin, tmax);
        return std::nullopt;
    }

    std::optional<IntersectionResult> Intersection::xPlanesWithRay(BoundingBox aabb,
                                                                   const RealRay_t& r)
    {
        // Slightly increase the size of the bounding box, such that center of voxels are actually
        // inside the bounding box, parallel rays which directly go through the center of the
        // voxel, will be recognized with this cheapo hack
        aabb.min()[0] += 0.5f;
        aabb.max()[0] -= 0.5f;

        auto [tmin, tmax, txmin, txmax] = intersect(aabb, r);

        auto dist_to_integer = [](auto real) {
            const auto aabbmin = static_cast<real_t>(static_cast<int>(std::round(real)));
            return std::abs(real - aabbmin);
        };

        auto fractional = [](auto real) {
            real_t trash = 0;
            return std::modf(real, &trash);
        };

        auto advance_to_next_voxel = [&](auto aabb, auto& tmin) -> real_t {
            // the x-axis coord for voxel centers
            const auto xvoxelcenter = dist_to_integer(aabb.min()[0]);

            RealVector_t entry = r.pointAt(tmin);

            // Calculate the distance from entry.x to the x coord of the next voxel
            const auto tmp = std::abs(fractional(entry[0] - xvoxelcenter));
            const auto frac = entry[0] < xvoxelcenter ? tmp : 1 - tmp;

            // Calculate distance in t, but don't advance if it's too small
            return tmp > 0.00001 && frac > 0.00001 ? frac / r.direction()[0] : 0;
        };

        auto retreat_to_last_voxel = [&](auto aabb, auto& tmax) -> real_t {
            const auto xvoxelcenter = dist_to_integer(aabb.min()[0]);

            RealVector_t exit = r.pointAt(tmax);

            // calculate the distance from entry.x to the x coord of the previous voxel
            const auto tmp = std::abs(fractional(exit[0] - xvoxelcenter));
            const auto frac = exit[0] < xvoxelcenter ? 1 - tmp : tmp;

            // Calculate distance in t, but don't advance if it's too small
            return tmp > 0.00001 && frac > 0.00001 ? frac / r.direction()[0] : 0;
        };

        // Sometimes we hit the y axis first, not the x axis, but we don't want
        // that, we always want the intersection with the x-plane
        if (txmin < tmin) {
            tmin += advance_to_next_voxel(aabb, tmin);
        }

        // This can also happen if we leave at the top of bottom
        if (txmax > tmax) {
            tmax -= retreat_to_last_voxel(aabb, tmax);
        }

        if (tmax == 0.0 && tmin == 0.0)
            return std::nullopt;

        if (tmax - maxNum(tmin, 0.0f) > -0.000001) // hit
            return std::make_optional<IntersectionResult>(tmin, tmax);

        return std::nullopt;
    }
} // namespace elsa
