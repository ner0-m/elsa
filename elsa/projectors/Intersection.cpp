#include "Intersection.h"

namespace elsa
{
    std::optional<IntersectionResult> Intersection::withRay(const BoundingBox& aabb, const Ray& r)
    {
        real_t invDir = 1 / r.direction()(0);

        real_t t1 = (aabb._min(0) - r.origin()(0)) * invDir;
        real_t t2 = (aabb._max(0) - r.origin()(0)) * invDir;

        real_t tmin = invDir >= 0 ? t1 : t2;
        real_t tmax = invDir >= 0 ? t2 : t1;

        for (int i = 1; i < aabb._min.rows(); ++i) {
            invDir = 1 / r.direction()(i);

            t1 = (aabb._min(i) - r.origin()(i)) * invDir;
            t2 = (aabb._max(i) - r.origin()(i)) * invDir;

            tmin = maxNum(tmin, invDir >= 0 ? t1 : t2);
            tmax = minNum(tmax, invDir >= 0 ? t2 : t1);
        }

        if (tmax == 0.0 && tmin == 0.0)
            return std::nullopt;
        if (tmax >= maxNum(tmin, 0.0f)) // hit
            return std::make_optional<IntersectionResult>(tmin, tmax);
        return std::nullopt;
    }

} // namespace elsa
