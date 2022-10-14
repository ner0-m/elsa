#include "Intersection.h"

namespace elsa
{
    namespace detail
    {
        template <class data_t>
        std::tuple<data_t, data_t, data_t, data_t> intersect(const BoundingBox& aabb,
                                                             const Ray_t<data_t>& r)
        {
            data_t invDir = 1 / r.direction()(0);

            data_t t1 = (aabb.min()(0) - r.origin()(0)) * invDir;
            data_t t2 = (aabb.max()(0) - r.origin()(0)) * invDir;

            data_t tmin = invDir >= 0 ? t1 : t2;
            data_t tmax = invDir >= 0 ? t2 : t1;
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
    } // namespace detail

    template <class data_t>
    std::optional<IntersectionResult<data_t>> intersectRay(const BoundingBox& aabb,
                                                           const Ray_t<data_t>& r)
    {
        const auto [tmin, tmax, txmin, txmax] = detail::intersect(aabb, r);

        if (tmax == 0.0 && tmin == 0.0)
            return std::nullopt;
        if (tmax >= maxNum(tmin, 0.0f)) // hit
            return std::make_optional<IntersectionResult<data_t>>(tmin, tmax);
        return std::nullopt;
    }

    template <class data_t>
    std::optional<IntersectionResult<data_t>> intersectXPlanes(BoundingBox aabb,
                                                               const Ray_t<data_t>& r)
    {
        // Slightly increase the size of the bounding box, such that center of voxels are actually
        // inside the bounding box, parallel rays which directly go through the center of the
        // voxel, will be recognized with this cheapo hack
        aabb.min()[0] += 0.5f;
        aabb.max()[0] -= 0.5f;

        auto [tmin, tmax, txmin, txmax] = detail::intersect(aabb, r);

        auto dist_to_integer = [](auto real) {
            const auto aabbmin = static_cast<data_t>(static_cast<int>(std::round(real)));
            return std::abs(real - aabbmin);
        };

        auto fractional = [](auto real) {
            data_t trash = 0;
            return std::modf(real, &trash);
        };

        auto advance_to_next_voxel = [&](auto aabb, auto& tmin) -> data_t {
            // the x-axis coord for voxel centers
            const auto xvoxelcenter = dist_to_integer(aabb.min()[0]);

            Vector_t<data_t> entry = r.pointAt(tmin);

            // Calculate the distance from entry.x to the x coord of the next voxel
            const auto tmp = std::abs(fractional(entry[0] - xvoxelcenter));
            const auto frac = entry[0] < xvoxelcenter ? tmp : 1 - tmp;

            // Calculate distance in t, but don't advance if it's too small
            return tmp > 0.00001 && frac > 0.00001 ? frac / r.direction()[0] : 0;
        };

        auto retreat_to_last_voxel = [&](auto aabb, auto& tmax) -> data_t {
            const auto xvoxelcenter = dist_to_integer(aabb.min()[0]);

            Vector_t<data_t> exit = r.pointAt(tmax);

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
            return std::make_optional<IntersectionResult<data_t>>(tmin, tmax);

        return std::nullopt;
    }

    // ------------------------------------------
    // explicit template instantiation
    namespace detail
    {
        template std::tuple<float, float, float, float> intersect<float>(const BoundingBox& aabb,
                                                                         const Ray_t<float>& r);
        template std::tuple<double, double, double, double>
            intersect<double>(const BoundingBox& aabb, const Ray_t<double>& r);
    } // namespace detail

    template struct IntersectionResult<float>;
    template struct IntersectionResult<double>;

    template std::optional<IntersectionResult<float>> intersectRay<float>(const BoundingBox& aabb,
                                                                          const Ray_t<float>& r);
    template std::optional<IntersectionResult<double>> intersectRay<double>(const BoundingBox& aabb,
                                                                            const Ray_t<double>& r);

    template std::optional<IntersectionResult<float>>
        intersectXPlanes<float>(BoundingBox aabb, const Ray_t<float>& r);
    template std::optional<IntersectionResult<double>>
        intersectXPlanes<double>(BoundingBox aabb, const Ray_t<double>& r);

} // namespace elsa
