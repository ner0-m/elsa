#pragma once

#include "elsaDefines.h"
#include "BoundingBox.h"

#include "spdlog/fmt/fmt.h"

#include <Eigen/Geometry>
#include <limits>
#include <optional>
#include <functional>

namespace elsa
{
    /**
     * @brief Helper struct for results of intersection tests
     */
    template <class data_t>
    struct IntersectionResult {
        /// the parameters for entry/exit points
        data_t _tmin, _tmax;

        /// default constructor
        IntersectionResult()
            : _tmin{std::numeric_limits<data_t>::infinity()},
              _tmax{std::numeric_limits<data_t>::infinity()}
        {
        }

        /// simple constructor with values tmin, tmax
        IntersectionResult(data_t tmin, data_t tmax) : _tmin{tmin}, _tmax{tmax} {}
    };

    namespace detail
    {
        template <class data_t>
        std::tuple<data_t, data_t, data_t, data_t> intersect(const BoundingBox& aabb,
                                                             const Ray_t<data_t>& r);
    } // namespace detail

    /**
     * @brief Compute entry and exit point of ray in a volume (given as an AABB)
     *
     * If the ray is running along a border of the bounding box, the lower bound will
     * be counted as in the bounding and the upper bound will be  counted as outside.
     *
     * Method adapted from
     * https://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/
     *
     * @param[in] aabb the volume specified through an axis-aligned bounding box
     * @param[in] r the ray which we test for intersection with aabb
     *
     * @returns nullopt if the volume is not hit, otherwise IntersectionResult
     *          with entry/exit parameters tmin/tmax
     */
    template <class data_t>
    std::optional<IntersectionResult<data_t>> intersectRay(const BoundingBox& aabb,
                                                           const Ray_t<data_t>& r);

    /**
     * @brief Compute the intersection of a ray with the nearest x-plane
     *
     * @param[in] aabb the volume specified through an axis-aligned bounding box
     * @param[in] r the ray which we test for intersection with aabb
     *
     * @returns nullopt if the volume is not hit, otherwise IntersectionResult
     *          with entry/exit parameters tmin/tmax
     */
    template <class data_t>
    std::optional<IntersectionResult<data_t>> intersectXPlanes(BoundingBox aabb,
                                                               const Ray_t<data_t>& r);

    /**
     * @brief min helper function which behaves like the IEEE standard suggests
     *
     * @param[in] x first value to find minimum of
     * @param[in] y second value to find minimum of
     * @returns the smaller value of x and y
     *
     * This function is used, because it tries to suppress NaN's. This behavior is
     * crucial for edge cases of the intersection algorithm. std::min(x,y) does
     * not provide this security and should be avoided in this case
     */
    template <typename T1, typename T2>
    constexpr T1 minNum(T1 x, T2 y)
    {
        return std::not_equal_to<>()(y, y) ? x : (std::less<>()(x, y) ? x : y);
    }

    /**
     * @brief max helper function which behaves like the IEEE standard suggests
     *
     * @param[in] x first value to find maximum of
     * @param[in] y second value to find maximum of
     * @returns the bigger value of x and y
     *
     * Same thing for NaN's as for the min function
     */
    template <typename T1, typename T2>
    constexpr T1 maxNum(T1 x, T2 y)
    {
        return std::not_equal_to<>()(y, y) ? x : (std::greater<>()(x, y) ? x : y);
    }

} // namespace elsa

template <class data_t>
struct fmt::formatter<elsa::IntersectionResult<data_t>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const elsa::IntersectionResult<data_t>& hit, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(), "{{ tmin: {}, tmax: {} }}", hit._tmin, hit._tmax);
    }
};
