#include "doctest/doctest.h"

#include <optional>
#include <vector>
#include "spdlog/fmt/fmt.h"

namespace doctest
{
    template <typename T>
    struct StringMaker<std::optional<T>> {
        static String convert(std::optional<T> opt)
        {
            if (opt) {
                return fmt::format("{{ {} }}", *opt).c_str();
            } else {
                return "{ empty }";
            }
        }
    };

    template <typename T>
    struct StringMaker<std::vector<T>> {
        static String convert(const std::vector<T>& value)
        {
            return fmt::format("{}", value).c_str();
        }
    };
} // namespace doctest
