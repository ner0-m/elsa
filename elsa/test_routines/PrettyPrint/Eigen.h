#include "doctest/doctest.h"

#include <Eigen/Core>
#include <sstream>

namespace doctest
{
    template <typename T, int Rows, int Cols>
    struct StringMaker<Eigen::Matrix<T, Rows, Cols>> {
        static String convert(const Eigen::Matrix<T, Rows, Cols>& mat)
        {
            std::ostringstream oss;
            Eigen::IOFormat fmt(4, 0, ", ", "\n", "[", "]");
            oss << "\n" << mat.format(fmt);
            return oss.str().c_str();
        }
    };

    template <typename T, int Rows>
    struct StringMaker<Eigen::Matrix<T, Rows, 1>> {
        static String convert(const Eigen::Matrix<T, Rows, 1>& vec)
        {
            std::ostringstream oss;
            Eigen::IOFormat fmt(10, 0, ", ", ", ", "", "", "[", "]");
            oss << "\n" << vec.format(fmt);
            return oss.str().c_str();
        }
    };

} // namespace doctest
