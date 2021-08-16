#include "testHelpers.h"

namespace elsa
{
    namespace detail
    {
        template <typename First, typename Second>
        bool isCwiseApproxImpl(const First& x, const Second& y)
        {
            for (index_t i = 0; i < x.getSize(); ++i) {
                INFO("DataHandler is different as pos ", i);
                auto eq = approxEq(x[i], y[i]);
                if (!eq)
                    return false;
            }

            return true;
        }

        template <typename data_t, typename First, typename Second, typename Third>
        bool isApproxImpl(const First& x, const Second& y, Third& z, real_t prec)
        {
            z -= y;

            if constexpr (std::is_same_v<data_t, index_t>) {
                auto lhs = z.l2Norm();
                auto rhs = std::min(x.l2Norm(), y.l2Norm());
                return lhs == rhs;
            } else {
                auto lhs = z.l2Norm();
                auto rhs = prec * std::min(x.l2Norm(), y.l2Norm());
                return lhs <= rhs;
            }
        }
    } // namespace detail

    template <typename data_t>
    bool isCwiseApprox(const DataHandler<data_t>& x, const Vector_t<data_t>& y)
    {
        CHECK_EQ(x.getSize(), y.size());

        return detail::isCwiseApproxImpl(x, y);
    }

    template <typename data_t>
    bool isCwiseApprox(const DataHandler<data_t>& x, const DataHandler<data_t>& y)
    {
        CHECK_EQ(x.getSize(), y.getSize());

        return detail::isCwiseApproxImpl(x, y);
    }

    template <typename data_t>
    bool isCwiseApprox(const DataContainer<data_t>& x, const Vector_t<data_t>& y)
    {
        CHECK_EQ(x.getSize(), y.size());

        return detail::isCwiseApproxImpl(x, y);
    }

    template <typename data_t>
    bool isCwiseApprox(const DataContainer<data_t>& x, const DataContainer<data_t>& y)
    {
        CHECK_EQ(x.getSize(), y.getSize());

        return detail::isCwiseApproxImpl(x, y);
    }

    template <typename data_t>
    bool isApprox(const DataContainer<data_t>& x, const DataContainer<data_t>& y, real_t prec)
    {
        // check if size is the same, but do not throw an exception
        REQUIRE_EQ(x.getSize(), y.getSize());

        DataContainer<data_t> z = x;

        return detail::isApproxImpl<data_t>(x, y, z, prec);
    }

    template <typename data_t>
    bool isApprox(const DataContainer<data_t>& x, const Vector_t<data_t>& y, real_t prec)
    {
        // check if size is the same, but do not throw an exception
        REQUIRE_EQ(x.getSize(), y.size());

        DataContainer<data_t> z = x;
        DataContainer<data_t> yDc(x.getDataDescriptor(), y);

        return detail::isApproxImpl<data_t>(x, yDc, z, prec);
    }

    // template <typename data_t>
    // bool isApprox(const DataHandler<data_t>& x, const Eigen::Matrix<data_t, Eigen::Dynamic, 1>&
    // y,
    //               real_t prec)
    // {
    //     // check if size is the same, but do not throw an exception
    //     REQUIRE_EQ(x.getSize(), y.size());

    //     auto z = x.clone();

    //     return detail::isApproxImpl<data_t>(x, y, *z, prec);
    // }

    template <typename data_t>
    bool isApprox(const DataHandler<data_t>& x, const DataHandler<data_t>& y, real_t prec)
    {
        // check if size is the same, but do not throw an exception
        REQUIRE_EQ(x.getSize(), y.getSize());

        auto z = x.clone();

        return detail::isApproxImpl<data_t>(x, y, *z, prec);
    }

    doctest::Approx operator"" _a(long double val)
    {
        return doctest::Approx(static_cast<double>(val));
    }

    doctest::Approx operator"" _a(unsigned long long val)
    {
        return doctest::Approx(static_cast<double>(val));
    }

    // ------------------------------------------
    // explicit template instantiation
    template bool isCwiseApprox(const DataContainer<index_t>& c, const DataContainer<index_t>& y);
    template bool isCwiseApprox(const DataContainer<float>& c, const DataContainer<float>& y);
    template bool isCwiseApprox(const DataContainer<double>& c, const DataContainer<double>& y);
    template bool isCwiseApprox(const DataContainer<std::complex<float>>& c,
                                const DataContainer<std::complex<float>>& y);
    template bool isCwiseApprox(const DataContainer<std::complex<double>>& c,
                                const DataContainer<std::complex<double>>& y);

    template bool isCwiseApprox(const DataContainer<index_t>& c, const Vector_t<index_t>& y);
    template bool isCwiseApprox(const DataContainer<float>& c, const Vector_t<float>& y);
    template bool isCwiseApprox(const DataContainer<double>& c, const Vector_t<double>& y);
    template bool isCwiseApprox(const DataContainer<std::complex<float>>& c,
                                const Vector_t<std::complex<float>>& y);
    template bool isCwiseApprox(const DataContainer<std::complex<double>>& c,
                                const Vector_t<std::complex<double>>& y);

    template bool isCwiseApprox(const DataHandler<index_t>& c, const DataHandler<index_t>& y);
    template bool isCwiseApprox(const DataHandler<float>& c, const DataHandler<float>& y);
    template bool isCwiseApprox(const DataHandler<double>& c, const DataHandler<double>& y);
    template bool isCwiseApprox(const DataHandler<std::complex<float>>& c,
                                const DataHandler<std::complex<float>>& y);
    template bool isCwiseApprox(const DataHandler<std::complex<double>>& c,
                                const DataHandler<std::complex<double>>& y);

    template bool isCwiseApprox(const DataHandler<index_t>& c, const Vector_t<index_t>& y);
    template bool isCwiseApprox(const DataHandler<float>& c, const Vector_t<float>& y);
    template bool isCwiseApprox(const DataHandler<double>& c, const Vector_t<double>& y);
    template bool isCwiseApprox(const DataHandler<std::complex<float>>& c,
                                const Vector_t<std::complex<float>>& y);
    template bool isCwiseApprox(const DataHandler<std::complex<double>>& c,
                                const Vector_t<std::complex<double>>& y);

    template bool isApprox(const DataContainer<index_t>& x, const DataContainer<index_t>& y,
                           real_t prec);
    template bool isApprox(const DataContainer<float>& x, const DataContainer<float>& y,
                           real_t prec);
    template bool isApprox(const DataContainer<double>& x, const DataContainer<double>& y,
                           real_t prec);
    template bool isApprox(const DataContainer<std::complex<float>>& x,
                           const DataContainer<std::complex<float>>& y, real_t prec);
    template bool isApprox(const DataContainer<std::complex<double>>& x,
                           const DataContainer<std::complex<double>>& y, real_t prec);

    template bool isApprox(const DataContainer<index_t>& x, const Vector_t<index_t>& y,
                           real_t prec);
    template bool isApprox(const DataContainer<float>& x, const Vector_t<float>& y, real_t prec);
    template bool isApprox(const DataContainer<double>& x, const Vector_t<double>& y, real_t prec);
    template bool isApprox(const DataContainer<std::complex<float>>& x,
                           const Vector_t<std::complex<float>>& y, real_t prec);
    template bool isApprox(const DataContainer<std::complex<double>>& x,
                           const Vector_t<std::complex<double>>& y, real_t prec);

    template bool isApprox(const DataHandler<index_t>& x, const DataHandler<index_t>& y,
                           real_t prec);
    template bool isApprox(const DataHandler<float>& x, const DataHandler<float>& y, real_t prec);
    template bool isApprox(const DataHandler<double>& x, const DataHandler<double>& y, real_t prec);
    template bool isApprox(const DataHandler<std::complex<float>>& x,
                           const DataHandler<std::complex<float>>& y, real_t prec);
    template bool isApprox(const DataHandler<std::complex<double>>& x,
                           const DataHandler<std::complex<double>>& y, real_t prec);
} // namespace elsa
