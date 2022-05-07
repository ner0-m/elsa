#include "Bessel.h"
#include "elsaDefines.h"
#include <unsupported/Eigen/SpecialFunctions>

namespace elsa::math
{
    double bessi0(double x)
    {
        double ax = std::abs(x);

        // polynomial fit, different polynoms for different ranges
        if (ax < 3.75) {
            const auto y = std::pow(x / 3.75, 2);

            // different terms
            const auto p0 = 0.45813e-2;
            const auto p1 = 0.360768e-1 + y * p0;
            const auto p2 = 0.2659732 + y * p1;
            const auto p3 = 1.2067492 + y * p2;
            const auto p4 = 3.0899424 + y * p3;
            const auto p5 = 3.5156229 + y * p4;
            return 1.0 + y * p5;
        } else {
            const auto y = 3.75 / ax;

            // different terms
            const auto p0 = 0.392377e-2;
            const auto p1 = -0.1647633e-1 + y * p0;
            const auto p2 = 0.2635537e-1 + y * p1;
            const auto p3 = -0.2057706e-1 + y * p2;
            const auto p4 = 0.916281e-2 + y * p3;
            const auto p5 = -0.157565e-2 + y * p4;
            const auto p6 = 0.225319e-2 + y * p5;
            const auto p7 = 0.1328592e-1 + y * p6;
            const auto p8 = 0.39894228 + y * p7;

            return (std::exp(ax) / std::sqrt(ax)) * p8;
        }
    }

    double bessi1(double x)
    {
        double result = 0;
        double ax = std::abs(x);

        // polynomial fit, different polynoms for different ranges
        if (ax < 3.75) {
            const auto y = std::pow(x / 3.75, 2);

            const auto p0 = 0.32411e-3;
            const auto p1 = 0.301532e-2 + y * p0;
            const auto p2 = 0.2658733e-1 + y * p1;
            const auto p3 = 0.15084934 + y * p2;
            const auto p4 = 0.51498869 + y * p3;
            const auto p5 = 0.87890594 + y * p4;
            const auto p6 = 0.5 + y * p5;

            result = ax * p6;
        } else {
            const auto y = 3.75 / ax;

            const auto p0 = 0.420059e-2;
            const auto p1 = 0.1787654e-1 - y * p0;
            const auto p2 = -0.2895312e-1 + y * p1;
            const auto p3 = 0.2282967e-1 + y * p2;
            const auto p4 = -0.1031555e-1 + y * p3;
            const auto p5 = 0.163801e-2 + y * p4;
            const auto p6 = -0.362018e-2 + y * p5;
            const auto p7 = -0.3988024e-1 + y * p6;
            const auto p8 = 0.39894228 + y * p7;

            result = p8 * (exp(ax) / sqrt(ax));
        }
        return x < 0.0 ? -result : result;
    }

    double bessi2(double x) { return (x == 0) ? 0 : bessi0(x) - ((2 * 1) / x) * bessi1(x); }

    double bessi3(double x) { return (x == 0) ? 0 : bessi1(x) - ((2 * 2) / x) * bessi2(x); }

    double bessi4(double x) { return (x == 0) ? 0 : bessi2(x) - ((2 * 3) / x) * bessi3(x); }

    double bessi(int m, double x)
    {
        if (m == 0) {
            return bessi0(x);
        } else if (m == 1) {
            return bessi1(x);
        } else if (m == 2) {
            return bessi2(x);
        } else if (m == 3) {
            return bessi3(x);
        } else if (m == 4) {
            return bessi4(x);
        }

        constexpr double ACC = 40.0;
        constexpr double BIGNO = 1.0e10;
        constexpr double BIGNI = 1.0e-10;

        if (x == 0.0) {
            return 0.0;
        } else {
            double tox = 2.0 / std::abs(x);
            double result = 0.0;
            double bip = 0.0;
            double bi = 1.0;
            for (int j = 2 * (m + (int) std::sqrt(ACC * m)); j > 0; --j) {
                double bim = bip + j * tox * bi;
                bip = bi;
                bi = bim;
                if (std::abs(bi) > BIGNO) {
                    result *= BIGNI;
                    bi *= BIGNI;
                    bip *= BIGNI;
                }

                if (j == m) {
                    result = bip;
                }
            }
            result *= bessi0(x) / bi;
            return (((x < 0.0) && ((m % 2) == 0)) ? -result : result);
        }
    }
} // namespace elsa::math
