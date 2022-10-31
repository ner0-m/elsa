#pragma once

#include <cmath>

#include "elsaDefines.h"

namespace elsa::math
{
    /// Modified Bessel Function of the First Kind \f$ I_n(x) \f$, with \f$n = 0\f$
    /// See:
    /// - Chapter 9 of Handbook of Mathematical Functions: with Formulas, Graphs, and Mathematical
    /// Tables, by Milton Abramowitz and Irene A. Stegun
    /// - Chapter 6.6 of Numerical recipes in C: The art of scientific computing, second edition,
    /// by Jaob Winkler (1993)
    /// - https://www.astro.rug.nl/~gipsy/sub/bessel.c
    /// - https://stackoverflow.com/questions/8797722/modified-bessel-functions-of-order-n
    double bessi0(double x);

    /// Modified Bessel Function of the First Kind \f$ I_n(x) \f$, with \f$n = 1\f$
    /// See:
    /// - Chapter 9 of Handbook of Mathematical Functions: with Formulas, Graphs, and Mathematical
    /// Tables, by Milton Abramowitz and Irene A. Stegun
    /// - Chapter 6.6 of Numerical recipes in C: The art of scientific computing, second edition,
    /// by Jaob Winkler (1993)
    double bessi1(double x);

    /// Modified Bessel Function of the First Kind \f$ I_n(x) \f$, with \f$n = 2\f$, using the
    /// recurrence relations, i.e. \f$ I_{n+1}(x) = I_{n-1}(x) - (2 * n / x) I_{n}(x)\f$
    ///
    /// See:
    /// - Chapter 9.6.26(i) of Handbook of Mathematical Functions: with Formulas, Graphs, and
    /// Mathematical Tables, by Milton Abramowitz and Irene A. Stegun
    /// - Equation 6.6.4 of Numerical recipes in C: The art of scientific computing, second edition,
    /// by Jaob Winkler (1993)
    double bessi2(double x);

    /// Modified Bessel Function of the First Kind \f$ I_n(x) \f$, with \f$n = 3\f$, using the
    /// recurrence relations, i.e. \f$ I_{n+1}(x) = I_{n-1}(x) - (2 * n / x) I_{n}(x)\f$
    ///
    /// See:
    /// - Chapter 9.6.26(i) of Handbook of Mathematical Functions: with Formulas, Graphs, and
    /// Mathematical Tables, by Milton Abramowitz and Irene A. Stegun
    /// - Equation 6.6.4 of Numerical recipes in C: The art of scientific computing, second edition,
    /// by Jaob Winkler (1993)
    double bessi3(double x);

    /// Modified Bessel Function of the First Kind \f$ I_n(x) \f$, with \f$n = 4\f$, using the
    /// recurrence relations, i.e. \f$ I_{n+1}(x) = I_{n-1}(x) - (2 * n / x) I_{n}(x)\f$
    ///
    /// See:
    /// - Chapter 9.6.26(i) of Handbook of Mathematical Functions: with Formulas, Graphs, and
    /// Mathematical Tables, by Milton Abramowitz and Irene A. Stegun
    /// - Equation 6.6.4 of Numerical recipes in C: The art of scientific computing, second edition,
    /// by Jaob Winkler (1993)
    double bessi4(double x);

    /// See: https://stackoverflow.com/questions/8797722/modified-bessel-functions-of-order-n
    double bessi(index_t m, double x);
} // namespace elsa::math
