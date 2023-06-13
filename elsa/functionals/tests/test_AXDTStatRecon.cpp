/**
* @file test_AXDTStatRecon.cpp
*
* @brief Tests for the AXDTStatRecon class
*
* @author Shen Hu - initial code
*/

#include <doctest/doctest.h>

#include "AXDTStatRecon.h"
#include "LinearResidual.h"
#include "VolumeDescriptor.h"
#include "Identity.h"
#include "Scaling.h"
#include "IdenticalBlocksDescriptor.h"
#include "testHelpers.h"
#include "Logger.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("AXDTStatRecon: Testing reconstruction type - Gaussian_log_d", TestType, float, double)
{
    // eliminate the timing info from console for the tests
    Logger::setLevel(Logger::LogLevel::OFF);
    srand((unsigned int) 666);

    GIVEN("dummy axdt projection data and the corresponding operator")
   {
       IndexVector_t numCoeff(1);
       numCoeff << 12;
       VolumeDescriptor dd(numCoeff);

       Vector_t<TestType> axdt_proj_raw(dd.getNumberOfCoefficients());
       axdt_proj_raw.setRandom();
       DataContainer<TestType> axdt_proj(dd, axdt_proj_raw);

       Vector_t<TestType> axdt_scaling_raw(dd.getNumberOfCoefficients());
       axdt_scaling_raw.setRandom();
       DataContainer<TestType> axdt_scaling_factors(dd, axdt_scaling_raw);

       Scaling<TestType> axdt_op(dd, axdt_scaling_factors);

       WHEN("instantiating AXDTStatRecon instance of recon_type Gaussian_log_d")
       {
           AXDTStatRecon<TestType> func(axdt_proj, axdt_op, AXDTStatRecon<TestType>::Gaussian_log_d);

           THEN("the functional is as expected")
           {
               auto& funcDomainDesc_raw = func.getDomainDescriptor();
               auto* funcDomainDesc = dynamic_cast<const RandomBlocksDescriptor*>(&funcDomainDesc_raw);
               REQUIRE_UNARY(funcDomainDesc);
               REQUIRE_EQ(funcDomainDesc->getDescriptorOfBlock(1), dd);

               auto* linRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
               REQUIRE_UNARY(linRes);
               REQUIRE_UNARY_FALSE(linRes->hasOperator());
               REQUIRE_UNARY_FALSE(linRes->hasDataVector());
           }

           THEN("a clone behaves as expected")
           {
               auto funcClone = func.clone();

               REQUIRE_NE(funcClone.get(), &func);
               REQUIRE_EQ(*funcClone, func);
           }

           THEN("the evaluate, gradient and Hessian work as expected")
           {
               auto x_dd = func.getDomainDescriptor().clone();

               Vector_t<TestType> x_raw(x_dd->getNumberOfCoefficients());
               x_raw.setRandom();
               DataContainer<TestType> x(*x_dd, x_raw);

               REQUIRE_UNARY(checkApproxEq(func.evaluate(x),
                                           square(-axdt_op.apply(x.getBlock(1)) + axdt_proj).sum()));

               auto true_grad = static_cast<TestType>(2.0)
                                * axdt_op.applyAdjoint(axdt_op.apply(x.getBlock(1)) - axdt_proj);
               auto grad = func.getGradient(x);

               REQUIRE_UNARY(checkApproxEq(grad.getBlock(0).sum(), 0));
               for (index_t i = 0; i < axdt_op.getDomainDescriptor().getNumberOfCoefficients(); ++i) {
                   REQUIRE_UNARY(checkApproxEq(grad.getBlock(1)[i], true_grad[i]));
               }

               // generate a random vector y
               Vector_t<TestType> y_raw(x_dd->getNumberOfCoefficients());
               y_raw.setRandom();
               DataContainer<TestType> y(*x_dd, y_raw);

               auto true_hess_on_y = static_cast<TestType>(2.0) * axdt_op.applyAdjoint(axdt_op.apply(y.getBlock(1)));
               auto hess = func.getHessian(x);
               auto hess_on_y = hess.apply(y);

               REQUIRE_UNARY(checkApproxEq(hess_on_y.getBlock(0).sum(), 0));
               for (index_t i = 0; i < axdt_op.getDomainDescriptor().getNumberOfCoefficients(); ++i) {
                   REQUIRE_UNARY(checkApproxEq(true_hess_on_y[i], hess_on_y.getBlock(1)[i]));
               }
           }
       }
   }
}

TEST_CASE_TEMPLATE("AXDTStatRecon: Testing reconstruction type - Gaussian_d", TestType, float, double)
{
   // eliminate the timing info from console for the tests
   Logger::setLevel(Logger::LogLevel::OFF);
   srand((unsigned int) 666);

   GIVEN("dummy axdt projection data and the corresponding operator")
   {
       IndexVector_t numCoeff(1);
       numCoeff << 12;
       VolumeDescriptor dd(numCoeff);

       Vector_t<TestType> axdt_proj_raw(dd.getNumberOfCoefficients());
       axdt_proj_raw.setRandom();
       DataContainer<TestType> axdt_proj(dd, axdt_proj_raw);

       Vector_t<TestType> axdt_scaling_raw(dd.getNumberOfCoefficients());
       axdt_scaling_raw.setRandom();
       DataContainer<TestType> axdt_scaling_factors(dd, axdt_scaling_raw);

       Scaling<TestType> axdt_op(dd, axdt_scaling_factors);

       WHEN("instantiating AXDTStatRecon instance of recon_type Gaussian_d")
       {
           AXDTStatRecon<TestType> func(axdt_proj, axdt_op, AXDTStatRecon<TestType>::Gaussian_d);

           THEN("the functional is as expected")
           {
               auto& funcDomainDesc_raw = func.getDomainDescriptor();
               auto* funcDomainDesc = dynamic_cast<const RandomBlocksDescriptor*>(&funcDomainDesc_raw);
               REQUIRE_UNARY(funcDomainDesc);
               REQUIRE_EQ(funcDomainDesc->getDescriptorOfBlock(1), dd);

               auto* linRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
               REQUIRE_UNARY(linRes);
               REQUIRE_UNARY_FALSE(linRes->hasOperator());
               REQUIRE_UNARY_FALSE(linRes->hasDataVector());
           }

           THEN("a clone behaves as expected")
           {
               auto funcClone = func.clone();

               REQUIRE_NE(funcClone.get(), &func);
               REQUIRE_EQ(*funcClone, func);
           }

           THEN("the evaluate, gradient and Hessian work as expected")
           {
               auto x_dd = func.getDomainDescriptor().clone();

               Vector_t<TestType> x_raw(x_dd->getNumberOfCoefficients());
               x_raw.setRandom();
               DataContainer<TestType> x(*x_dd, x_raw);

               REQUIRE_UNARY(checkApproxEq(func.evaluate(x),
                                           square(exp(-axdt_op.apply(x.getBlock(1))) - exp(-axdt_proj)).sum()));

               auto d = exp(-axdt_op.apply(x.getBlock(1)));
               auto d_tilde = exp(-axdt_proj);

               auto true_grad = -static_cast<TestType>(2.0)
                                * axdt_op.applyAdjoint(d * (d - d_tilde));
               auto grad = func.getGradient(x);

               REQUIRE_UNARY(checkApproxEq(grad.getBlock(0).sum(), 0));
               for (index_t i = 0; i < axdt_op.getDomainDescriptor().getNumberOfCoefficients(); ++i) {
                   REQUIRE_UNARY(checkApproxEq(grad.getBlock(1)[i], true_grad[i]));
               }

               // generate a random vector y
               Vector_t<TestType> y_raw(x_dd->getNumberOfCoefficients());
               y_raw.setRandom();
               DataContainer<TestType> y(*x_dd, y_raw);

               auto true_hess_on_y = static_cast<TestType>(2.0)
                                     * axdt_op.applyAdjoint(d * (static_cast<TestType>(2.0) * d - d_tilde))
                                     * axdt_op.apply(y.getBlock(1));
               auto hess = func.getHessian(x);
               auto hess_on_y = hess.apply(y);

               REQUIRE_UNARY(checkApproxEq(hess_on_y.getBlock(0).sum(), 0));
               for (index_t i = 0; i < axdt_op.getDomainDescriptor().getNumberOfCoefficients(); ++i) {
                   REQUIRE_UNARY(checkApproxEq(true_hess_on_y[i], hess_on_y.getBlock(1)[i]));
               }
           }
       }
   }
}

TEST_CASE_TEMPLATE("AXDTStatRecon: Testing reconstruction type - Gaussian_approximate_racian", TestType, float, double)
{
   // eliminate the timing info from console for the tests
   Logger::setLevel(Logger::LogLevel::OFF);
   srand((unsigned int) 666);

   GIVEN("dummy absorption and axdt projection data and the corresponding operators")
   {
       IndexVector_t numCoeff(2);
       numCoeff << 8, 12;
       VolumeDescriptor dd(numCoeff);

       // --- setting up absorption related input data ---
       Vector_t<TestType> a_tilde_raw(dd.getNumberOfCoefficients());
       a_tilde_raw.setRandom();
       DataContainer<TestType> a_tilde(dd, a_tilde_raw);
       a_tilde = clip(a_tilde, static_cast<TestType>(1e-5), static_cast<TestType>(1e20));

       Vector_t<TestType> ffa_raw(dd.getNumberOfCoefficients());
       ffa_raw.setRandom();
       DataContainer<TestType> ffa(dd, ffa_raw);
       ffa = clip(ffa, static_cast<TestType>(1e-5), static_cast<TestType>(1e20));

       Vector_t<TestType> absorp_scaling_raw(dd.getNumberOfCoefficients());
       absorp_scaling_raw.setRandom();
       DataContainer<TestType> absorp_scaling_factors(dd, absorp_scaling_raw);

       Scaling<TestType> absorp_op(dd, absorp_scaling_factors);

       // --- setting up axdt related input data ---
       Vector_t<TestType> b_tilde_raw(dd.getNumberOfCoefficients());
       b_tilde_raw.setRandom();
       DataContainer<TestType> b_tilde(dd, b_tilde_raw);
       b_tilde = clip(b_tilde, static_cast<TestType>(1e-5), static_cast<TestType>(1e20));

       Vector_t<TestType> ffb_raw(dd.getNumberOfCoefficients());
       ffb_raw.setRandom();
       DataContainer<TestType> ffb(dd, ffb_raw);
       ffb = clip(ffb, static_cast<TestType>(1e-5), static_cast<TestType>(1e20));

       Vector_t<TestType> axdt_scaling_raw(dd.getNumberOfCoefficients());
       axdt_scaling_raw.setRandom();
       DataContainer<TestType> axdt_scaling_factors(dd, axdt_scaling_raw);

       Scaling<TestType> axdt_op(dd, axdt_scaling_factors);

       index_t N {12};

       WHEN("instantiating AXDTStatRecon instance of recon_type Gaussian_approximate_racian")
       {
           AXDTStatRecon<TestType> func(ffa, ffb, a_tilde, b_tilde, absorp_op, axdt_op, N, AXDTStatRecon<TestType>::Gaussian_approximate_racian);

           THEN("the functional is as expected")
           {
               auto& funcDomainDesc_raw = func.getDomainDescriptor();
               auto* funcDomainDesc = dynamic_cast<const RandomBlocksDescriptor*>(&funcDomainDesc_raw);
               REQUIRE_UNARY(funcDomainDesc);
               REQUIRE_EQ(funcDomainDesc->getDescriptorOfBlock(0), dd);
               REQUIRE_EQ(funcDomainDesc->getDescriptorOfBlock(1), dd);

               auto* linRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
               REQUIRE_UNARY(linRes);
               REQUIRE_UNARY_FALSE(linRes->hasOperator());
               REQUIRE_UNARY_FALSE(linRes->hasDataVector());
           }

           THEN("a clone behaves as expected")
           {
               auto funcClone = func.clone();

               REQUIRE_NE(funcClone.get(), &func);
               REQUIRE_EQ(*funcClone, func);
           }

           auto x_dd = func.getDomainDescriptor().clone();

           Vector_t<TestType> x_raw(x_dd->getNumberOfCoefficients());
           x_raw.setRandom();
           DataContainer<TestType> x(*x_dd, x_raw);

           auto alpha = ffb / ffa;
           auto a = exp(-absorp_op.apply(x.getBlock(0))) * ffa;
           auto b = exp(-axdt_op.apply(x.getBlock(1))) * alpha * a;
           auto d = exp(-axdt_op.apply(x.getBlock(1)));

           THEN("the evaluate works as expected")
           {
               auto numerator_1 = static_cast<TestType>(2.0) * static_cast<TestType>(N) * (a_tilde - a) * (a_tilde - a);
               auto numerator_2 = static_cast<TestType>(N) * (b_tilde - a * alpha * d) * (b_tilde - a * alpha * d);
               auto expected_result = - (-log(a) - ((numerator_1 + numerator_2) / a / static_cast<TestType>(4.0))).sum();

               REQUIRE_UNARY(checkApproxEq(func.evaluate(x), expected_result));
           }

           THEN("the gradient works as expected")
           {
               auto numerator = static_cast<TestType>(2.0) * (a * a - (a_tilde * a_tilde))
                                + (b * b) - (b_tilde * b_tilde);
               auto tmp = static_cast<TestType>(N) * numerator / a / static_cast<TestType>(4.0);
               auto true_grad_absorp = - absorp_op.apply(static_cast<TestType>(1.0) + tmp);

               auto true_grad_axdt = - axdt_op.apply(static_cast<TestType>(N) * alpha * d
                                                   * (a * alpha * d - b_tilde) / static_cast<TestType>(2.0));

               auto grad = func.getGradient(x);

               for (index_t i = 0; i < dd.getNumberOfCoefficients(); ++i) {
                   REQUIRE_UNARY(checkApproxEq(grad.getBlock(0)[i], true_grad_absorp[i]));
                   REQUIRE_UNARY(checkApproxEq(grad.getBlock(1)[i], true_grad_axdt[i]));
               }
           }

           THEN("the Hessian works as expected")
           {
               // generate a random vector y
               Vector_t<TestType> y_raw(x_dd->getNumberOfCoefficients());
               y_raw.setRandom();
               DataContainer<TestType> y(*x_dd, y_raw);

               auto tmp_1_1 = static_cast<TestType>(2.0) * a * a;
               tmp_1_1 += a * a * alpha * alpha * d * d;
               tmp_1_1 += static_cast<TestType>(2.0) * a_tilde * a_tilde;
               tmp_1_1 += b_tilde * b_tilde;
               tmp_1_1 /= static_cast<TestType>(4.0) * a;
               auto H_1_1 = - static_cast<TestType>(N)  * tmp_1_1;

               auto tmp_1_2 = alpha * alpha * d * d * a;
               auto H_1_2 = - static_cast<TestType>(N) / static_cast<TestType>(2.0) * tmp_1_2;

               auto tmp_2_2 = alpha * d * (static_cast<TestType>(2.0) * alpha * d * a - b_tilde);
               auto H_2_2 = - static_cast<TestType>(N) / static_cast<TestType>(2.0) * tmp_2_2;

               auto true_hess_on_y_absorp = - absorp_op.applyAdjoint(H_1_1 * absorp_op.apply(y.getBlock(0))
                                                                   + H_1_2 * axdt_op.apply(y.getBlock(1)));

               auto true_hess_on_y_axdt = - axdt_op.applyAdjoint(H_1_2 * absorp_op.apply(y.getBlock(0))
                                                               + H_2_2 * axdt_op.apply(y.getBlock(1)));

               auto hess = func.getHessian(x);
               auto hess_on_y = hess.apply(y);

               for (index_t i = 0; i < dd.getNumberOfCoefficients(); ++i) {
                   CHECK_UNARY(checkApproxEq(hess_on_y.getBlock(0)[i], true_hess_on_y_absorp[i]));
                   CHECK_UNARY(checkApproxEq(hess_on_y.getBlock(1)[i], true_hess_on_y_axdt[i]));
               }
           }
       }
   }
}

TEST_CASE_TEMPLATE("AXDTStatRecon: Testing reconstruction type - Racian_direct", TestType, float, double)
{
   // eliminate the timing info from console for the tests
   Logger::setLevel(Logger::LogLevel::OFF);
   srand((unsigned int) 666);

   GIVEN("dummy absorption and axdt projection data and the corresponding operators")
   {
       IndexVector_t numCoeff(2);
       numCoeff << 8, 12;
       VolumeDescriptor dd(numCoeff);

       // --- setting up absorption related input data ---
       Vector_t<TestType> a_tilde_raw(dd.getNumberOfCoefficients());
       a_tilde_raw.setRandom();
       DataContainer<TestType> a_tilde(dd, a_tilde_raw);
       a_tilde = clip(a_tilde, static_cast<TestType>(1e-5), static_cast<TestType>(1e20));

       Vector_t<TestType> ffa_raw(dd.getNumberOfCoefficients());
       ffa_raw.setRandom();
       DataContainer<TestType> ffa(dd, ffa_raw);
       ffa = clip(ffa, static_cast<TestType>(1e-5), static_cast<TestType>(1e20));

       Vector_t<TestType> absorp_scaling_raw(dd.getNumberOfCoefficients());
       absorp_scaling_raw.setRandom();
       DataContainer<TestType> absorp_scaling_factors(dd, absorp_scaling_raw);

       Scaling<TestType> absorp_op(dd, absorp_scaling_factors);

       // --- setting up axdt related input data ---
       Vector_t<TestType> b_tilde_raw(dd.getNumberOfCoefficients());
       b_tilde_raw.setRandom();
       DataContainer<TestType> b_tilde(dd, b_tilde_raw);
       b_tilde = clip(b_tilde, static_cast<TestType>(1e-5), static_cast<TestType>(1e20));

       Vector_t<TestType> ffb_raw(dd.getNumberOfCoefficients());
       ffb_raw.setRandom();
       DataContainer<TestType> ffb(dd, ffb_raw);
       ffb = clip(ffb, static_cast<TestType>(1e-5), static_cast<TestType>(1e20));

       Vector_t<TestType> axdt_scaling_raw(dd.getNumberOfCoefficients());
       axdt_scaling_raw.setRandom();
       DataContainer<TestType> axdt_scaling_factors(dd, axdt_scaling_raw);

       Scaling<TestType> axdt_op(dd, axdt_scaling_factors);

       index_t N {12};

       WHEN("instantiating AXDTStatRecon instance of recon_type Gaussian_approximate_racian")
       {
           AXDTStatRecon<TestType> func(ffa, ffb, a_tilde, b_tilde, absorp_op, axdt_op, N, AXDTStatRecon<TestType>::Racian_direct);

           THEN("the functional is as expected")
           {
               auto& funcDomainDesc_raw = func.getDomainDescriptor();
               auto* funcDomainDesc = dynamic_cast<const RandomBlocksDescriptor*>(&funcDomainDesc_raw);
               REQUIRE_UNARY(funcDomainDesc);
               REQUIRE_EQ(funcDomainDesc->getDescriptorOfBlock(0), dd);
               REQUIRE_EQ(funcDomainDesc->getDescriptorOfBlock(1), dd);

               auto* linRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
               REQUIRE_UNARY(linRes);
               REQUIRE_UNARY_FALSE(linRes->hasOperator());
               REQUIRE_UNARY_FALSE(linRes->hasDataVector());
           }

           THEN("a clone behaves as expected")
           {
               auto funcClone = func.clone();

               REQUIRE_NE(funcClone.get(), &func);
               REQUIRE_EQ(*funcClone, func);
           }

           auto x_dd = func.getDomainDescriptor().clone();

           Vector_t<TestType> x_raw(x_dd->getNumberOfCoefficients());
           x_raw.setRandom();
           DataContainer<TestType> x(*x_dd, x_raw);

           auto alpha = ffb / ffa;
           auto a = exp(-absorp_op.apply(x.getBlock(0))) * ffa;
           auto b = exp(-axdt_op.apply(x.getBlock(1))) * alpha * a;
           auto d = exp(-axdt_op.apply(x.getBlock(1)));

           THEN("the evaluate works as expected")
           {
               auto z = b_tilde * alpha * d * static_cast<TestType>(N) / static_cast<TestType>(2.0);
               auto term_1 = static_cast<TestType>(-1.5) * log(a);
               auto term_2 = static_cast<TestType>(2.0) * a_tilde * a_tilde;
               term_2 += static_cast<TestType>(2.0) * a * a;
               term_2 += b_tilde * b_tilde;
               term_2 += a * a * alpha * alpha * d * d;
               term_2 *= - static_cast<TestType>(N) / static_cast<TestType>(4.0) / a;
               auto term_3 = axdt::log_bessel_0(z);
               auto expected_result = - (term_1 + term_2 + term_3).sum();

               REQUIRE_UNARY(checkApproxEq(func.evaluate(x), expected_result));
           }

           THEN("the gradient works as expected")
           {
               auto numerator = static_cast<TestType>(2.0) * (a * a - (a_tilde * a_tilde))
                                + (b * b) - (b_tilde * b_tilde);
               auto tmp_absorp = static_cast<TestType>(N) * numerator / a / static_cast<TestType>(4.0);
               auto true_grad_absorp = - absorp_op.apply(static_cast<TestType>(1.5) + tmp_absorp);

               auto tmp_axdt = static_cast<TestType>(0.5) * static_cast<TestType>(N) * a * alpha * alpha * d * d;
               auto z = static_cast<TestType>(0.5) * static_cast<TestType>(N) * b_tilde * alpha * d;
               auto true_grad_axdt = - axdt_op.apply(tmp_axdt - (z * axdt::quot_bessel_1_0(z)));

               auto grad = func.getGradient(x);

               for (index_t i = 0; i < dd.getNumberOfCoefficients(); ++i) {
                   REQUIRE_UNARY(checkApproxEq(grad.getBlock(0)[i], true_grad_absorp[i]));
                   REQUIRE_UNARY(checkApproxEq(grad.getBlock(1)[i], true_grad_axdt[i]));
               }
           }

           THEN("the Hessian works as expected")
           {
               // generate a random vector y
               Vector_t<TestType> y_raw(x_dd->getNumberOfCoefficients());
               y_raw.setRandom();
               DataContainer<TestType> y(*x_dd, y_raw);

               auto tmp_1_1 = static_cast<TestType>(2.0) * a * a;
               tmp_1_1 += a * a * alpha * alpha * d * d;
               tmp_1_1 += static_cast<TestType>(2.0) * a_tilde * a_tilde;
               tmp_1_1 += b_tilde * b_tilde;
               tmp_1_1 /= static_cast<TestType>(4.0) * a;
               auto H_1_1 = - static_cast<TestType>(N)  * tmp_1_1;

               auto tmp_1_2 = alpha * alpha * d * d * a;
               auto H_1_2 = - static_cast<TestType>(N) / static_cast<TestType>(2.0) * tmp_1_2;

               auto z = static_cast<TestType>(0.5) * static_cast<TestType>(N) * b_tilde * alpha * d;
               auto tmp_2_2 = - static_cast<TestType>(N) * alpha * alpha * d * d * a;
               auto H_2_2 = tmp_2_2 + z * z * (static_cast<TestType>(1.0) - square(axdt::quot_bessel_1_0(z)));

               auto true_hess_on_y_absorp = - absorp_op.applyAdjoint(H_1_1 * absorp_op.apply(y.getBlock(0))
                                                                    + H_1_2 * axdt_op.apply(y.getBlock(1)));

               auto true_hess_on_y_axdt = - axdt_op.applyAdjoint(H_1_2 * absorp_op.apply(y.getBlock(0))
                                                                + H_2_2 * axdt_op.apply(y.getBlock(1)));

               auto hess = func.getHessian(x);
               auto hess_on_y = hess.apply(y);

               for (index_t i = 0; i < dd.getNumberOfCoefficients(); ++i) {
                   CHECK_UNARY(checkApproxEq(hess_on_y.getBlock(0)[i], true_hess_on_y_absorp[i]));
                   CHECK_UNARY(checkApproxEq(hess_on_y.getBlock(1)[i], true_hess_on_y_axdt[i]));
               }
           }
       }
   }
}

TEST_SUITE_END();
