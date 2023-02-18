/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for scalar types and constexpr math functions.
 */

#include <gtest/gtest.h>
#include <complex>
#include "basics/basics.hpp"

using namespace OpenKalman;
using namespace OpenKalman::internal;

namespace
{
  template<auto ep>
  auto tolerance = [](const auto& a, const auto& b){ return are_within_tolerance<ep>(a, b); };
}


TEST(basics, scalar_traits)
{
  static_assert(complex_number<std::complex<double>>);
  static_assert(complex_number<std::complex<int>>);
  static_assert(scalar_type<double>);
  static_assert(scalar_type<int>);
  static_assert(scalar_type<std::complex<int>>);
  static_assert(not floating_scalar_type<int>);
  static_assert(floating_scalar_type<float>);
  static_assert(floating_scalar_type<double>);
  static_assert(floating_scalar_type<long double>);
  static_assert(not floating_scalar_type<std::complex<double>>);
  static_assert(not floating_scalar_type<std::complex<int>>);
  static_assert(scalar_constant<int, CompileTimeStatus::unknown>);
  static_assert(scalar_constant<double, CompileTimeStatus::unknown>);
  static_assert(scalar_constant<double, CompileTimeStatus::any>);
  static_assert(scalar_constant<std::integral_constant<int, 5>, CompileTimeStatus::known>);
  static_assert(scalar_constant<std::integral_constant<int, 6>, CompileTimeStatus::any>);
  static_assert(scalar_constant<internal::scalar_constant_operation<std::multiplies<>, double, double>>);
  EXPECT_EQ(get_scalar_constant_value(7), 7);
  EXPECT_EQ(get_scalar_constant_value(std::integral_constant<int, 7>{}), 7);
  EXPECT_EQ(get_scalar_constant_value([](){ return 8; }), 8);
  EXPECT_EQ(get_scalar_constant_value(internal::scalar_constant_operation{[](){ return 9; }}), 9);
  int k = 9; EXPECT_EQ(get_scalar_constant_value(internal::scalar_constant_operation{[&k](){ return k; }}), 9);
  EXPECT_EQ(get_scalar_constant_value(internal::scalar_constant_operation{std::plus{}, 4, 5}), 9);
  static_assert(internal::scalar_constant_operation{std::plus{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 5>{}}() == 9);
}


TEST(basics, scalar_functions)
{
  static_assert(are_within_tolerance(10., 10.));
  static_assert(are_within_tolerance<2>(0., 0. + std::numeric_limits<double>::epsilon()));
  static_assert(not are_within_tolerance<2>(0., 0. + 3 * std::numeric_limits<double>::epsilon()));
  static_assert(std::is_same_v<double, std::decay_t<decltype(real_part(std::declval<double>()))>>);
  static_assert(std::is_same_v<double, std::decay_t<decltype(real_part(std::declval<int>()))>>);
  static_assert(std::is_same_v<double, std::decay_t<decltype(real_part(std::declval<std::complex<double>>()))>>);
  static_assert(std::is_same_v<double, std::decay_t<decltype(real_part(std::declval<std::complex<int>>()))>>);
  static_assert(real_part(std::complex{3.5, 4.5}) == 3.5);
  static_assert(real_part(std::complex{3, 4}) == 3.);
  static_assert(internal::inverse_real_projection(std::complex{3.5, 4.5}, 5.5) == std::complex{5.5, 4.5});
  static_assert(internal::inverse_real_projection(std::complex{3, 4}, 5.2) == std::complex{5, 4}); // truncation occurs
  static_assert(imaginary_part(std::complex{3.5, 4.5}) == 4.5);
  static_assert(conjugate(std::complex{3.5, 4.5}) == std::complex{3.5, -4.5});
  EXPECT_NEAR(std::real(sine(std::complex{3.5, 4.5})), std::real(std::sin(std::complex{3.5, 4.5})), 1e-9);
  EXPECT_NEAR(std::imag(sine(std::complex{3.5, 4.5})), std::imag(std::sin(std::complex{3.5, 4.5})), 1e-9);
}


TEST(basics, constexpr_sqrt)
{
  static_assert(constexpr_sqrt(0) == 0);
  static_assert(constexpr_sqrt(1) == 1);
  static_assert(constexpr_sqrt(4) == 2);
  static_assert(constexpr_sqrt(9) == 3);
  static_assert(constexpr_sqrt(1000000) == 1000);
  static_assert(are_within_tolerance(constexpr_sqrt(2.), numbers::sqrt2));
  static_assert(are_within_tolerance(constexpr_sqrt(3.), numbers::sqrt3));
  static_assert(are_within_tolerance(constexpr_sqrt(4.0e6), 2.0e3));
  static_assert(are_within_tolerance(constexpr_sqrt(9.0e-2), 3.0e-1));
  static_assert(are_within_tolerance(constexpr_sqrt(2.5e-11), 5.0e-6));
  EXPECT_TRUE(std::isnan(constexpr_sqrt(-1)));
  EXPECT_NEAR(constexpr_sqrt(5.), std::sqrt(5), 1e-9);
  EXPECT_NEAR(constexpr_sqrt(1.0e20), std::sqrt(1.0e20), 1e-9);
  EXPECT_NEAR(constexpr_sqrt(0.001), std::sqrt(0.001), 1e-9);
  EXPECT_NEAR(constexpr_sqrt(1e-20), std::sqrt(1e-20), 1e-9);
  EXPECT_PRED2(tolerance<10>, constexpr_sqrt(std::complex<double>{-4}), std::sqrt(std::complex<double>{-4}));
  EXPECT_PRED2(tolerance<10>, constexpr_sqrt(std::complex<double>{3, 4}), std::sqrt(std::complex<double>{3, 4}));
  EXPECT_PRED2(tolerance<10>, constexpr_sqrt(std::complex<double>{3, -4}), std::sqrt(std::complex<double>{3, -4}));
  EXPECT_PRED2(tolerance<10>, constexpr_sqrt(std::complex<double>{-3, 4}), std::sqrt(std::complex<double>{-3, 4}));
  EXPECT_PRED2(tolerance<10>, constexpr_sqrt(std::complex<double>{-3, -4}), std::sqrt(std::complex<double>{-3, -4}));
  EXPECT_PRED2(tolerance<10>, constexpr_sqrt(std::complex<double>{-3e10, 4e10}), std::sqrt(std::complex<double>{-3e10, 4e10}));
  EXPECT_PRED2(tolerance<10>, constexpr_sqrt(std::complex<int>{3, -4}), std::sqrt(std::complex<double>{3, -4}));
}


TEST(basics, constexpr_sin_arithmetic)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;
  static_assert(constexpr_sin(0) == 0);
  static_assert(constexpr_sin(2*pi) == 0);
  static_assert(constexpr_sin(-2*pi) == 0);
  static_assert(are_within_tolerance(constexpr_sin(pi), 0));
  static_assert(are_within_tolerance(constexpr_sin(-pi), 0));
  static_assert(are_within_tolerance(constexpr_sin(32*pi), 0));
  static_assert(are_within_tolerance(constexpr_sin(-32*pi), 0));
  static_assert(are_within_tolerance(constexpr_sin(0x1p16*pi), 0));
  static_assert(are_within_tolerance(constexpr_sin(-0x1p16*pi), 0));
  static_assert(are_within_tolerance(constexpr_sin(0x1p16L*piL), 0));
  static_assert(are_within_tolerance(constexpr_sin(-0x1p16L*piL), 0));
  static_assert(are_within_tolerance(constexpr_sin(0x1p16F*piF), 0));
  static_assert(are_within_tolerance(constexpr_sin(-0x1p16F*piF), 0));
  static_assert(are_within_tolerance(constexpr_sin((std::numeric_limits<std::intmax_t>::max())), 0));
  static_assert(are_within_tolerance(constexpr_sin((std::numeric_limits<std::intmax_t>::lowest())), 0));
  static_assert(are_within_tolerance(constexpr_sin((std::numeric_limits<std::intmax_t>::max())), 0));
  static_assert(are_within_tolerance(constexpr_sin((std::numeric_limits<std::intmax_t>::lowest())), 0));
  static_assert(are_within_tolerance(constexpr_sin((std::numeric_limits<std::intmax_t>::max())), 0));
  static_assert(are_within_tolerance(constexpr_sin((std::numeric_limits<std::intmax_t>::lowest())), 0));
  static_assert(are_within_tolerance(constexpr_sin(0x1p100*pi), 0));
  static_assert(are_within_tolerance(constexpr_sin(-0x1p100*pi), 0));
  static_assert(are_within_tolerance(constexpr_sin(0x1p180*pi), 0));
  static_assert(are_within_tolerance(constexpr_sin(-0x1p180*pi), 0));
  static_assert(are_within_tolerance(constexpr_sin(0x1p250*pi), 0));
  static_assert(are_within_tolerance(constexpr_sin(-0x1p250*pi), 0));
  static_assert(are_within_tolerance(constexpr_sin(pi/2), 1));
  static_assert(are_within_tolerance(constexpr_sin(-pi/2), -1));
  static_assert(are_within_tolerance(constexpr_sin(pi/4), numbers::sqrt2_v<double>/2));
  static_assert(are_within_tolerance(constexpr_sin(piL/4), numbers::sqrt2_v<long double>/2));
  static_assert(are_within_tolerance(constexpr_sin(piF/4), numbers::sqrt2_v<float>/2));
  static_assert(are_within_tolerance<0x10>(constexpr_sin(pi/4 + 32*pi), numbers::sqrt2_v<double>/2));
  EXPECT_NEAR(constexpr_sin(2), std::sin(2), 1e-9);
  EXPECT_NEAR(constexpr_sin(-32), std::sin(-32), 1e-9);
  EXPECT_NEAR(constexpr_sin(0x1p16), std::sin(0x1p16), 1e-9);
}


TEST(basics, constexpr_cos_arithmetic)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;
  static_assert(constexpr_cos(0) == 1);
  static_assert(constexpr_cos(2*pi) == 1);
  static_assert(constexpr_cos(-2*pi) == 1);
  static_assert(are_within_tolerance(constexpr_cos(pi), -1));
  static_assert(are_within_tolerance(constexpr_cos(-pi), -1));
  static_assert(are_within_tolerance(constexpr_cos(32*pi), 1));
  static_assert(are_within_tolerance(constexpr_cos(-32*pi), 1));
  static_assert(are_within_tolerance(constexpr_cos(0x1p16*pi), 1));
  static_assert(are_within_tolerance(constexpr_cos(-0x1p16*pi), 1));
  static_assert(are_within_tolerance(constexpr_cos(0x1p16L*piL), 1));
  static_assert(are_within_tolerance(constexpr_cos(-0x1p16L*piL), 1));
  static_assert(are_within_tolerance(constexpr_cos(0x1p16F*piF), 1));
  static_assert(are_within_tolerance(constexpr_cos(-0x1p16F*piF), 1));
  static_assert(are_within_tolerance(constexpr_cos((std::numeric_limits<std::intmax_t>::max())), 1));
  static_assert(are_within_tolerance(constexpr_cos((std::numeric_limits<std::intmax_t>::lowest())), 1));
  static_assert(are_within_tolerance(constexpr_cos((std::numeric_limits<std::intmax_t>::max())), 1));
  static_assert(are_within_tolerance(constexpr_cos((std::numeric_limits<std::intmax_t>::lowest())), 1));
  static_assert(are_within_tolerance(constexpr_cos((std::numeric_limits<std::intmax_t>::max())), 1));
  static_assert(are_within_tolerance(constexpr_cos((std::numeric_limits<std::intmax_t>::lowest())), 1));
  static_assert(are_within_tolerance(constexpr_cos(0x1p100*pi), 1));
  static_assert(are_within_tolerance(constexpr_cos(-0x1p100*pi), 1));
  static_assert(are_within_tolerance(constexpr_cos(0x1p180*pi), 1));
  static_assert(are_within_tolerance(constexpr_cos(-0x1p180*pi), 1));
  static_assert(are_within_tolerance(constexpr_cos(0x1p250*pi), 1));
  static_assert(are_within_tolerance(constexpr_cos(-0x1p250*pi), 1));
  static_assert(are_within_tolerance(constexpr_cos(pi/2), 0));
  static_assert(are_within_tolerance(constexpr_cos(-pi/2), 0));
  static_assert(are_within_tolerance(constexpr_cos(pi/4), numbers::sqrt2_v<double>/2));
  static_assert(are_within_tolerance(constexpr_cos(piL/4), numbers::sqrt2_v<long double>/2));
  static_assert(are_within_tolerance(constexpr_cos(piF/4), numbers::sqrt2_v<float>/2));
  EXPECT_NEAR(constexpr_cos(2), std::cos(2), 1e-9);
  EXPECT_NEAR(constexpr_cos(-32), std::cos(-32), 1e-9);
  EXPECT_NEAR(constexpr_cos(0x1p16), std::cos(0x1p16), 1e-9);
}


TEST(basics, constexpr_tan_arithmetic)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;
  static_assert(constexpr_tan(0) == 0);
  static_assert(constexpr_tan(2*pi) == 0);
  static_assert(constexpr_tan(-2*pi) == 0);
  static_assert(are_within_tolerance(constexpr_tan(pi), 0));
  static_assert(are_within_tolerance(constexpr_tan(-pi), 0));
  static_assert(are_within_tolerance(constexpr_tan(32*pi), 0));
  static_assert(are_within_tolerance(constexpr_tan(-32*pi), 0));
  static_assert(are_within_tolerance(constexpr_tan(0x1p16*pi), 0));
  static_assert(are_within_tolerance(constexpr_tan(-0x1p16*pi), 0));
  static_assert(are_within_tolerance(constexpr_tan(0x1p16L*piL), 0));
  static_assert(are_within_tolerance(constexpr_tan(-0x1p16L*piL), 0));
  static_assert(are_within_tolerance(constexpr_tan(0x1p16F*piF), 0));
  static_assert(are_within_tolerance(constexpr_tan(-0x1p16F*piF), 0));
  static_assert(are_within_tolerance(constexpr_tan((std::numeric_limits<std::intmax_t>::max())), 0));
  static_assert(are_within_tolerance(constexpr_tan((std::numeric_limits<std::intmax_t>::lowest())), 0));
  static_assert(are_within_tolerance(constexpr_tan((std::numeric_limits<std::intmax_t>::max())), 0));
  static_assert(are_within_tolerance(constexpr_tan((std::numeric_limits<std::intmax_t>::lowest())), 0));
  static_assert(are_within_tolerance(constexpr_tan((std::numeric_limits<std::intmax_t>::max())), 0));
  static_assert(are_within_tolerance(constexpr_tan((std::numeric_limits<std::intmax_t>::lowest())), 0));
  static_assert(are_within_tolerance(constexpr_tan(0x1p100*pi), 0));
  static_assert(are_within_tolerance(constexpr_tan(-0x1p100*pi), 0));
  static_assert(are_within_tolerance(constexpr_tan(0x1p180*pi), 0));
  static_assert(are_within_tolerance(constexpr_tan(-0x1p180*pi), 0));
  static_assert(are_within_tolerance(constexpr_tan(0x1p250*pi), 0));
  static_assert(are_within_tolerance(constexpr_tan(-0x1p250*pi), 0));
  static_assert(are_within_tolerance(constexpr_tan(pi/4), 1));
  static_assert(are_within_tolerance(constexpr_tan(piL/4), 1));
  static_assert(are_within_tolerance(constexpr_tan(piF/4), 1));
  static_assert(are_within_tolerance<0x20>(constexpr_tan(pi/4 + 32*pi), 1));
  EXPECT_NEAR(constexpr_tan(2), std::tan(2), 1e-9);
  EXPECT_NEAR(constexpr_tan(-32), std::tan(-32), 1e-9);
  EXPECT_NEAR(constexpr_tan(0x1p16), std::tan(0x1p16), 1e-9);
}


TEST(basics, constexpr_exp)
{
  constexpr auto e = numbers::e_v<double>;
  constexpr auto eL = numbers::e_v<long double>;

  static_assert(constexpr_exp(0) == 1);
  static_assert(constexpr_exp(1) == e);
  static_assert(constexpr_exp(2) == e*e);
  static_assert(constexpr_exp(3) == e*e*e);
  static_assert(constexpr_exp<long double>(1) == eL);
  static_assert(constexpr_exp<long double>(2) == eL*eL);
  static_assert(constexpr_exp<long double>(3) == eL*eL*eL);
  static_assert(constexpr_exp(-1) == 1 / e);
  EXPECT_NEAR(constexpr_exp(5), std::exp(5), 1e-9);
  EXPECT_NEAR(constexpr_exp(-10), std::exp(-10), 1e-9);
  EXPECT_NEAR(constexpr_exp(50), std::exp(50), 1e8);
  EXPECT_NEAR(constexpr_exp(50.5), std::exp(50.5), 1e8);
  EXPECT_NEAR(constexpr_exp(300), std::exp(300), 1e120);
  EXPECT_NEAR(constexpr_exp(300.7), std::exp(300.7), 1e120);
  EXPECT_NEAR(constexpr_exp(1e-5), std::exp(1e-5), 1e-12);
  EXPECT_NEAR(constexpr_exp(1e-10), std::exp(1e-10), 1e-16);
  EXPECT_PRED2(tolerance<1000>, constexpr_exp(std::complex<double>{3.3, -4.3}), std::exp(std::complex<double>{3.3, -4.3}));
  EXPECT_PRED2(tolerance<1000000>, constexpr_exp(std::complex<double>{10.4, 3.4}), std::exp(std::complex<double>{10.4, 3.4}));
  EXPECT_PRED2(tolerance<2>, constexpr_exp(std::complex<double>{-30.6, 20.6}), std::exp(std::complex<double>{-30.6, 20.6}));
  EXPECT_PRED2(tolerance<100>, constexpr_exp(std::complex<int>{3, -4}), std::exp(std::complex<double>{3, -4}));

  static_assert(constexpr_expm1(0) == 0);
  static_assert(are_within_tolerance(constexpr_expm1(1), e - 1));
  static_assert(are_within_tolerance<0x8>(constexpr_expm1(2), e * e - 1));
  static_assert(are_within_tolerance<0x10>(constexpr_expm1(3), e * e * e - 1));
  static_assert(are_within_tolerance<0x8>(constexpr_expm1(-1), 1 / e - 1));
  EXPECT_PRED2(tolerance<10>, constexpr_expm1(1e-4), std::expm1(1e-4));
  EXPECT_PRED2(tolerance<10>, constexpr_expm1(1e-8), std::expm1(1e-8));
  EXPECT_PRED2(tolerance<10>, constexpr_expm1(1e-32), std::expm1(1e-32));
  EXPECT_PRED2(tolerance<1000>, constexpr_expm1(5.2), std::expm1(5.2));
  EXPECT_PRED2(tolerance<100000>, constexpr_expm1(10.2), std::expm1(10.2));
  EXPECT_PRED2(tolerance<10>, constexpr_expm1(-10.2), std::expm1(-10.2));
  EXPECT_PRED2(tolerance<100>, constexpr_expm1(std::complex<double>{0.001, -0.001}), std::exp(std::complex<double>{0.001, -0.001}) - 1.0);
  EXPECT_PRED2(tolerance<1000>, constexpr_expm1(std::complex<double>{3.2, -4.2}), std::exp(std::complex<double>{3.2, -4.2}) - 1.0);
  EXPECT_PRED2(tolerance<1000000>, constexpr_expm1(std::complex<double>{10.3, 3.3}), std::exp(std::complex<double>{10.3, 3.3}) - 1.0);
  EXPECT_PRED2(tolerance<100000>, constexpr_expm1(std::complex<double>{-10.4, 10.4}), std::exp(std::complex<double>{-10.4, 10.4}) - 1.0);
  EXPECT_PRED2(tolerance<100>, constexpr_expm1(std::complex<int>{3, -4}), std::exp(std::complex<double>{3, -4}) - 1.0);
}


TEST(basics, constexpr_sinh)
{
  constexpr auto e = numbers::e_v<double>;
  static_assert(constexpr_sinh(0) == 0);
  static_assert(are_within_tolerance(constexpr_sinh(1), (e - 1/e)/2));
  static_assert(are_within_tolerance(constexpr_sinh(2), (e*e - 1/e/e)/2));
  static_assert(are_within_tolerance<8>(constexpr_sinh(3), (e*e*e - 1/e/e/e)/2));
  static_assert(are_within_tolerance(constexpr_sinh(-1), (1/e - e)/2));
  static_assert(are_within_tolerance(constexpr_sinh(-2), (1/e/e - e*e)/2));
  static_assert(are_within_tolerance<8>(constexpr_sinh(-3), (1/e/e/e - e*e*e)/2));
  EXPECT_NEAR(constexpr_sinh(5), std::sinh(5), 1e-9);
  EXPECT_NEAR(constexpr_sinh(-10), std::sinh(-10), 1e-9);
  EXPECT_PRED2(tolerance<1000>, constexpr_sinh(std::complex<double>{3.3, -4.3}), std::sinh(std::complex<double>{3.3, -4.3}));
  EXPECT_PRED2(tolerance<1000000>, constexpr_sinh(std::complex<double>{10.4, 3.4}), std::sinh(std::complex<double>{10.4, 3.4}));
  EXPECT_PRED2(tolerance<100000>, constexpr_sinh(std::complex<double>{-10.6, 10.6}), std::sinh(std::complex<double>{-10.6, 10.6}));
  EXPECT_PRED2(tolerance<100>, constexpr_sinh(std::complex<int>{3, -4}), std::sinh(std::complex<double>{3, -4}));

}


TEST(basics, constexpr_cosh)
{
  constexpr auto e = numbers::e_v<double>;
  static_assert(constexpr_cosh(0) == 1);
  static_assert(are_within_tolerance(constexpr_cosh(1), (e + 1/e)/2));
  static_assert(are_within_tolerance(constexpr_cosh(2), (e*e + 1/e/e)/2));
  static_assert(are_within_tolerance<8>(constexpr_cosh(3), (e*e*e + 1/e/e/e)/2));
  static_assert(are_within_tolerance(constexpr_cosh(-1), (1/e + e)/2));
  static_assert(are_within_tolerance(constexpr_cosh(-2), (1/e/e + e*e)/2));
  static_assert(are_within_tolerance<8>(constexpr_cosh(-3), (1/e/e/e + e*e*e)/2));
  EXPECT_NEAR(constexpr_cosh(5), std::cosh(5), 1e-9);
  EXPECT_NEAR(constexpr_cosh(-10), std::cosh(-10), 1e-9);
  EXPECT_PRED2(tolerance<1000>, constexpr_cosh(std::complex<double>{3.3, -4.3}), std::cosh(std::complex<double>{3.3, -4.3}));
  EXPECT_PRED2(tolerance<1000000>, constexpr_cosh(std::complex<double>{10.4, 3.4}), std::cosh(std::complex<double>{10.4, 3.4}));
  EXPECT_PRED2(tolerance<100000>, constexpr_cosh(std::complex<double>{-10.6, 10.6}), std::cosh(std::complex<double>{-10.6, 10.6}));
  EXPECT_PRED2(tolerance<100>, constexpr_cosh(std::complex<int>{3, -4}), std::cosh(std::complex<double>{3, -4}));

}


TEST(basics, constexpr_tanh)
{
  constexpr auto e = numbers::e_v<double>;
  static_assert(constexpr_tanh(0) == 0);
  static_assert(are_within_tolerance(constexpr_tanh(1), (e*e - 1)/(e*e + 1)));
  static_assert(are_within_tolerance(constexpr_tanh(2), (e*e*e*e - 1)/(e*e*e*e + 1)));
  static_assert(are_within_tolerance(constexpr_tanh(3), (e*e*e*e*e*e - 1)/(e*e*e*e*e*e + 1)));
  static_assert(are_within_tolerance(constexpr_tanh(-1), (1 - e*e)/(1 + e*e)));
  static_assert(are_within_tolerance(constexpr_tanh(-2), (1 - e*e*e*e)/(1 + e*e*e*e)));
  static_assert(are_within_tolerance(constexpr_tanh(-3), (1 - e*e*e*e*e*e)/(1 + e*e*e*e*e*e)));
  EXPECT_NEAR(constexpr_tanh(5), std::tanh(5), 1e-9);
  EXPECT_NEAR(constexpr_tanh(-10), std::tanh(-10), 1e-9);
  EXPECT_PRED2(tolerance<1000>, constexpr_tanh(std::complex<double>{3.3, -4.3}), std::tanh(std::complex<double>{3.3, -4.3}));
  EXPECT_PRED2(tolerance<1000000>, constexpr_tanh(std::complex<double>{10.4, 3.4}), std::tanh(std::complex<double>{10.4, 3.4}));
  EXPECT_PRED2(tolerance<2>, constexpr_tanh(std::complex<double>{-30.6, 20.6}), std::tanh(std::complex<double>{-30.6, 20.6}));
  EXPECT_PRED2(tolerance<100>, constexpr_tanh(std::complex<int>{3, -4}), std::tanh(std::complex<double>{3, -4}));
}


TEST(basics, constexpr_sin_complex)
{
  EXPECT_PRED2(tolerance<100>, constexpr_sin(std::complex<double>{4.1, 3.1}), std::sin(std::complex<double>{4.1, 3.1}));
  EXPECT_PRED2(tolerance<100>, constexpr_sin(std::complex<double>{3.2, -4.2}), std::sin(std::complex<double>{3.2, -4.2}));
  EXPECT_PRED2(tolerance<100>, constexpr_sin(std::complex<double>{-3.3, 4.3}), std::sin(std::complex<double>{-3.3, 4.3}));
  EXPECT_PRED2(tolerance<100000>, constexpr_sin(std::complex<double>{-9.3, 10.3}), std::sin(std::complex<double>{-9.3, 10.3}));
  EXPECT_PRED2(tolerance<100>, constexpr_sin(std::complex<int>{3, -4}), std::sin(std::complex<double>{3, -4}));
}


TEST(basics, constexpr_cos_complex)
{
  EXPECT_PRED2(tolerance<100>, constexpr_cos(std::complex<double>{4.1, 3.1}), std::cos(std::complex<double>{4.1, 3.1}));
  EXPECT_PRED2(tolerance<100>, constexpr_cos(std::complex<double>{3.2, -4.2}), std::cos(std::complex<double>{3.2, -4.2}));
  EXPECT_PRED2(tolerance<100>, constexpr_cos(std::complex<double>{-3.3, 4.3}), std::cos(std::complex<double>{-3.3, 4.3}));
  EXPECT_PRED2(tolerance<100000>, constexpr_cos(std::complex<double>{-9.3, 10.3}), std::cos(std::complex<double>{-9.3, 10.3}));
  EXPECT_PRED2(tolerance<100>, constexpr_cos(std::complex<int>{3, -4}), std::cos(std::complex<double>{3, -4}));
}


TEST(basics, constexpr_tan_complex)
{
  EXPECT_PRED2(tolerance<100>, constexpr_tan(std::complex<double>{4.1, 3.1}), std::tan(std::complex<double>{4.1, 3.1}));
  EXPECT_PRED2(tolerance<100>, constexpr_tan(std::complex<double>{3.2, -4.2}), std::tan(std::complex<double>{3.2, -4.2}));
  EXPECT_PRED2(tolerance<100>, constexpr_tan(std::complex<double>{-3.3, 4.3}), std::tan(std::complex<double>{-3.3, 4.3}));
  EXPECT_PRED2(tolerance<10000000>, constexpr_tan(std::complex<double>{-30.3, 40.3}), std::tan(std::complex<double>{-30.3, 40.3}));
  EXPECT_PRED2(tolerance<100>, constexpr_tan(std::complex<int>{3, -4}), std::tan(std::complex<double>{3, -4}));
}


TEST(basics, constexpr_asin_arithmetic)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;
  static_assert(constexpr_asin(0) == 0);
  static_assert(constexpr_asin(1) == pi/2);
  static_assert(constexpr_asin(1.0L) == piL/2);
  static_assert(constexpr_asin(1.0F) == piF/2);
  static_assert(constexpr_asin(-1) == -pi/2);
  static_assert(are_within_tolerance(constexpr_asin(numbers::sqrt2_v<double>/2), pi/4));
  EXPECT_NEAR(constexpr_asin(numbers::sqrt2_v<double>/2), pi/4, 1e-9);
  EXPECT_NEAR(constexpr_asin(-0.7), std::asin(-0.7), 1e-9);
  EXPECT_NEAR(constexpr_asin(0.9), std::asin(0.9), 1e-9);
  EXPECT_NEAR(constexpr_asin(0.999), std::asin(0.999), 1e-9);
  EXPECT_NEAR(constexpr_asin(-0.999), std::asin(-0.999), 1e-9);
}


TEST(basics, constexpr_acos_arithmetic)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;
  static_assert(constexpr_acos(0) == pi/2);
  static_assert(constexpr_acos(1) == 0);
  static_assert(constexpr_acos(-1) == pi);
  static_assert(constexpr_acos(-1.0L) == piL);
  static_assert(constexpr_acos(-1.0F) == piF);
  static_assert(are_within_tolerance(constexpr_acos(0.5), numbers::pi/3));
  static_assert(are_within_tolerance(constexpr_acos(-0.5), 2*numbers::pi/3));
  static_assert(are_within_tolerance(constexpr_acos(numbers::sqrt2_v<double>/2), pi/4));
  EXPECT_NEAR(constexpr_acos(-0.7), std::acos(-0.7), 1e-9);
  EXPECT_NEAR(constexpr_acos(0.9), std::acos(0.9), 1e-9);
  EXPECT_NEAR(constexpr_acos(0.999), std::acos(0.999), 1e-9);
  EXPECT_NEAR(constexpr_acos(-0.999), std::acos(-0.999), 1e-9);
}


TEST(basics, constexpr_atan_arithmetic)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;
  static_assert(constexpr_atan(0) == 0);
  static_assert(are_within_tolerance(constexpr_atan(1.), pi/4));
  static_assert(are_within_tolerance(constexpr_atan(-1.), -pi/4));
  static_assert(are_within_tolerance(constexpr_atan(-1.L), -piL/4));
  static_assert(are_within_tolerance(constexpr_atan(-1.F), -piF/4));
  EXPECT_NEAR(constexpr_atan(-0.7), std::atan(-0.7), 1e-9);
  EXPECT_NEAR(constexpr_atan(0.9), std::atan(0.9), 1e-9);
  EXPECT_NEAR(constexpr_atan(5.0), std::atan(5.0), 1e-9);
  EXPECT_NEAR(constexpr_atan(-10.0), std::atan(-10.0), 1e-9);
  EXPECT_NEAR(constexpr_atan(100.0), std::atan(100.0), 1e-9);
}


TEST(basics, constexpr_atan2_arithmetic)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;
  constexpr auto inf = std::numeric_limits<double>::infinity();
  static_assert(constexpr_atan2(0, 1) == 0);
  static_assert(constexpr_atan2(0, -1) == pi);
  static_assert(constexpr_atan2(1, 0) == pi/2);
  static_assert(constexpr_atan2(-1, 0) == -pi/2);
  static_assert(are_within_tolerance(constexpr_atan2(0.5, 0.5), pi/4));
  static_assert(are_within_tolerance(constexpr_atan2(1., -1.), 3*pi/4));
  static_assert(are_within_tolerance(constexpr_atan2(-0.5, 0.5), -pi/4));
  static_assert(are_within_tolerance(constexpr_atan2(-1.L, -1.L), -3*piL/4));
  static_assert(are_within_tolerance(constexpr_atan2(-1.F, -1.F), -3*piF/4));
  static_assert(constexpr_atan2(0, 0) == 0);
  static_assert(constexpr_atan2(inf, 0.) == pi/2);
  static_assert(constexpr_atan2(-inf, 0.) == -pi/2);
  static_assert(constexpr_atan2(0., inf) == 0);
  static_assert(constexpr_atan2(0., -inf) == pi);
  static_assert(constexpr_atan2(inf, inf) == pi/4);
  static_assert(constexpr_atan2(inf, -inf) == 3*pi/4);
  static_assert(constexpr_atan2(-inf, inf) == -pi/4);
  static_assert(constexpr_atan2(-inf, -inf) == -3*pi/4);
  EXPECT_NEAR(constexpr_atan2(-0.7, 4.5), std::atan2(-0.7, 4.5), 1e-9);
  EXPECT_NEAR(constexpr_atan2(0.9, -2.3), std::atan2(0.9, -2.3), 1e-9);
  EXPECT_NEAR(constexpr_atan2(5.0, 3.1), std::atan2(5.0, 3.1), 1e-9);
  EXPECT_NEAR(constexpr_atan2(-10.0, 9.0), std::atan2(-10.0, 9.0), 1e-9);
  EXPECT_NEAR(constexpr_atan2(100.0, 200.0), std::atan2(100.0, 200.0), 1e-9);
}


TEST(basics, constexpr_log)
{
  constexpr auto e = numbers::e_v<double>;
  static_assert(constexpr_log(1) == 0);
  static_assert(constexpr_log(e * e) == 2);
  static_assert(constexpr_log(e*e*e*e*e*e*e*e*e*e*e*e*e*e*e*e) == 16);
  static_assert(are_within_tolerance(constexpr_log(1), 0));
  static_assert(are_within_tolerance(constexpr_log(e), 1));
  static_assert(are_within_tolerance<10>(constexpr_log(e * e), 2));
  static_assert(are_within_tolerance<20>(constexpr_log(e * e * e), 3));
  static_assert(are_within_tolerance<10>(constexpr_log(1 / e), -1));
  EXPECT_NEAR(constexpr_log(5.0L), std::log(5.0L), 1e-9);
  EXPECT_NEAR(constexpr_log(0.2L), std::log(0.2L), 1e-9);
  EXPECT_NEAR(constexpr_log(5), std::log(5), 1e-9);
  EXPECT_NEAR(constexpr_log(0.2), std::log(0.2), 1e-9);
  EXPECT_NEAR(constexpr_log(20), std::log(20), 1e-9);
  EXPECT_NEAR(constexpr_log(0.05), std::log(0.05), 1e-9);
  EXPECT_NEAR(constexpr_log(100), std::log(100), 1e-9);
  EXPECT_NEAR(constexpr_log(0.01), std::log(0.01), 1e-9);
  EXPECT_NEAR(constexpr_log(1e20), std::log(1e20), 1e-9);
  EXPECT_NEAR(constexpr_log(1e-20), std::log(1e-20), 1e-9);
  EXPECT_NEAR(constexpr_log(1e200), std::log(1e200), 1e-9);
  EXPECT_NEAR(constexpr_log(1e-200), std::log(1e-200), 1e-9);
  EXPECT_NEAR(constexpr_log(1e200L), std::log(1e200L), 1e-9);
  EXPECT_NEAR(constexpr_log(1e-200L), std::log(1e-200L), 1e-9);
  EXPECT_PRED2(tolerance<10>, constexpr_log(std::complex<double>{-4}), std::log(std::complex<double>{-4}));
  EXPECT_PRED2(tolerance<10>, constexpr_log(std::complex<double>{3, 4}), std::log(std::complex<double>{3, 4}));
  EXPECT_PRED2(tolerance<10>, constexpr_log(std::complex<double>{3, -4}), std::log(std::complex<double>{3, -4}));
  EXPECT_PRED2(tolerance<10>, constexpr_log(std::complex<double>{-3, 4}), std::log(std::complex<double>{-3, 4}));
  EXPECT_PRED2(tolerance<10>, constexpr_log(std::complex<double>{-3, -4}), std::log(std::complex<double>{-3, -4}));
  EXPECT_PRED2(tolerance<10>, constexpr_log(std::complex<int>{3, -4}), std::log(std::complex<double>{3, -4}));
}


TEST(basics, constexpr_asin_complex)
{
  EXPECT_PRED2(tolerance<100>, constexpr_asin(std::complex<double>{4.1, 3.1}), std::asin(std::complex<double>{4.1, 3.1}));
  EXPECT_PRED2(tolerance<100>, constexpr_asin(std::complex<double>{3.2, -4.2}), std::asin(std::complex<double>{3.2, -4.2}));
  EXPECT_PRED2(tolerance<100>, constexpr_asin(std::complex<double>{-3.3, 4.3}), std::asin(std::complex<double>{-3.3, 4.3}));
  EXPECT_PRED2(tolerance<100>, constexpr_asin(std::complex<double>{-9.3, 10.3}), std::asin(std::complex<double>{-9.3, 10.3}));
  EXPECT_PRED2(tolerance<100>, constexpr_asin(std::complex<int>{3, -4}), std::asin(std::complex<double>{3, -4}));
}


TEST(basics, constexpr_acos_complex)
{
  EXPECT_PRED2(tolerance<100>, constexpr_acos(std::complex<double>{4.1, 3.1}), std::acos(std::complex<double>{4.1, 3.1}));
  EXPECT_PRED2(tolerance<100>, constexpr_acos(std::complex<double>{3.2, -4.2}), std::acos(std::complex<double>{3.2, -4.2}));
  EXPECT_PRED2(tolerance<100>, constexpr_acos(std::complex<double>{-3.3, 4.3}), std::acos(std::complex<double>{-3.3, 4.3}));
  EXPECT_PRED2(tolerance<100>, constexpr_acos(std::complex<double>{-9.3, 10.3}), std::acos(std::complex<double>{-9.3, 10.3}));
  EXPECT_PRED2(tolerance<100>, constexpr_acos(std::complex<int>{3, -4}), std::acos(std::complex<double>{3, -4}));
}


TEST(basics, constexpr_atan_complex)
{
  EXPECT_PRED2(tolerance<100>, constexpr_atan(std::complex<double>{4.1, 3.1}), std::atan(std::complex<double>{4.1, 3.1}));
  EXPECT_PRED2(tolerance<100>, constexpr_atan(std::complex<double>{3.2, -4.2}), std::atan(std::complex<double>{3.2, -4.2}));
  EXPECT_PRED2(tolerance<100>, constexpr_atan(std::complex<double>{-3.3, 4.3}), std::atan(std::complex<double>{-3.3, 4.3}));
  EXPECT_PRED2(tolerance<1000>, constexpr_atan(std::complex<double>{-9.3, 10.3}), std::atan(std::complex<double>{-9.3, 10.3}));
  EXPECT_PRED2(tolerance<100>, constexpr_atan(std::complex<int>{3, -4}), std::atan(std::complex<double>{3, -4}));
}


TEST(basics, constexpr_atans_complex)
{
  constexpr auto pi = numbers::pi_v<double>;
  EXPECT_PRED2(tolerance<100>, constexpr_atan2(std::complex<double>{4.1, 3.1}, std::complex<double>{2.1, 5.1}), std::atan(std::complex<double>{4.1, 3.1} / std::complex<double>{2.1, 5.1}));
  EXPECT_PRED2(tolerance<100>, constexpr_atan2(std::complex<double>{3.2, -4.2}, std::complex<double>{-4.1, 3.1}), std::atan(std::complex<double>{3.2, -4.2} / std::complex<double>{-4.1, 3.1}) + pi);
  EXPECT_PRED2(tolerance<100>, constexpr_atan2(std::complex<double>{-3.3, 4.3}, std::complex<double>{2.1, 5.1}), std::atan(std::complex<double>{-3.3, 4.3} / std::complex<double>{2.1, 5.1}));
  EXPECT_PRED2(tolerance<1000>, constexpr_atan2(std::complex<double>{-9.3, 10.3}, std::complex<double>{-5.1, 2.1}), std::atan(std::complex<double>{-9.3, 10.3} / std::complex<double>{-5.1, 2.1}) - pi);
  EXPECT_PRED2(tolerance<100>, constexpr_atan2(std::complex<double>{0., 0.}, std::complex<double>{0., 0.}), std::complex<double>{0});
  EXPECT_PRED2(tolerance<100>, constexpr_atan2(std::complex<double>{0., 3.1}, std::complex<double>{2.1, 5.1}), std::atan(std::complex<double>{0., 3.1} / std::complex<double>{2.1, 5.1}));
  EXPECT_PRED2(tolerance<100>, constexpr_atan2(std::complex<double>{0., 3.1}, std::complex<double>{-2.1, 5.1}), std::atan(std::complex<double>{0., 3.1} / std::complex<double>{-2.1, 5.1}) + pi);
  EXPECT_PRED2(tolerance<100>, constexpr_atan2(std::complex<double>{4.1, 3.1}, std::complex<double>{0., 5.1}), std::atan(std::complex<double>{4.1, 3.1} / std::complex<double>{0., 5.1}));
  EXPECT_PRED2(tolerance<100>, constexpr_atan2(std::complex<double>{-4.1, 3.1}, std::complex<double>{0., 5.1}), std::atan(std::complex<double>{-4.1, 3.1} / std::complex<double>{0., 5.1}));
  EXPECT_PRED2(tolerance<100>, constexpr_atan2(std::complex<int>{3, -4}, std::complex<int>{2, 5}), std::atan(std::complex<double>{3, -4} / std::complex<double>{2, 5}));
}


TEST(basics, constexpr_pow)
{
  static_assert(constexpr_pow(1, 0U) == 1);
  static_assert(constexpr_pow(0, 1U) == 0);
  static_assert(constexpr_pow(1, 1U) == 1);
  static_assert(constexpr_pow(1, 2U) == 1);
  static_assert(constexpr_pow(2, 1U) == 2);
  static_assert(constexpr_pow(2, 5U) == 32);
  static_assert(constexpr_pow(2, 16U) == 65536);
  static_assert(std::is_integral_v<decltype(constexpr_pow(2, 16U))>);
  static_assert(constexpr_pow(2.0, 16U) == 65536);
  static_assert(std::is_floating_point_v<decltype(constexpr_pow(2.0, 16U))>);
  static_assert(constexpr_pow(2, -4) == 0.0625);
  static_assert(constexpr_pow(2, -5) == 0.03125);
  static_assert(std::is_floating_point_v<decltype(constexpr_pow(2, -4))>);
  static_assert(are_within_tolerance(constexpr_pow(2, -4.), 0.0625));
  static_assert(are_within_tolerance(constexpr_pow(10, -4.), 1e-4));
  static_assert(are_within_tolerance<10000000>(constexpr_pow(10., 6.), 1e6));
  EXPECT_DOUBLE_EQ(constexpr_pow(5.0L, 4.0L), std::pow(5.0L, 4.0L));
  EXPECT_DOUBLE_EQ(constexpr_pow(5.0L, -4.0L), std::pow(5.0L, -4.0L));
  EXPECT_DOUBLE_EQ(constexpr_pow(1e20L, 2.L), std::pow(1e20L, 2.L));
  EXPECT_DOUBLE_EQ(constexpr_pow(1e20L, -2.L), std::pow(1e20L, -2.L));
  EXPECT_DOUBLE_EQ(constexpr_pow(1e100L, 2.L), std::pow(1e100L, 2.L));
  EXPECT_DOUBLE_EQ(constexpr_pow(1e100L, -2.L), std::pow(1e100L, -2.L));
  EXPECT_PRED2(tolerance<10>, constexpr_pow(2., std::complex<double>{-4}), std::pow(2., std::complex<double>{-4}));
  EXPECT_PRED2(tolerance<100>, constexpr_pow(std::complex<double>{3, 4}, 2.), std::pow(std::complex<double>{3, 4}, 2.));
  EXPECT_PRED2(tolerance<100>, constexpr_pow(2., std::complex<double>{3, -4}), std::pow(2., std::complex<double>{3, -4}));
  EXPECT_PRED2(tolerance<10>, constexpr_pow(std::complex<double>{1, 2}, std::complex<double>{-3, 4}), std::pow(std::complex<double>{1, 2}, std::complex<double>{-3, 4}));
  EXPECT_PRED2(tolerance<10000>, constexpr_pow(std::complex<double>{-3, -4}, std::complex<double>{1, 2}), std::pow(std::complex<double>{-3, -4}, std::complex<double>{1, 2}));
  EXPECT_PRED2(tolerance<100>, constexpr_pow(std::complex<int>{3, -4}, std::complex<double>{1, 2}), std::pow(std::complex<double>{3, -4}, std::complex<double>{1, 2}));
}

