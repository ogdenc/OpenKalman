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


#if defined(__GNUC__) or defined(__clang__)
#define COMPLEXINTEXISTS(F) F
#else
#define COMPLEXINTEXISTS(F)
#endif


namespace
{
  auto tolerance = [](const auto& a, const auto& b, const auto& err){ return internal::are_within_tolerance(a, b, err); };

  struct NullaryFunc { constexpr auto operator()() { return 5.5; } };
  struct ConstDefinitely : std::integral_constant<int, 6> { static constexpr auto status = Likelihood::definitely; };
  struct ConstMaybe : std::integral_constant<int, 6> { static constexpr auto status = Likelihood::maybe; };
  struct ConstNoStatus : std::integral_constant<int, 6> {};
}


TEST(basics, ScalarConstant)
{
  static_assert(internal::ScalarConstant<Likelihood::definitely, double, 3>{}() == 3);
  static_assert(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{}() == std::complex<double>{3, 4});
  static_assert(internal::ScalarConstant<Likelihood::maybe, std::integral_constant<int, 7>>{}() == 7);
  static_assert(internal::ScalarConstant{std::integral_constant<int, 7>{}}.value == 7);
  static_assert(internal::ScalarConstant{3}() == 3);
  static_assert(internal::ScalarConstant{3.} == 3.);
  static_assert(std::is_same_v<decltype(internal::ScalarConstant{std::integral_constant<int, 7>{}})::value_type, int>);
  static_assert(std::is_same_v<decltype(internal::ScalarConstant{3})::value_type, int>);
  static_assert(std::is_same_v<decltype(internal::ScalarConstant{3.})::value_type, double>);
  static_assert(internal::ScalarConstant{internal::ScalarConstant<Likelihood::maybe, double, 3>{}}.status == Likelihood::maybe);
  static_assert(internal::ScalarConstant{internal::ScalarConstant<Likelihood::definitely, double, 3>{}}.status == Likelihood::definitely);
  static_assert(internal::ScalarConstant{std::integral_constant<int, 7>{}}.status == Likelihood::definitely);
  static_assert(internal::ScalarConstant{3}.status == Likelihood::definitely);

  static_assert(std::decay_t<decltype(+internal::ScalarConstant<Likelihood::definitely, double, 3>{})>::value == 3);
  static_assert(std::decay_t<decltype(-internal::ScalarConstant<Likelihood::definitely, double, 3>{})>::value == -3);
  static_assert(std::decay_t<decltype(internal::ScalarConstant<Likelihood::definitely, double, 3>{} + std::integral_constant<int, 2>{})>::value == 5);
  static_assert(std::decay_t<decltype(internal::ScalarConstant<Likelihood::definitely, double, 3>{} - std::integral_constant<int, 2>{})>::value == 1);
  static_assert(std::decay_t<decltype(internal::ScalarConstant<Likelihood::definitely, double, 3>{} * std::integral_constant<int, 2>{})>::value == 6);
  static_assert(std::decay_t<decltype(internal::ScalarConstant<Likelihood::definitely, double, 3>{} / std::integral_constant<int, 2>{})>::value == 1.5);

  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value + internal::ScalarConstant<Likelihood::definitely, double, 3>{} == 5);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value - internal::ScalarConstant<Likelihood::definitely, double, 3>{} == -1);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value * internal::ScalarConstant<Likelihood::definitely, double, 3>{} == 6);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 3>{})>::value / internal::ScalarConstant<Likelihood::definitely, double, 2>{} == 1.5);
}


TEST(basics, scalar_traits)
{
  static_assert(complex_number<std::complex<double>>);
  COMPLEXINTEXISTS(static_assert(complex_number<std::complex<int>>));
  static_assert(scalar_type<double>);
  static_assert(scalar_type<int>);
  COMPLEXINTEXISTS(static_assert(scalar_type<std::complex<int>>));
  static_assert(not floating_scalar_type<int>);
  static_assert(floating_scalar_type<float>);
  static_assert(floating_scalar_type<double>);
  static_assert(floating_scalar_type<long double>);
  static_assert(not floating_scalar_type<std::complex<double>>);
  COMPLEXINTEXISTS(static_assert(not floating_scalar_type<std::complex<int>>));
  static_assert(scalar_constant<int, CompileTimeStatus::unknown>);
  static_assert(scalar_constant<double, CompileTimeStatus::unknown>);
  static_assert(scalar_constant<double, CompileTimeStatus::any>);
  static_assert(scalar_constant<std::integral_constant<int, 5>, CompileTimeStatus::known>);
  static_assert(scalar_constant<std::integral_constant<int, 6>, CompileTimeStatus::any>);

  struct return8 { constexpr auto operator()() { return 8; } };
  static_assert(scalar_constant<return8, CompileTimeStatus::known>);
  struct return8r { auto operator()() { return 8; } };
  static_assert(scalar_constant<return8r, CompileTimeStatus::unknown>);

  EXPECT_EQ(get_scalar_constant_value(7), 7);
  EXPECT_EQ(get_scalar_constant_value(std::integral_constant<int, 7>{}), 7);
  EXPECT_EQ(get_scalar_constant_value([](){ return 8; }), 8);
}


TEST(basics, scalar_constant_operation)
{
  static_assert(scalar_constant<internal::scalar_constant_operation<NullaryFunc>, CompileTimeStatus::known>);
  static_assert(get_scalar_constant_value(internal::scalar_constant_operation<NullaryFunc>{}) == 5.5);
  static_assert(internal::scalar_constant_operation<NullaryFunc>::status == Likelihood::definitely);
  static_assert(scalar_constant<internal::scalar_constant_operation<std::negate<>, ConstMaybe>, CompileTimeStatus::known>);
  static_assert(scalar_constant<internal::scalar_constant_operation<std::negate<>, double>, CompileTimeStatus::unknown>);
  static_assert(internal::scalar_constant_operation<std::negate<>, ConstDefinitely>::status == Likelihood::definitely);
  static_assert(internal::scalar_constant_operation<std::negate<>, ConstMaybe>::status == Likelihood::maybe);
  static_assert(internal::scalar_constant_operation<std::negate<>, ConstNoStatus>::status == Likelihood::definitely);
  static_assert(scalar_constant<internal::scalar_constant_operation<std::multiplies<>, double, double>, CompileTimeStatus::unknown>);
  static_assert(internal::scalar_constant_operation{std::plus{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 5>{}}() == 9);
  static_assert(internal::scalar_constant_operation<std::plus<>, ConstDefinitely, ConstDefinitely>::status == Likelihood::definitely);
  static_assert(internal::scalar_constant_operation<std::plus<>, ConstDefinitely, ConstMaybe>::status == Likelihood::maybe);
  static_assert(internal::scalar_constant_operation<std::plus<>, ConstNoStatus, ConstDefinitely>::status == Likelihood::definitely);
  static_assert(internal::scalar_constant_operation<std::plus<>, ConstNoStatus, ConstMaybe>::status == Likelihood::maybe);
  static_assert(internal::scalar_constant_operation<std::plus<>, ConstMaybe, ConstMaybe>::status == Likelihood::maybe);
  static_assert(internal::scalar_constant_operation<std::plus<>, ConstNoStatus, ConstNoStatus>::status == Likelihood::definitely);
  EXPECT_EQ(get_scalar_constant_value(internal::scalar_constant_operation{[](){ return 9; }}), 9);
  int k = 9; EXPECT_EQ(get_scalar_constant_value(internal::scalar_constant_operation{[&k](){ return k; }}), 9);
  EXPECT_EQ(get_scalar_constant_value(internal::scalar_constant_operation{std::plus{}, 4, 5}), 9);

  auto sc3 = internal::scalar_constant_operation{std::minus<>{}, internal::ScalarConstant<Likelihood::definitely, double, 7>{}, std::integral_constant<int, 4>{}};
  static_assert(std::decay_t<decltype(+sc3)>::value == 3);
  static_assert(std::decay_t<decltype(-sc3)>::value == -3);
  static_assert(std::decay_t<decltype(sc3 + std::integral_constant<int, 2>{})>::value == 5);
  static_assert(std::decay_t<decltype(sc3 - std::integral_constant<int, 2>{})>::value == 1);
  static_assert(std::decay_t<decltype(sc3 * std::integral_constant<int, 2>{})>::value == 6);
  static_assert(std::decay_t<decltype(sc3 / std::integral_constant<int, 2>{})>::value == 1.5);

  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value + sc3 == 5);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value - sc3 == -1);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value * sc3 == 6);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 9>{})>::value / sc3 == 3);
}


TEST(basics, scalar_functions)
{
  static_assert(internal::are_within_tolerance(10., 10.));
  static_assert(internal::are_within_tolerance<2>(0., 0. + std::numeric_limits<double>::epsilon()));
  static_assert(not internal::are_within_tolerance<2>(0., 0. + 3 * std::numeric_limits<double>::epsilon()));
  static_assert(internal::update_real_part(std::complex{3.5, 4.5}, 5.5) == std::complex{5.5, 4.5});
  static_assert(internal::update_real_part(std::complex{3, 4}, 5.2) == std::complex{5, 4}); // truncation occurs
}


TEST(basics, constexpr_real_imag_conj)
{
  static_assert(std::is_floating_point_v<decltype(internal::constexpr_real(3))>);
  static_assert(std::is_floating_point_v<decltype(internal::constexpr_imag(3))>);
  static_assert(std::is_floating_point_v<decltype(internal::constexpr_real(internal::constexpr_conj(3)))>);
  static_assert(std::is_floating_point_v<decltype(internal::constexpr_real(std::complex<double>{3., 4.}))>);
  static_assert(internal::constexpr_real(3.) == 3);
  static_assert(internal::constexpr_real(3.f) == 3);
  static_assert(internal::constexpr_real(3.l) == 3);
  static_assert(internal::constexpr_imag(3.) == 0);
  static_assert(internal::constexpr_imag(3.f) == 0);
  static_assert(internal::constexpr_imag(3.l) == 0);
  static_assert(internal::constexpr_conj(3.) == 3.);
  static_assert(internal::constexpr_conj(3.f) == 3.f);
  static_assert(internal::constexpr_conj(3.l) == 3.l);
  EXPECT_EQ(internal::constexpr_real(std::complex<double>{3, 4}), 3);
  EXPECT_EQ(internal::constexpr_imag(std::complex<double>{3, 4}), 4);
  EXPECT_TRUE((internal::constexpr_conj(std::complex<double>{3, 4}) == std::complex<double>{3, -4}));

  static_assert(internal::constexpr_real(std::integral_constant<int, 9>{}) == 9);
  static_assert(internal::constexpr_real(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{}) == 3);
  static_assert(internal::constexpr_imag(std::integral_constant<int, 9>{}) == 0);
  static_assert(internal::constexpr_imag(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{}) == 4);
  static_assert(internal::constexpr_real(internal::constexpr_conj(std::integral_constant<int, 9>{})) == 9);
  static_assert(internal::constexpr_imag(internal::constexpr_conj(std::integral_constant<int, 9>{})) == 0);
  static_assert(internal::constexpr_real(internal::constexpr_conj(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})) == 3);
  static_assert(internal::constexpr_imag(internal::constexpr_conj(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})) == -4);
}


TEST(basics, constexpr_signbit)
{
  static_assert(internal::constexpr_signbit(-3));
  static_assert(not internal::constexpr_signbit(3));
  static_assert(not internal::constexpr_signbit(3.));
  static_assert(internal::constexpr_signbit(-3.));
  static_assert(internal::constexpr_signbit(-3.));
  static_assert(internal::constexpr_signbit(-3.f));
  static_assert(internal::constexpr_signbit(-3.l));
  static_assert(not internal::constexpr_signbit(0.));
#ifdef __cpp_lib_constexpr_cmath
  static_assert(internal::constexpr_signbit(-0.));
  static_assert(not internal::constexpr_signbit(+0.));
  static_assert(internal::constexpr_signbit(-NAN));
  static_assert(not internal::constexpr_signbit(NAN));
#elif defined(__cpp_lib_is_constant_evaluated)
  EXPECT_TRUE(internal::constexpr_signbit(-0.));
  EXPECT_TRUE(not internal::constexpr_signbit(+0.));
  EXPECT_TRUE(internal::constexpr_signbit(-NAN));
  EXPECT_TRUE(not internal::constexpr_signbit(NAN));
#endif
  EXPECT_TRUE(internal::constexpr_signbit(-INFINITY));
  EXPECT_TRUE(not internal::constexpr_signbit(INFINITY));

  static_assert(internal::constexpr_signbit(std::integral_constant<int, -3>{}));
  static_assert(internal::constexpr_signbit(internal::ScalarConstant<Likelihood::definitely, double, -3>{}));
  static_assert(not internal::constexpr_signbit(internal::ScalarConstant<Likelihood::definitely, double, 3>{}));
}


TEST(basics, constexpr_copysign)
{
  static_assert(internal::constexpr_copysign(5, -3) == -5.);
  static_assert(internal::constexpr_copysign(5, 3) == 5.);
  static_assert(internal::constexpr_copysign(5U, -3) == -5.);
  static_assert(std::is_floating_point_v<decltype(internal::constexpr_copysign(5U, -3))>);
  static_assert(internal::constexpr_copysign(3., -5.) == -3.);
  static_assert(internal::constexpr_copysign(-3., 5.) == 3.);
  static_assert(internal::constexpr_copysign(-3.f, 5.f) == 3.f);
  static_assert(internal::constexpr_copysign(3.l, -5.l) == -3.l);
  static_assert(internal::constexpr_copysign(3.f, -INFINITY) == -3.f);
  static_assert(internal::constexpr_copysign(3.f, INFINITY) == 3.f);
  static_assert(internal::constexpr_copysign(INFINITY, -3.f) == -INFINITY);
  static_assert(internal::constexpr_copysign(-INFINITY, 3.f) == INFINITY);
#if defined(__cpp_lib_constexpr_cmath) or defined(__cpp_lib_is_constant_evaluated)
  EXPECT_TRUE(std::signbit(internal::constexpr_copysign(0., -1.)));
  EXPECT_FALSE(std::signbit(internal::constexpr_copysign(0., 1.)));
  EXPECT_TRUE(std::signbit(internal::constexpr_copysign(-0., -1.)));
  EXPECT_FALSE(std::signbit(internal::constexpr_copysign(-0., 1.)));
#endif

  static_assert(internal::constexpr_copysign(internal::ScalarConstant<Likelihood::definitely, int, 5>{}, internal::ScalarConstant<Likelihood::definitely, int, -3>{}) == -5);
  static_assert(internal::constexpr_copysign(internal::ScalarConstant<Likelihood::definitely, int, -5>{}, internal::ScalarConstant<Likelihood::definitely, int, 3>{}) == 5);
  static_assert(internal::constexpr_copysign(internal::ScalarConstant<Likelihood::definitely, double, 5>{}, internal::ScalarConstant<Likelihood::definitely, double, -3>{}) == -5);
  static_assert(internal::constexpr_copysign(internal::ScalarConstant<Likelihood::definitely, double, -5>{}, internal::ScalarConstant<Likelihood::definitely, double, 3>{}) == 5);
}


TEST(basics, constexpr_sqrt)
{
  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::isnan(internal::constexpr_sqrt(-1)));
    EXPECT_TRUE(std::isnan(internal::constexpr_sqrt(internal::constexpr_NaN<double>())));
    EXPECT_EQ(internal::constexpr_sqrt(internal::constexpr_infinity<double>()), internal::constexpr_infinity<double>());
    EXPECT_TRUE(std::signbit(internal::constexpr_sqrt(-0.)));
    EXPECT_FALSE(std::signbit(internal::constexpr_sqrt(+0.)));
    EXPECT_TRUE(internal::constexpr_sqrt(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_sqrt(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_sqrt(0) == 0);
  static_assert(internal::constexpr_sqrt(1) == 1);
  static_assert(internal::constexpr_sqrt(4) == 2);
  static_assert(internal::constexpr_sqrt(9) == 3);
  static_assert(internal::constexpr_sqrt(1000000) == 1000);
  static_assert(internal::are_within_tolerance(internal::constexpr_sqrt(2.), numbers::sqrt2));
  static_assert(internal::are_within_tolerance(internal::constexpr_sqrt(3.), numbers::sqrt3));
  static_assert(internal::are_within_tolerance(internal::constexpr_sqrt(4.0e6), 2.0e3));
  static_assert(internal::are_within_tolerance(internal::constexpr_sqrt(9.0e-2), 3.0e-1));
  static_assert(internal::are_within_tolerance(internal::constexpr_sqrt(2.5e-11), 5.0e-6));
  EXPECT_NEAR(internal::constexpr_sqrt(5.), std::sqrt(5), 1e-9);
  EXPECT_NEAR(internal::constexpr_sqrt(1.0e20), std::sqrt(1.0e20), 1e-9);
  EXPECT_NEAR(internal::constexpr_sqrt(0.001), std::sqrt(0.001), 1e-9);
  EXPECT_NEAR(internal::constexpr_sqrt(1e-20), std::sqrt(1e-20), 1e-9);

  EXPECT_PRED3(tolerance, internal::constexpr_sqrt(std::complex<double>{-4}), std::sqrt(std::complex<double>{-4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_sqrt(std::complex<double>{3, 4}), std::sqrt(std::complex<double>{3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_sqrt(std::complex<double>{3, -4}), (std::complex<double>{2, -1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_sqrt(std::complex<double>{3, 4}), (std::complex<double>{2, 1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_sqrt(std::complex<double>{3, -4}), std::sqrt(std::complex<double>{3, -4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_sqrt(std::complex<double>{-3, 4}), std::sqrt(std::complex<double>{-3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_sqrt(std::complex<double>{-3, -4}), std::sqrt(std::complex<double>{-3, -4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_sqrt(std::complex<double>{-3e10, 4e10}), std::sqrt(std::complex<double>{-3e10, 4e10}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_sqrt(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_sqrt(std::complex<int>{3, 4})));

  static_assert(internal::constexpr_sqrt(std::integral_constant<int, 9>{}) == 3);
  static_assert(internal::are_within_tolerance(internal::constexpr_sqrt(internal::ScalarConstant<Likelihood::definitely, double, 9>{}), 3, 1e-6));
}


TEST(basics, constexpr_abs)
{
  static_assert(std::is_integral_v<decltype(internal::constexpr_abs(-3))>);
  static_assert(std::is_floating_point_v<decltype(internal::constexpr_abs(3.))>);
  static_assert(std::is_floating_point_v<decltype(internal::constexpr_abs(std::complex<double>{3., 4.}))>);
  static_assert(internal::constexpr_abs(3) == 3);
  static_assert(internal::constexpr_abs(-3) == 3);
  static_assert(internal::constexpr_abs(3.) == 3);
  static_assert(internal::constexpr_abs(-3.) == 3);
  static_assert(internal::constexpr_abs(-3.f) == 3);
  static_assert(internal::constexpr_abs(-3.l) == 3);
  static_assert(internal::constexpr_abs(INFINITY) == INFINITY);
  static_assert(internal::constexpr_abs(-INFINITY) == INFINITY);
  EXPECT_TRUE(std::isnan(internal::constexpr_abs(-NAN)));
  EXPECT_FALSE(std::signbit(internal::constexpr_abs(NAN)));
#if defined(__cpp_lib_constexpr_cmath) or defined(__cpp_lib_is_constant_evaluated)
  EXPECT_FALSE(std::signbit(internal::constexpr_abs(-0.)));
  EXPECT_FALSE(std::signbit(internal::constexpr_abs(-NAN)));
#endif
  EXPECT_EQ(internal::constexpr_abs(std::complex<double>{3, -4}), 5);
  EXPECT_EQ(internal::constexpr_abs(std::complex<double>{-3, 4}), 5);

  static_assert(internal::constexpr_abs(std::integral_constant<int, -9>{}) == 9);
  static_assert(internal::are_within_tolerance(internal::constexpr_abs(internal::ScalarConstant<Likelihood::definitely, double, -9>{}), 9, 1e-6));
  static_assert(internal::constexpr_abs(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{}) == 5);
}


TEST(basics, constexpr_exp)
{
  constexpr auto e = numbers::e_v<double>;
  constexpr auto eL = numbers::e_v<long double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::isnan(internal::constexpr_exp(internal::constexpr_NaN<double>())));
    EXPECT_TRUE(internal::constexpr_exp(-internal::constexpr_infinity<double>()) == 0);
    EXPECT_TRUE(internal::constexpr_exp(internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_exp(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_exp(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::are_within_tolerance<10>(internal::constexpr_exp(0), 1));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_exp(1), e));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_exp(2), e*e));
  static_assert(internal::are_within_tolerance<100>(internal::constexpr_exp(3), e*e*e));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_exp(-1), 1/e));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_exp(-2), 1/(e*e)));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_exp(1.0), e));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_exp(2.0), e*e));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_exp(1.0L), eL));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_exp(2.0L), eL*eL));
  static_assert(internal::are_within_tolerance<100>(internal::constexpr_exp(3.0L), eL*eL*eL));
  EXPECT_NEAR(internal::constexpr_exp(5), std::exp(5), 1e-9);
  EXPECT_NEAR(internal::constexpr_exp(-10), std::exp(-10), 1e-9);
  EXPECT_NEAR(internal::constexpr_exp(50), std::exp(50), 1e8);
  EXPECT_NEAR(internal::constexpr_exp(50.5), std::exp(50.5), 1e8);
  EXPECT_NEAR(internal::constexpr_exp(300), std::exp(300), 1e120);
  EXPECT_NEAR(internal::constexpr_exp(300.7), std::exp(300.7), 1e120);
  EXPECT_NEAR(internal::constexpr_exp(1e-5), std::exp(1e-5), 1e-12);
  EXPECT_NEAR(internal::constexpr_exp(1e-10), std::exp(1e-10), 1e-16);

  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_exp(std::complex<double>{2, 0})), e*e, 1e-6));
  EXPECT_PRED3(tolerance, internal::constexpr_exp(std::complex<double>{3.3, -4.3}), std::exp(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_exp(std::complex<double>{10.4, 3.4}), std::exp(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_exp(std::complex<double>{-30.6, 20.6}), std::exp(std::complex<double>{-30.6, 20.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_exp(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_exp(std::integral_constant<int, 2>{}), e*e, 1e-6));
  static_assert(internal::are_within_tolerance(internal::constexpr_exp(internal::ScalarConstant<Likelihood::definitely, double, -2>{}), 1/(e*e), 1e-6));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_exp(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 2, 0>{})), e*e, 1e-6));
}


TEST(basics, constexpr_expm1)
{
  constexpr auto e = numbers::e_v<double>;
  constexpr auto eL = numbers::e_v<long double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_expm1(internal::constexpr_NaN<double>()) != internal::constexpr_expm1(internal::constexpr_NaN<double>()));
    EXPECT_TRUE(internal::constexpr_expm1(internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_expm1(-internal::constexpr_infinity<double>()) == -1);
    EXPECT_TRUE(std::signbit(internal::constexpr_expm1(-0.)));
    EXPECT_FALSE(std::signbit(internal::constexpr_expm1(+0.)));
    EXPECT_TRUE(internal::constexpr_expm1(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_expm1(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_expm1(0) == 0);
  static_assert(internal::are_within_tolerance(internal::constexpr_expm1(1), e - 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_expm1(2), e*e - 1, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_expm1(3), e*e*e - 1, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_expm1(-1), 1 / e - 1, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_expm1(3.L), eL*eL*eL - 1, 1e-9));
  static_assert(std::real(internal::constexpr_expm1(std::complex<double>{3e-12, 0})) == internal::constexpr_expm1(3e-12));
  EXPECT_PRED3(tolerance, internal::constexpr_expm1(1e-4), std::expm1(1e-4), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_expm1(1e-8), std::expm1(1e-8), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_expm1(1e-32), std::expm1(1e-32), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_expm1(5.2), std::expm1(5.2), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_expm1(10.2), std::expm1(10.2), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_expm1(-10.2), std::expm1(-10.2), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_expm1(3e-12), std::expm1(3e-12), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_expm1(-3e-12), std::expm1(-3e-12), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_expm1(std::complex<double>{2, 0})), e*e - 1, 1e-6));
  EXPECT_PRED3(tolerance, internal::constexpr_expm1(std::complex<double>{0.001, -0.001}), std::exp(std::complex<double>{0.001, -0.001}) - 1.0, 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_expm1(std::complex<double>{3.2, -4.2}), std::exp(std::complex<double>{3.2, -4.2}) - 1.0, 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_expm1(std::complex<double>{10.3, 3.3}), std::exp(std::complex<double>{10.3, 3.3}) - 1.0, 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_expm1(std::complex<double>{-10.4, 10.4}), std::exp(std::complex<double>{-10.4, 10.4}) - 1.0, 1e-9);
  EXPECT_PRED3(tolerance, std::real(internal::constexpr_expm1(std::complex<double>{3e-12, 0})), std::expm1(3e-12), 1e-20);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_expm1(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_expm1(std::integral_constant<int, 2>{}), e*e - 1, 1e-6));
  static_assert(internal::are_within_tolerance(internal::constexpr_expm1(internal::ScalarConstant<Likelihood::definitely, double, -2>{}), 1/(e*e) - 1, 1e-6));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_expm1(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 2, 0>{})), e*e - 1, 1e-6));
}


TEST(basics, constexpr_sinh)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_sinh(internal::constexpr_NaN<double>()) != internal::constexpr_sinh(internal::constexpr_NaN<double>()));
    EXPECT_TRUE(internal::constexpr_sinh(internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_sinh(-internal::constexpr_infinity<double>()) == -internal::constexpr_infinity<double>());
    EXPECT_TRUE(std::signbit(internal::constexpr_sinh(-0.)));
    EXPECT_TRUE(not std::signbit(internal::constexpr_sinh(0.)));
    EXPECT_TRUE(internal::constexpr_sinh(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_sinh(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_sinh(0) == 0);
  static_assert(internal::are_within_tolerance(internal::constexpr_sinh(1), (e - 1/e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sinh(2), (e*e - 1/e/e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sinh(3), (e*e*e - 1/e/e/e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sinh(-1), (1/e - e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sinh(-2), (1/e/e - e*e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sinh(-3), (1/e/e/e - e*e*e)/2, 1e-9));
  EXPECT_NEAR(internal::constexpr_sinh(5), std::sinh(5), 1e-9);
  EXPECT_NEAR(internal::constexpr_sinh(-10), std::sinh(-10), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_sinh(std::complex<double>{2, 0})), (e*e - 1/e/e)/2, 1e-9));
  EXPECT_PRED3(tolerance, internal::constexpr_sinh(std::complex<double>{3.3, -4.3}), std::sinh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_sinh(std::complex<double>{10.4, 3.4}), std::sinh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_sinh(std::complex<double>{-10.6, 10.6}), std::sinh(std::complex<double>{-10.6, 10.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_sinh(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_sinh(std::integral_constant<int, 2>{}), (e*e - 1/e/e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sinh(internal::ScalarConstant<Likelihood::definitely, double, -2>{}), (1/e/e - e*e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_sinh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 2, 0>{})), (e*e - 1/e/e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_sinh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, -4>{})), -6.548120040911001647767, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_sinh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, -4>{})), 7.619231720321410208487, 1e-9));
}


TEST(basics, constexpr_cosh)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_cosh(internal::constexpr_NaN<double>()) != internal::constexpr_cosh(internal::constexpr_NaN<double>()));
    EXPECT_TRUE(internal::constexpr_cosh(internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_cosh(-internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_cosh(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_cosh(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_cosh(0) == 1);
  static_assert(internal::are_within_tolerance(internal::constexpr_cosh(1), (e + 1/e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_cosh(2), (e*e + 1/e/e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_cosh(3), (e*e*e + 1/e/e/e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_cosh(-1), (1/e + e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_cosh(-2), (1/e/e + e*e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_cosh(-3), (1/e/e/e + e*e*e)/2, 1e-9));
  EXPECT_NEAR(internal::constexpr_cosh(5), std::cosh(5), 1e-9);
  EXPECT_NEAR(internal::constexpr_cosh(-10), std::cosh(-10), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_cosh(std::complex<double>{2, 0})), (e*e + 1/e/e)/2, 1e-9));
  EXPECT_PRED3(tolerance, internal::constexpr_cosh(std::complex<double>{3.3, -4.3}), std::cosh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_cosh(std::complex<double>{10.4, 3.4}), std::cosh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_cosh(std::complex<double>{-10.6, 10.6}), std::cosh(std::complex<double>{-10.6, 10.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_cosh(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_cosh(std::integral_constant<int, 2>{}), (e*e + 1/e/e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_cosh(internal::ScalarConstant<Likelihood::definitely, double, -2>{}), (1/e/e + e*e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_cosh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 2, 0>{})), (e*e + 1/e/e)/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_cosh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, -4>{})), -6.580663040551156432561, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_cosh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, -4>{})), 7.581552742746544353716, 1e-9));
}


TEST(basics, constexpr_tanh)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_tanh(internal::constexpr_NaN<double>()) != internal::constexpr_tanh(internal::constexpr_NaN<double>()));
    EXPECT_TRUE(internal::constexpr_tanh(internal::constexpr_infinity<double>()) == 1);
    EXPECT_TRUE(internal::constexpr_tanh(-internal::constexpr_infinity<double>()) == -1);
    EXPECT_TRUE(std::signbit(internal::constexpr_tanh(-0.)));
    EXPECT_TRUE(not std::signbit(internal::constexpr_tanh(0.)));
    EXPECT_TRUE(internal::constexpr_tanh(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_tanh(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_tanh(0) == 0);
  static_assert(internal::are_within_tolerance(internal::constexpr_tanh(1), (e*e - 1)/(e*e + 1), 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tanh(2), (e*e*e*e - 1)/(e*e*e*e + 1), 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tanh(3), (e*e*e*e*e*e - 1)/(e*e*e*e*e*e + 1), 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tanh(-1), (1 - e*e)/(1 + e*e), 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tanh(-2), (1 - e*e*e*e)/(1 + e*e*e*e), 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tanh(-3), (1 - e*e*e*e*e*e)/(1 + e*e*e*e*e*e), 1e-9));
  EXPECT_NEAR(internal::constexpr_tanh(5), std::tanh(5), 1e-9);
  EXPECT_NEAR(internal::constexpr_tanh(-10), std::tanh(-10), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_tanh(std::complex<double>{2, 0})), (e*e*e*e - 1)/(e*e*e*e + 1), 1e-9));
  EXPECT_PRED3(tolerance, internal::constexpr_tanh(std::complex<double>{3.3, -4.3}), std::tanh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_tanh(std::complex<double>{10.4, 3.4}), std::tanh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_tanh(std::complex<double>{-30.6, 20.6}), std::tanh(std::complex<double>{-30.6, 20.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_tanh(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_tanh(std::integral_constant<int, 2>{}), (e*e*e*e - 1)/(e*e*e*e + 1), 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tanh(internal::ScalarConstant<Likelihood::definitely, double, -2>{}), (1 - e*e*e*e)/(1 + e*e*e*e), 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_tanh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 1.00070953606723293933, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_tanh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 0.004908258067496060259079, 1e-9));
}


TEST(basics, constexpr_sin)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::isnan(internal::constexpr_sin(internal::constexpr_NaN<double>())));
    EXPECT_TRUE(std::isnan(internal::constexpr_sin(internal::constexpr_infinity<double>())));
    EXPECT_TRUE(std::isnan(internal::constexpr_sin(-internal::constexpr_infinity<double>())));
    EXPECT_FALSE(std::signbit(internal::constexpr_sin(+0.)));
    EXPECT_TRUE(std::signbit(internal::constexpr_sin(-0.)));
    EXPECT_TRUE(internal::constexpr_sin(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_sin(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::are_within_tolerance(internal::constexpr_sin(0), 0));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(2*pi), 0));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(-2*pi), 0));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(pi), 0));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(-pi), 0));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(32*pi), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(-32*pi), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(0x1p16*pi), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(-0x1p16*pi), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(0x1p16L*piL), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(-0x1p16L*piL), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(0x1p16F*piF), 0, 1e-2));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(-0x1p16F*piF), 0, 1e-2));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(0x1p100L*piL), 0, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(-0x1p100L*piL), 0, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(0x1p180L*piL), 0, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(-0x1p180L*piL), 0, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(0x1p250L*piL), 0, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(-0x1p250L*piL), 0, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(pi/2), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(-pi/2), -1));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(pi/4), numbers::sqrt2_v<double>/2));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(piL/4), numbers::sqrt2_v<long double>/2));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(piF/4), numbers::sqrt2_v<float>/2));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(pi/4 + 32*pi), numbers::sqrt2_v<double>/2, 1e-9));
  EXPECT_NEAR(internal::constexpr_sin(2), std::sin(2), 1e-9);
  EXPECT_NEAR(internal::constexpr_sin(-32), std::sin(-32), 1e-9);
  EXPECT_NEAR(internal::constexpr_sin(0x1p16), std::sin(0x1p16), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_sin(std::complex<double>{pi/2, 0}), 1));
  EXPECT_PRED3(tolerance, internal::constexpr_sin(std::complex<double>{4.1, 3.1}), std::sin(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_sin(std::complex<double>{3.2, -4.2}), std::sin(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_sin(std::complex<double>{-3.3, 4.3}), std::sin(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_sin(std::complex<double>{-9.3, 10.3}), std::sin(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_sin(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_sin(std::integral_constant<int, 2>{}), 0.909297426825681695396, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_sin(internal::ScalarConstant<Likelihood::definitely, double, 2>{}), 0.909297426825681695396, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_sin(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 2, 0>{})), 0.909297426825681695396, 1e-9));
}


TEST(basics, constexpr_cos)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::isnan(internal::constexpr_cos(internal::constexpr_NaN<double>())));
    EXPECT_TRUE(std::isnan(internal::constexpr_cos(internal::constexpr_infinity<double>())));
    EXPECT_TRUE(std::isnan(internal::constexpr_cos(-internal::constexpr_infinity<double>())));
    EXPECT_TRUE(internal::constexpr_cos(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_cos(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_cos(2*pi) == 1);
  static_assert(internal::constexpr_cos(-2*pi) == 1);
  static_assert(internal::constexpr_cos(0) == 1);
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(pi), -1));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(-pi), -1));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(32*pi), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(-32*pi), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(0x1p16*pi), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(-0x1p16*pi), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(0x1p16L*piL), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(-0x1p16L*piL), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(0x1p16F*piF), 1, 1e-4));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(-0x1p16F*piF), 1, 1e-4));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(0x1p100L*piL), 1, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(-0x1p100L*piL), 1, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(0x1p180L*piL), 1, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(-0x1p180L*piL), 1, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(0x1p250L*piL), 1, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(-0x1p250L*piL), 1, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(pi/2), 0));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(-pi/2), 0));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(pi/4), numbers::sqrt2_v<double>/2));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(piL/4), numbers::sqrt2_v<long double>/2));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(piF/4), numbers::sqrt2_v<float>/2));
  EXPECT_NEAR(internal::constexpr_cos(2), std::cos(2), 1e-9);
  EXPECT_NEAR(internal::constexpr_cos(-32), std::cos(-32), 1e-9);
  EXPECT_NEAR(internal::constexpr_cos(0x1p16), std::cos(0x1p16), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_cos(std::complex<double>{pi/2, 0}), 0));
  EXPECT_PRED3(tolerance, internal::constexpr_cos(std::complex<double>{4.1, 3.1}), std::cos(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_cos(std::complex<double>{3.2, -4.2}), std::cos(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_cos(std::complex<double>{-3.3, 4.3}), std::cos(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_cos(std::complex<double>{-9.3, 10.3}), std::cos(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_cos(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_cos(std::integral_constant<int, 2>{}), -0.4161468365471423869976, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_cos(internal::ScalarConstant<Likelihood::definitely, double, 2>{}), -0.4161468365471423869976, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_cos(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 2, 0>{})), -0.4161468365471423869976, 1e-9));
}


TEST(basics, constexpr_tan)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_tan(internal::constexpr_NaN<double>()) != internal::constexpr_tan(internal::constexpr_NaN<double>()));
    EXPECT_TRUE(internal::constexpr_tan(internal::constexpr_infinity<double>()) != internal::constexpr_tan(internal::constexpr_infinity<double>()));
    EXPECT_TRUE(internal::constexpr_tan(-internal::constexpr_infinity<double>()) != internal::constexpr_tan(internal::constexpr_infinity<double>()));
    EXPECT_TRUE(internal::constexpr_tan(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_tan(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_tan(0) == 0);
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(2*pi), 0));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(-2*pi), 0));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(pi), 0));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(-pi), 0));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(32*pi), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(-32*pi), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(0x1p16*pi), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(-0x1p16*pi), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(0x1p16L*piL), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(-0x1p16L*piL), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(0x1p16F*piF), 0, 1e-2));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(-0x1p16F*piF), 0, 1e-2));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(0x1p100L*piL), 0, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(-0x1p100L*piL), 0, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(0x1p180L*piL), 0, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(-0x1p180L*piL), 0, 1.));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(0x1p250L*piL), 0, 2.));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(-0x1p250L*piL), 0, 2.));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(pi/4), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(piL/4), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(piF/4), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(pi/4 + 32*pi), 1, 1e-9));
  EXPECT_NEAR(internal::constexpr_tan(2), std::tan(2), 1e-9);
  EXPECT_NEAR(internal::constexpr_tan(-32), std::tan(-32), 1e-9);
  EXPECT_NEAR(internal::constexpr_tan(0x1p16), std::tan(0x1p16), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_tan(std::complex<double>{pi/4, 0}), 1));
  EXPECT_PRED3(tolerance, internal::constexpr_tan(std::complex<double>{4.1, 3.1}), std::tan(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_tan(std::complex<double>{3.2, -4.2}), std::tan(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_tan(std::complex<double>{-3.3, 4.3}), std::tan(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_tan(std::complex<double>{-30.3, 40.3}), std::tan(std::complex<double>{-30.3, 40.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_tan(std::complex<int>{30, -2})));

  static_assert(internal::are_within_tolerance(internal::constexpr_tan(std::integral_constant<int, 2>{}), -2.185039863261518991643, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_tan(internal::ScalarConstant<Likelihood::definitely, double, 2>{}), -2.185039863261518991643, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_tan(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), -1.873462046294784262243E-4, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_tan(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 0.9993559873814731413917, 1e-9));
}


TEST(basics, constexpr_log)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_log(0) == -internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_log(-0) == -internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_log(+internal::constexpr_infinity<double>()) == +internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_log(-1) != internal::constexpr_log(-1)); // Nan
    EXPECT_FALSE(std::signbit(internal::constexpr_log(1)));
    EXPECT_TRUE(internal::constexpr_log(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_log(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_log(1) == 0);
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_log(2), numbers::ln2_v<double>));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_log(10), numbers::ln10_v<double>));
  static_assert(internal::are_within_tolerance(internal::constexpr_log(e), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_log(e*e), 2));
  static_assert(internal::are_within_tolerance(internal::constexpr_log(e*e*e), 3));
  static_assert(internal::are_within_tolerance(internal::constexpr_log(e*e*e*e*e*e*e*e*e*e*e*e*e*e*e*e), 16));
  static_assert(internal::are_within_tolerance(internal::constexpr_log(1 / e), -1));
  EXPECT_NEAR(internal::constexpr_log(5.0L), std::log(5.0L), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(0.2L), std::log(0.2L), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(5), std::log(5), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(0.2), std::log(0.2), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(20), std::log(20), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(0.05), std::log(0.05), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(100), std::log(100), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(0.01), std::log(0.01), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(1e20), std::log(1e20), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(1e-20), std::log(1e-20), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(1e200), std::log(1e200), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(1e-200), std::log(1e-200), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(1e200L), std::log(1e200L), 1e-9);
  EXPECT_NEAR(internal::constexpr_log(1e-200L), std::log(1e-200L), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_log(std::complex<double>{e*e, 0}), 2));
  EXPECT_PRED3(tolerance, internal::constexpr_log(std::complex<double>{-4}), std::log(std::complex<double>{-4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_log(std::complex<double>{3, 4}), std::log(std::complex<double>{3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_log(std::complex<double>{3, -4}), std::log(std::complex<double>{3, -4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_log(std::complex<double>{-3, 4}), std::log(std::complex<double>{-3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_log(std::complex<double>{-3, -4}), std::log(std::complex<double>{-3, -4}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_log(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_log(std::integral_constant<int, 2>{}), numbers::ln2_v<double>, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_log(internal::ScalarConstant<Likelihood::definitely, double, 2>{}), numbers::ln2_v<double>, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_log(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 1.609437912434100374601, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_log(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 0.9272952180016122324285, 1e-9));
}


TEST(basics, constexpr_log1p)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::signbit(internal::constexpr_log1p(-0.)));
    EXPECT_FALSE(std::signbit(internal::constexpr_log1p(+0.)));
    EXPECT_EQ(internal::constexpr_log1p(-1), -internal::constexpr_infinity<double>());
    EXPECT_EQ(internal::constexpr_log1p(+internal::constexpr_infinity<double>()), +internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_log1p(-2) != internal::constexpr_log(-2)); // Nan
    EXPECT_TRUE(internal::constexpr_log1p(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_log1p(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::are_within_tolerance<10>(internal::constexpr_log1p(-0.), 0));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_log1p(1.), numbers::ln2_v<double>));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_log1p(9.), numbers::ln10_v<double>));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_log1p(e - 1), 1));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_log1p(e*e - 1), 2));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_log1p(e*e*e - 1), 3));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_log1p(e*e*e*e*e*e*e*e*e*e*e*e*e*e*e*e - 1), 16));
  static_assert(internal::are_within_tolerance<10>(internal::constexpr_log1p(1/e - 1), -1));
  EXPECT_NEAR(internal::constexpr_log1p(5.0L), std::log1p(5.0L), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(0.2L), std::log1p(0.2L), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(5), std::log1p(5), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(0.2), std::log1p(0.2), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(20), std::log1p(20), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(0.05), std::log1p(0.05), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(100), std::log1p(100), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(0.01), std::log1p(0.01), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(0.001), std::log1p(0.001), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(0.0001), std::log1p(0.0001), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(0.00001), std::log1p(0.00001), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(0.000001), std::log1p(0.000001), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(1e-20), std::log1p(1e-20), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(1e-200), std::log1p(1e-200), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(1e-200L), std::log1p(1e-200L), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(1e20), std::log1p(1e20), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(1e200), std::log1p(1e200), 1e-9);
  EXPECT_NEAR(internal::constexpr_log1p(1e200L), std::log1p(1e200L), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_log1p(std::complex<double>{e*e - 1, 0}), 2));
  EXPECT_PRED3(tolerance, std::real(internal::constexpr_log1p(std::complex<double>{4e-21})), std::log1p(4e-21), 1e-30);
  EXPECT_PRED3(tolerance, internal::constexpr_log1p(std::complex<double>{-4}), std::log(std::complex<double>{-3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_log1p(std::complex<double>{3, 4}), std::log(std::complex<double>{4, 4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_log1p(std::complex<double>{3, -4}), std::log(std::complex<double>{4, -4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_log1p(std::complex<double>{-3, 4}), std::log(std::complex<double>{-2, 4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_log1p(std::complex<double>{-3, -4}), std::log(std::complex<double>{-2, -4}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_log1p(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_log1p(std::integral_constant<int, 1>{}), numbers::ln2_v<double>, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_log1p(internal::ScalarConstant<Likelihood::definitely, double, 1>{}), numbers::ln2_v<double>, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_log1p(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 2, 4>{})), 1.609437912434100374601, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_log1p(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 2, 4>{})), 0.9272952180016122324285, 1e-9));
}


TEST(basics, constexpr_asinh)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_asinh(internal::constexpr_NaN<double>()) != internal::constexpr_asinh(internal::constexpr_NaN<double>()));
    EXPECT_TRUE(internal::constexpr_asinh(internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_asinh(-internal::constexpr_infinity<double>()) == -internal::constexpr_infinity<double>());
    EXPECT_FALSE(std::signbit(internal::constexpr_asinh(+0.)));
    EXPECT_TRUE(std::signbit(internal::constexpr_asinh(-0.)));
    EXPECT_TRUE(internal::constexpr_asinh(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_asinh(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_asinh(0) == 0);
  static_assert(internal::are_within_tolerance(internal::constexpr_asinh((e - 1/e)/2), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_asinh((e*e - 1/e/e)/2), 2));
  static_assert(internal::are_within_tolerance(internal::constexpr_asinh((e*e*e - 1/e/e/e)/2), 3, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_asinh((1/e - e)/2), -1));
  static_assert(internal::are_within_tolerance(internal::constexpr_asinh((1/e/e - e*e)/2), -2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_asinh((1/e/e/e - e*e*e)/2), -3, 1e-9));
  EXPECT_NEAR(internal::constexpr_asinh(5), std::asinh(5), 1e-9);
  EXPECT_NEAR(internal::constexpr_asinh(-10), std::asinh(-10), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_asinh(std::complex<double>{(e*e - 1/e/e)/2, 0}), 2));
  EXPECT_PRED3(tolerance, internal::constexpr_asinh(std::complex<double>{3.3, -4.3}), std::asinh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_asinh(std::complex<double>{10.4, 3.4}), std::asinh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_asinh(std::complex<double>{-10.6, 10.6}), std::asinh(std::complex<double>{-10.6, 10.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_asinh(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_asinh(std::integral_constant<int, 2>{}), 1.443635475178810342493, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_asinh(internal::ScalarConstant<Likelihood::definitely, double, 2>{}), 1.443635475178810342493, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_asinh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 2.299914040879269649956, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_asinh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 0.9176168533514786557599, 1e-9));
}


TEST(basics, constexpr_acosh)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_acosh(internal::constexpr_NaN<double>()) != internal::constexpr_acosh(internal::constexpr_NaN<double>()));
    EXPECT_TRUE(internal::constexpr_acosh(-1) != internal::constexpr_acosh(-1));
    EXPECT_TRUE(internal::constexpr_acosh(internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_acosh(-internal::constexpr_infinity<double>()) != internal::constexpr_acosh(-internal::constexpr_infinity<double>()));
    EXPECT_TRUE(internal::constexpr_acosh(0.9) != internal::constexpr_acosh(0.9));
    EXPECT_TRUE(internal::constexpr_acosh(-1) != internal::constexpr_acosh(-1));
    EXPECT_FALSE(std::signbit(internal::constexpr_acosh(1)));
    EXPECT_TRUE(internal::constexpr_acosh(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_acosh(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_acosh(1) == 0);
  static_assert(internal::are_within_tolerance(internal::constexpr_acosh((e + 1/e)/2), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_acosh((e*e + 1/e/e)/2), 2));
  static_assert(internal::are_within_tolerance(internal::constexpr_acosh((e*e*e + 1/e/e/e)/2), 3, 1e-9));
  EXPECT_NEAR(internal::constexpr_acosh(5), std::acosh(5), 1e-9);
  EXPECT_NEAR(internal::constexpr_acosh(10), std::acosh(10), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_acosh(std::complex<double>{(e*e + 1/e/e)/2, 0}), 2));
  EXPECT_PRED3(tolerance, internal::constexpr_acosh(std::complex<double>{-2, 0}), std::acosh(std::complex<double>{-2, 0}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_acosh(std::complex<double>{3.3, -4.3}), std::acosh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_acosh(std::complex<double>{5.4, 3.4}), std::acosh(std::complex<double>{5.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_acosh(std::complex<double>{-5.6, 5.6}), std::acosh(std::complex<double>{-5.6, 5.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_acosh(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_acosh(std::integral_constant<int, 2>{}), 1.316957896924816708625, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_acosh(internal::ScalarConstant<Likelihood::definitely, double, 2>{}), 1.316957896924816708625, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_acosh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 2.305509031243476942042, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_acosh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 0.9368124611557199029125, 1e-9));
}


TEST(basics, constexpr_atanh)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_atanh(internal::constexpr_NaN<double>()) != internal::constexpr_atanh(internal::constexpr_NaN<double>()));
    EXPECT_TRUE(internal::constexpr_atanh(2) != internal::constexpr_atanh(2));
    EXPECT_TRUE(internal::constexpr_atanh(-2) != internal::constexpr_atanh(-2));
    EXPECT_TRUE(internal::constexpr_atanh(1) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_atanh(-1) == -internal::constexpr_infinity<double>());
    EXPECT_FALSE(std::signbit(internal::constexpr_atanh(+0.)));
    EXPECT_TRUE(std::signbit(internal::constexpr_atanh(-0.)));
    EXPECT_TRUE(internal::constexpr_atanh(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_atanh(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_atanh(0) == 0);
  static_assert(internal::are_within_tolerance(internal::constexpr_atanh((e*e - 1)/(e*e + 1)), 1));
  static_assert(internal::are_within_tolerance(internal::constexpr_atanh((e*e*e*e - 1)/(e*e*e*e + 1)), 2));
  static_assert(internal::are_within_tolerance(internal::constexpr_atanh((e*e*e*e*e*e - 1)/(e*e*e*e*e*e + 1)), 3, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_atanh((1 - e*e)/(1 + e*e)), -1));
  static_assert(internal::are_within_tolerance(internal::constexpr_atanh((1 - e*e*e*e)/(1 + e*e*e*e)), -2));
  static_assert(internal::are_within_tolerance(internal::constexpr_atanh((1 - e*e*e*e*e*e)/(1 + e*e*e*e*e*e)), -3, 1e-9));
  EXPECT_NEAR(internal::constexpr_atanh(0.99), std::atanh(0.99), 1e-9);
  EXPECT_NEAR(internal::constexpr_atanh(-0.99), std::atanh(-0.99), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_atanh(std::complex<double>{(e*e*e*e - 1)/(e*e*e*e + 1), 0}), 2));
  EXPECT_PRED3(tolerance, internal::constexpr_atanh(std::complex<double>{3.3, -4.3}), std::atanh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atanh(std::complex<double>{10.4, 3.4}), std::atanh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atanh(std::complex<double>{-30.6, 20.6}), std::atanh(std::complex<double>{-30.6, 20.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_atanh(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_atanh(std::integral_constant<int, 0>{}), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_atanh(internal::ScalarConstant<Likelihood::definitely, double, 0>{}), 0, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_atanh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 0.1175009073114338884127, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_atanh(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 1.409921049596575522531, 1e-9));
}


TEST(basics, constexpr_asin)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_asin(internal::constexpr_NaN<double>()) != internal::constexpr_asin(internal::constexpr_NaN<double>()));
    EXPECT_TRUE(internal::constexpr_asin(2.0) != internal::constexpr_asin(2.0));
    EXPECT_TRUE(internal::constexpr_asin(-2.0) != internal::constexpr_asin(-2.0));
    EXPECT_TRUE(std::signbit(internal::constexpr_asin(-0.)));
    EXPECT_TRUE(not std::signbit(internal::constexpr_asin(0.)));
    EXPECT_TRUE(internal::constexpr_asin(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_asin(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_asin(0) == 0);
  static_assert(internal::constexpr_asin(1) == pi/2);
  static_assert(internal::constexpr_asin(1.0L) == piL/2);
  static_assert(internal::constexpr_asin(1.0F) == piF/2);
  static_assert(internal::constexpr_asin(-1) == -pi/2);
  static_assert(internal::are_within_tolerance(internal::constexpr_asin(numbers::sqrt2_v<double>/2), pi/4));
  static_assert(internal::are_within_tolerance(internal::constexpr_asin(-numbers::sqrt2_v<double>/2), -pi/4));
  static_assert(internal::constexpr_asin(0.99995) > 0);
  static_assert(internal::constexpr_asin(-0.99995) < 0);
  EXPECT_NEAR(internal::constexpr_asin(numbers::sqrt2_v<double>/2), pi/4, 1e-9);
  EXPECT_NEAR(internal::constexpr_asin(-0.7), std::asin(-0.7), 1e-9);
  EXPECT_NEAR(internal::constexpr_asin(0.9), std::asin(0.9), 1e-9);
  EXPECT_NEAR(internal::constexpr_asin(0.99), std::asin(0.99), 1e-9);
  EXPECT_NEAR(internal::constexpr_asin(0.999), std::asin(0.999), 1e-9);
  EXPECT_NEAR(internal::constexpr_asin(-0.999), std::asin(-0.999), 1e-9);
  EXPECT_NEAR(internal::constexpr_asin(0.99999), std::asin(0.99999), 1e-9);
  EXPECT_NEAR(internal::constexpr_asin(0.99999999), std::asin(0.99999999), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_asin(std::complex<double>{numbers::sqrt2_v<double>/2, 0}), pi/4, 1e-9));
  EXPECT_PRED3(tolerance, internal::constexpr_asin(std::complex<double>{4.1, 3.1}), std::asin(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_asin(std::complex<double>{3.2, -4.2}), std::asin(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_asin(std::complex<double>{-3.3, 4.3}), std::asin(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_asin(std::complex<double>{-9.3, 10.3}), std::asin(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_asin(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_asin(std::integral_constant<int, 1>{}), pi/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_asin(internal::ScalarConstant<Likelihood::definitely, double, 1>{}), pi/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_asin(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 0.6339838656391767163188, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_asin(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 2.305509031243476942042, 1e-9));
}


TEST(basics, constexpr_acos)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_acos(internal::constexpr_NaN<double>()) != internal::constexpr_acos(internal::constexpr_NaN<double>()));
    EXPECT_TRUE(internal::constexpr_acos(-2) != internal::constexpr_acos(-2)); // NaN
    EXPECT_FALSE(std::signbit(internal::constexpr_cos(1)));
    EXPECT_TRUE(internal::constexpr_acos(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_acos(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_acos(0) == pi/2);
  static_assert(internal::constexpr_acos(1) == 0);
  static_assert(internal::constexpr_acos(-1) == pi);
  static_assert(internal::constexpr_acos(-1.0L) == piL);
  static_assert(internal::constexpr_acos(-1.0F) == piF);
  static_assert(internal::are_within_tolerance(internal::constexpr_acos(0.5), numbers::pi/3));
  static_assert(internal::are_within_tolerance(internal::constexpr_acos(-0.5), 2*numbers::pi/3));
  static_assert(internal::are_within_tolerance(internal::constexpr_acos(numbers::sqrt2_v<double>/2), pi/4));
  static_assert(internal::are_within_tolerance(internal::constexpr_acos(-numbers::sqrt2_v<double>/2), 3*pi/4));
  EXPECT_NEAR(internal::constexpr_acos(-0.7), std::acos(-0.7), 1e-9);
  EXPECT_NEAR(internal::constexpr_acos(0.9), std::acos(0.9), 1e-9);
  EXPECT_NEAR(internal::constexpr_acos(0.999), std::acos(0.999), 1e-9);
  EXPECT_NEAR(internal::constexpr_acos(-0.999), std::acos(-0.999), 1e-9);
  EXPECT_NEAR(internal::constexpr_acos(0.99999), std::acos(0.99999), 1e-9);
  EXPECT_NEAR(internal::constexpr_acos(0.9999999), std::acos(0.9999999), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_acos(std::complex<double>{0.5, 0}), pi/3, 1e-9));
  EXPECT_PRED3(tolerance, internal::constexpr_acos(std::complex<double>{4.1, 3.1}), std::acos(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_acos(std::complex<double>{3.2, -4.2}), std::acos(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_acos(std::complex<double>{-3.3, 4.3}), std::acos(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_acos(std::complex<double>{-9.3, 10.3}), std::acos(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_acos(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_acos(std::integral_constant<int, -1>{}), pi, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_acos(internal::ScalarConstant<Likelihood::definitely, double, -1>{}), pi, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_acos(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 0.9368124611557199029125, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_acos(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), -2.305509031243476942042, 1e-9));
}


TEST(basics, constexpr_atan)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_atan(internal::constexpr_NaN<double>()) != internal::constexpr_atan(internal::constexpr_NaN<double>()));
    EXPECT_TRUE(internal::constexpr_atan(internal::constexpr_infinity<double>()) == pi/2);
    EXPECT_TRUE(internal::constexpr_atan(-internal::constexpr_infinity<double>()) == -pi/2);
    EXPECT_TRUE(std::signbit(internal::constexpr_atan(-0.)));
    EXPECT_FALSE(std::signbit(internal::constexpr_atan(+0.)));
    EXPECT_TRUE(internal::constexpr_atan(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_atan(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_atan(0) == 0);
  static_assert(internal::are_within_tolerance(internal::constexpr_atan(1.), pi/4));
  static_assert(internal::are_within_tolerance(internal::constexpr_atan(-1.), -pi/4));
  static_assert(internal::are_within_tolerance(internal::constexpr_atan(-1.L), -piL/4));
  static_assert(internal::are_within_tolerance(internal::constexpr_atan(-1.F), -piF/4));
  EXPECT_NEAR(internal::constexpr_atan(-0.7), std::atan(-0.7), 1e-9);
  EXPECT_NEAR(internal::constexpr_atan(0.9), std::atan(0.9), 1e-9);
  EXPECT_NEAR(internal::constexpr_atan(5.0), std::atan(5.0), 1e-9);
  EXPECT_NEAR(internal::constexpr_atan(-10.0), std::atan(-10.0), 1e-9);
  EXPECT_NEAR(internal::constexpr_atan(100.0), std::atan(100.0), 1e-9);

  static_assert(internal::are_within_tolerance(internal::constexpr_atan(std::complex<double>{1, 0}), pi/4, 1e-9));
  EXPECT_PRED3(tolerance, internal::constexpr_atan(std::complex<double>{4.1, 0.}), std::atan(4.1), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atan(std::complex<double>{4.1, 3.1}), std::atan(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atan(std::complex<double>{3.2, -4.2}), std::atan(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atan(std::complex<double>{-3.3, 4.3}), std::atan(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atan(std::complex<double>{-9.3, 10.3}), std::atan(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_atan(std::complex<int>{3, -4})));

  static_assert(internal::are_within_tolerance(internal::constexpr_atan(std::integral_constant<int, 1>{}), pi/4, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_atan(internal::ScalarConstant<Likelihood::definitely, double, 1>{}), pi/4, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_atan(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 1.448306995231464542145, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_atan(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{})), 0.1589971916799991743648, 1e-9));
}


TEST(basics, constexpr_atan2)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(internal::constexpr_atan2(internal::constexpr_infinity<double>(), 0.f) == pi/2);
    EXPECT_TRUE(internal::constexpr_atan2(-internal::constexpr_infinity<double>(), 0.f) == -pi/2);
    EXPECT_TRUE(internal::constexpr_atan2(0., internal::constexpr_infinity<double>()) == 0);
    EXPECT_TRUE(internal::constexpr_atan2(0., -internal::constexpr_infinity<double>()) == pi);
    EXPECT_TRUE(internal::constexpr_atan2(internal::constexpr_infinity<double>(), internal::constexpr_infinity<double>()) == pi/4);
    EXPECT_TRUE(internal::constexpr_atan2(internal::constexpr_infinity<double>(), -internal::constexpr_infinity<double>()) == 3*pi/4);
    EXPECT_TRUE(internal::constexpr_atan2(-internal::constexpr_infinity<double>(), internal::constexpr_infinity<double>()) == -pi/4);
    EXPECT_TRUE(internal::constexpr_atan2(-internal::constexpr_infinity<double>(), -internal::constexpr_infinity<double>()) == -3*pi/4);
#ifdef __cpp_lib_constexpr_cmath
    EXPECT_TRUE(std::signbit(internal::constexpr_atan2(-0., internal::constexpr_infinity<double>())));
    static_assert(std::signbit(internal::constexpr_atan2(-0., +0.)));
    EXPECT_TRUE(not std::signbit(internal::constexpr_atan2(+0., internal::constexpr_infinity<double>())));
    static_assert(not std::signbit(internal::constexpr_atan2(+0., +0.)));
    EXPECT_TRUE(internal::constexpr_atan2(-0., -internal::constexpr_infinity<double>()) == -pi);
    static_assert(internal::constexpr_atan2(-0., -0.) == -pi);
    static_assert(internal::constexpr_atan2(-0., -1.) == -pi);
    static_assert(internal::constexpr_atan2(+0., -0.) == pi);
#endif
    EXPECT_TRUE(internal::constexpr_atan2(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}, std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_atan2(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}, std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_atan2(0, 0) == 0);
  static_assert(internal::constexpr_atan2(0, 1) == 0);
  static_assert(internal::constexpr_atan2(0, -1) == pi);
  static_assert(internal::constexpr_atan2(1, 0) == pi/2);
  static_assert(internal::constexpr_atan2(-1, 0) == -pi/2);
  static_assert(internal::are_within_tolerance(internal::constexpr_atan2(0.5, 0.5), pi/4));
  static_assert(internal::are_within_tolerance(internal::constexpr_atan2(1., -1.), 3*pi/4));
  static_assert(internal::are_within_tolerance(internal::constexpr_atan2(-0.5, 0.5), -pi/4));
  static_assert(internal::are_within_tolerance(internal::constexpr_atan2(-1.L, -1.L), -3*piL/4));
  static_assert(internal::are_within_tolerance(internal::constexpr_atan2(-1.F, -1.F), -3*piF/4));
  EXPECT_NEAR(internal::constexpr_atan2(-0.7, 4.5), std::atan2(-0.7, 4.5), 1e-9);
  EXPECT_NEAR(internal::constexpr_atan2(0.9, -2.3), std::atan2(0.9, -2.3), 1e-9);
  EXPECT_NEAR(internal::constexpr_atan2(5.0, 3.1), std::atan2(5.0, 3.1), 1e-9);
  EXPECT_NEAR(internal::constexpr_atan2(-10.0, 9.0), std::atan2(-10.0, 9.0), 1e-9);
  EXPECT_NEAR(internal::constexpr_atan2(100.0, 200.0), std::atan2(100.0, 200.0), 1e-9);

  static_assert(internal::constexpr_atan2(std::complex<double>{0, 0}, std::complex<double>{0, 0}) == 0.0);
  static_assert(internal::constexpr_atan2(std::complex<double>{0, 0}, std::complex<double>{1, 0}) == 0.0);
  static_assert(internal::are_within_tolerance(internal::constexpr_atan2(std::complex<double>{0, 0}, std::complex<double>{-1, 0}), pi, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_atan2(std::complex<double>{1, 0}, std::complex<double>{0, 0}), pi/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_atan2(std::complex<double>{-1, 0}, std::complex<double>{0, 0}), -pi/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_atan2(std::complex<double>{3.2, -4.2}, std::complex<double>{-4.1, 3.1})), -0.7993578098204363309621, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_atan2(std::complex<double>{3.2, -4.2}, std::complex<double>{-4.1, 3.1})), 0.1378262475816170392786, 1e-9));
  EXPECT_PRED3(tolerance, internal::constexpr_atan2(std::complex<double>{4.1, 3.1}, std::complex<double>{2.1, 5.1}), std::atan(std::complex<double>{4.1, 3.1} / std::complex<double>{2.1, 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atan2(std::complex<double>{-3.3, 4.3}, std::complex<double>{2.1, 5.1}), std::atan(std::complex<double>{-3.3, 4.3} / std::complex<double>{2.1, 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atan2(std::complex<double>{-9.3, 10.3}, std::complex<double>{-5.1, 2.1}), std::atan(std::complex<double>{-9.3, 10.3} / std::complex<double>{-5.1, 2.1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atan2(std::complex<double>{0., 0.}, std::complex<double>{0., 0.}), std::complex<double>{0}, 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atan2(std::complex<double>{0., 3.1}, std::complex<double>{2.1, 5.1}), std::atan(std::complex<double>{0., 3.1} / std::complex<double>{2.1, 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atan2(std::complex<double>{0., 3.1}, std::complex<double>{-2.1, 5.1}), std::atan(std::complex<double>{0., 3.1} / std::complex<double>{-2.1, 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atan2(std::complex<double>{4.1, 3.1}, std::complex<double>{0., 5.1}), std::atan(std::complex<double>{4.1, 3.1} / std::complex<double>{0., 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_atan2(std::complex<double>{-4.1, 3.1}, std::complex<double>{0., 5.1}), std::atan(std::complex<double>{-4.1, 3.1} / std::complex<double>{0., 5.1}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_atan2(std::complex<int>{3, -4}, std::complex<int>{2, 5})));

  static_assert(internal::are_within_tolerance(internal::constexpr_atan2(std::integral_constant<int, 1>{}, std::integral_constant<int, 0>{}), pi/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_atan2(internal::ScalarConstant<Likelihood::definitely, double, 1>{}, internal::ScalarConstant<Likelihood::definitely, double, 0>{}), pi/2, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_real(internal::constexpr_atan2(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{}, internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 5, 2>{})), 0.7420289940594557537102, 1e-9));
  static_assert(internal::are_within_tolerance(internal::constexpr_imag(internal::constexpr_atan2(internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 3, 4>{}, internal::ScalarConstant<Likelihood::definitely, std::complex<double>, 5, 2>{})), 0.2871556773106927669533, 1e-9));
}


TEST(basics, constexpr_pow)
{
  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(+0., 3U)));
    EXPECT_TRUE(std::signbit(internal::constexpr_pow(-0., 3U)));
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(+0., 2U)));
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(-0., 2U)));
    EXPECT_TRUE(internal::constexpr_pow(internal::constexpr_NaN<double>(), 0U) == 1);
    EXPECT_TRUE(internal::constexpr_pow(internal::constexpr_NaN<double>(), 1U) != internal::constexpr_pow(internal::constexpr_NaN<double>(), 1U));
    EXPECT_TRUE(internal::constexpr_pow(-internal::constexpr_infinity<double>(), 3U) == -internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_pow(-internal::constexpr_infinity<double>(), 4U) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_pow(internal::constexpr_infinity<double>(), 3U) == internal::constexpr_infinity<double>());

    EXPECT_TRUE(internal::constexpr_pow(+0., -3) == internal::constexpr_infinity<double>());
#ifdef __cpp_lib_constexpr_cmath
    EXPECT_TRUE(internal::constexpr_pow(-0., -3) == -internal::constexpr_infinity<double>());
#endif
    EXPECT_TRUE(internal::constexpr_pow(+0., -2) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_pow(-0., -2) == internal::constexpr_infinity<double>());
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(+0., 3)));
    EXPECT_TRUE(std::signbit(internal::constexpr_pow(-0., 3)));
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(+0., 2)));
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(-0., 2)));
    EXPECT_TRUE(internal::constexpr_pow(internal::constexpr_NaN<double>(), 0) == 1);
    EXPECT_TRUE(internal::constexpr_pow(internal::constexpr_NaN<double>(), 1) != internal::constexpr_pow(internal::constexpr_NaN<double>(), 1));
    EXPECT_TRUE(internal::constexpr_pow(-internal::constexpr_infinity<double>(), -3) == 0);
    EXPECT_TRUE(std::signbit(internal::constexpr_pow(-internal::constexpr_infinity<double>(), -3)));
    EXPECT_TRUE(internal::constexpr_pow(-internal::constexpr_infinity<double>(), -2) == 0);
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(-internal::constexpr_infinity<double>(), -2)));
    EXPECT_TRUE(internal::constexpr_pow(-internal::constexpr_infinity<double>(), 3) == -internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_pow(-internal::constexpr_infinity<double>(), 4) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_pow(internal::constexpr_infinity<double>(), -3) == 0);
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(internal::constexpr_infinity<double>(), -3)));
    EXPECT_TRUE(internal::constexpr_pow(internal::constexpr_infinity<double>(), 3) == internal::constexpr_infinity<double>());

    EXPECT_TRUE(internal::constexpr_pow(+0., -internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_pow(-0., -internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_pow(+0.5, -internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_pow(-0.5, -internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_pow(+1.5, -internal::constexpr_infinity<double>()) == 0);
    EXPECT_TRUE(internal::constexpr_pow(-1.5, -internal::constexpr_infinity<double>()) == 0);
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(+1.5, -internal::constexpr_infinity<double>())));
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(-1.5, -internal::constexpr_infinity<double>())));
    EXPECT_TRUE(internal::constexpr_pow(+0.5, internal::constexpr_infinity<double>()) == 0);
    EXPECT_TRUE(internal::constexpr_pow(-0.5, internal::constexpr_infinity<double>()) == 0);
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(+0.5, internal::constexpr_infinity<double>())));
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(-0.5, internal::constexpr_infinity<double>())));
    EXPECT_TRUE(internal::constexpr_pow(+1.5, internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_pow(-1.5, internal::constexpr_infinity<double>()) == internal::constexpr_infinity<double>());
    EXPECT_TRUE(internal::constexpr_pow(-internal::constexpr_infinity<double>(), -3.) == 0);
    EXPECT_TRUE(std::signbit(internal::constexpr_pow(-internal::constexpr_infinity<double>(), -3.))); // note: cpp reference says sign is reversed, which is maybe an error.
    EXPECT_EQ(internal::constexpr_pow(-internal::constexpr_infinity<double>(), 3.), -internal::constexpr_infinity<double>()); // note: cpp reference says sign is reversed, which is maybe an error.
    EXPECT_TRUE(internal::constexpr_pow(internal::constexpr_infinity<double>(), -3.) == 0);
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(internal::constexpr_infinity<double>(), -3.)));
    EXPECT_TRUE(internal::constexpr_pow(internal::constexpr_infinity<double>(), 3.) == internal::constexpr_infinity<double>());
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(+0, 3.)));
    EXPECT_FALSE(std::signbit(internal::constexpr_pow(-0, 3.)));
    EXPECT_TRUE(internal::constexpr_pow(-1., internal::constexpr_infinity<double>()) == 1);
    EXPECT_TRUE(internal::constexpr_pow(-1., -internal::constexpr_infinity<double>()) == 1);
    EXPECT_TRUE(internal::constexpr_pow(+1., internal::constexpr_NaN<double>()) == 1);
    EXPECT_TRUE(internal::constexpr_pow(internal::constexpr_NaN<double>(), +0) == 1);
    EXPECT_TRUE(internal::constexpr_pow(internal::constexpr_NaN<double>(), 1.) != internal::constexpr_pow(internal::constexpr_NaN<double>(), 1.));

    EXPECT_TRUE(internal::constexpr_pow(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}, std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}) != internal::constexpr_pow(std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}, std::complex<double>{internal::constexpr_NaN<double>(), internal::constexpr_NaN<double>()}));
  }

  static_assert(internal::constexpr_pow(+0., 3U) == 0);
  static_assert(internal::constexpr_pow(-0., 3U) == 0);
  static_assert(internal::constexpr_pow(+0., 2U) == 0);
  static_assert(internal::constexpr_pow(-0., 2U) == 0);
  static_assert(internal::constexpr_pow(1, 0U) == 1);
  static_assert(internal::constexpr_pow(0, 1U) == 0);
  static_assert(internal::constexpr_pow(1, 1U) == 1);
  static_assert(internal::constexpr_pow(1, 2U) == 1);
  static_assert(internal::constexpr_pow(2, 1U) == 2);
  static_assert(internal::constexpr_pow(2, 5U) == 32);
  static_assert(internal::constexpr_pow(2, 16U) == 65536);
  static_assert(std::is_floating_point_v<decltype(internal::constexpr_pow(2, 16U))>);
  static_assert(internal::constexpr_pow(2.0, 16U) == 65536);
  static_assert(std::is_floating_point_v<decltype(internal::constexpr_pow(2.0, 16U))>);

  static_assert(internal::constexpr_pow(+0., 3) == 0);
  static_assert(internal::constexpr_pow(-0., 3) == 0);
  static_assert(internal::constexpr_pow(+0., 2) == 0);
  static_assert(internal::constexpr_pow(-0., 2) == 0);
  static_assert(internal::constexpr_pow(2, -4) == 0.0625);
  static_assert(internal::constexpr_pow(2, -5) == 0.03125);
  static_assert(std::is_floating_point_v<decltype(internal::constexpr_pow(2, -4))>);

  static_assert(internal::constexpr_pow(+0., 3.) == +0);
  static_assert(internal::constexpr_pow(-0., 3.) == +0);
  static_assert(internal::constexpr_pow(+1., 5) == 1);
  static_assert(internal::constexpr_pow(-5., +0) == 1);
  EXPECT_TRUE(internal::constexpr_pow(-5., 1.5) != internal::constexpr_pow(-5., 1.5));
  EXPECT_TRUE(internal::constexpr_pow(-7.3, 3.3) != internal::constexpr_pow(-7.3, 3.3));
  static_assert(internal::are_within_tolerance(internal::constexpr_pow(2, -4.), 0.0625));
  static_assert(internal::are_within_tolerance(internal::constexpr_pow(10, -4.), 1e-4));
  static_assert(internal::are_within_tolerance(internal::constexpr_pow(10., 6.), 1e6, 1e-4));
  EXPECT_DOUBLE_EQ(internal::constexpr_pow(5.0L, 4.0L), std::pow(5.0L, 4.0L));
  EXPECT_DOUBLE_EQ(internal::constexpr_pow(5.0L, -4.0L), std::pow(5.0L, -4.0L));
  EXPECT_DOUBLE_EQ(internal::constexpr_pow(1e20L, 2.L), std::pow(1e20L, 2.L));
  EXPECT_DOUBLE_EQ(internal::constexpr_pow(1e20L, -2.L), std::pow(1e20L, -2.L));
  EXPECT_DOUBLE_EQ(internal::constexpr_pow(1e100L, 2.L), std::pow(1e100L, 2.L));
  EXPECT_DOUBLE_EQ(internal::constexpr_pow(1e100L, -2.L), std::pow(1e100L, -2.L));

  EXPECT_PRED3(tolerance, internal::constexpr_pow(2., std::complex<double>{-4}), std::pow(2., std::complex<double>{-4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_pow(2, std::complex<double>{-4}), std::pow(2, std::complex<double>{-4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_pow(std::complex<double>{3, 4}, 2.), std::pow(std::complex<double>{3, 4}, 2.), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_pow(std::complex<double>{3, 4}, 2), std::pow(std::complex<double>{3, 4}, 2), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_pow(std::complex<double>{3, 4}, -2), std::pow(std::complex<double>{3, 4}, -2), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_pow(std::complex<double>{3, 4}, 3), std::pow(std::complex<double>{3, 4}, 3), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_pow(std::complex<double>{3, 4}, -3), std::pow(std::complex<double>{3, 4}, -3), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_pow(2., std::complex<double>{3, -4}), std::pow(2., std::complex<double>{3, -4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_pow(std::complex<double>{1, 2}, std::complex<double>{-3, 4}), std::pow(std::complex<double>{1, 2}, std::complex<double>{-3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, internal::constexpr_pow(std::complex<double>{-3, -4}, std::complex<double>{1, 2}), std::pow(std::complex<double>{-3, -4}, std::complex<double>{1, 2}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(internal::constexpr_pow(std::complex<int>{-3, -4}, std::complex<int>{1, 2})));

  static_assert(internal::constexpr_pow(internal::ScalarConstant<Likelihood::definitely, double, 2>{}, std::integral_constant<int, 3>{}) == 8);
  static_assert(internal::constexpr_pow(internal::ScalarConstant<Likelihood::definitely, double, 2>{}, 3) == 8);
  static_assert(internal::are_within_tolerance(internal::constexpr_pow(2, internal::ScalarConstant<Likelihood::definitely, double, 3>{}), 8, 1e-6));
}
