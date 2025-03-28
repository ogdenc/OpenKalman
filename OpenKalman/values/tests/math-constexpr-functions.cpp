/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for scalar types and constexpr math functions.
 */

#include <complex>
#include "values/tests/tests.hpp"
#include "values/classes/Fixed.hpp"
#include "values/functions/internal/near.hpp"

using namespace OpenKalman;


#if defined(__GNUC__) or defined(__clang__)
#define COMPLEXINTEXISTS(F) F
#else
#define COMPLEXINTEXISTS(F)
#endif


namespace
{
  auto tolerance = [](const auto& a, const auto& b, const auto& err){ return value::internal::near(a, b, err); };
}


#include "values/math/internal/infinity.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/isinf.hpp"
#include "values/math/isnan.hpp"

TEST(values, infinity_nan)
{
  if (std::numeric_limits<double>::has_infinity)
  {
    static_assert(value::isinf(value::internal::infinity<double>()));
    static_assert(value::isinf(INFINITY));
    static_assert(value::isinf(-INFINITY));
  }
  if (std::numeric_limits<double>::has_quiet_NaN or std::numeric_limits<double>::has_signaling_NaN)
  {
    static_assert(value::isnan(value::internal::NaN<double>()));
    static_assert(value::isnan(NAN));
    static_assert(value::isnan(-NAN));
  }
}


#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/math/conj.hpp"

TEST(values, real_imag_conj)
{
  static_assert(std::is_floating_point_v<decltype(value::real(3))>);
  static_assert(std::is_floating_point_v<decltype(value::imag(3))>);
  static_assert(std::is_floating_point_v<decltype(value::real(value::conj(3)))>);
  static_assert(std::is_floating_point_v<decltype(value::real(std::complex<double>{3., 4.}))>);
  COMPLEXINTEXISTS(static_assert(value::integral<decltype(value::real(std::complex<int>{3, 4}))>));
  COMPLEXINTEXISTS(static_assert(value::integral<decltype(value::imag(std::complex<int>{3, 4}))>));
  COMPLEXINTEXISTS(static_assert(value::integral<decltype(value::real(value::conj(std::complex<int>{3, 4})))>));
  static_assert(value::real(3.) == 3);
  static_assert(value::real(3.f) == 3);
  static_assert(value::real(3.l) == 3);
  static_assert(value::imag(3.) == 0);
  static_assert(value::imag(3.f) == 0);
  static_assert(value::imag(3.l) == 0);
  static_assert(value::conj(3.) == 3.);
  static_assert(value::conj(3.f) == 3.f);
  static_assert(value::conj(3.l) == 3.l);
  EXPECT_EQ(value::real(std::complex<double>{3, 4}), 3);
  EXPECT_EQ(value::imag(std::complex<double>{3, 4}), 4);
  EXPECT_TRUE((value::conj(std::complex<double>{3, 4}) == std::complex<double>{3, -4}));

  static_assert(value::real(std::integral_constant<int, 9>{}) == 9);
  static_assert(value::real(value::Fixed<std::complex<double>, 3, 4>{}) == 3);
  static_assert(value::imag(std::integral_constant<int, 9>{}) == 0);
  static_assert(value::imag(value::Fixed<std::complex<double>, 3, 4>{}) == 4);
  static_assert(value::real(value::conj(std::integral_constant<int, 9>{})) == 9);
  static_assert(value::imag(value::conj(std::integral_constant<int, 9>{})) == 0);
  static_assert(value::real(value::conj(value::Fixed<std::complex<double>, 3, 4>{})) == 3);
  static_assert(value::imag(value::conj(value::Fixed<std::complex<double>, 3, 4>{})) == -4);
}


#include "values/math/signbit.hpp"

TEST(values, signbit)
{
  static_assert(not value::signbit(0));
  static_assert(value::signbit(-3));
  static_assert(not value::signbit(3));
  static_assert(not value::signbit(3.));
  static_assert(value::signbit(-3.));
  static_assert(not value::signbit(3.));
  static_assert(value::signbit(-3.f));
  static_assert(not value::signbit(3.f));
  static_assert(value::signbit(-3.l));
  static_assert(not value::signbit(3.l));
  static_assert(not value::signbit(INFINITY));
  static_assert(value::signbit(-INFINITY));
#ifdef __cpp_lib_constexpr_cmath
  static_assert(not value::signbit(0.));
  static_assert(value::signbit(+0.) == std::signbit(+0.));
  static_assert(value::signbit(-0.) == std::signbit(-0.));
  static_assert(value::signbit(NAN) == std::signbit(NAN));
  static_assert(value::signbit(-NAN) == std::signbit(-NAN));
#endif
  EXPECT_EQ(value::signbit(+0.), std::signbit(+0.));
  EXPECT_EQ(value::signbit(NAN), std::signbit(NAN));
  EXPECT_EQ(value::signbit(INFINITY), std::signbit(INFINITY));
  EXPECT_EQ(value::signbit(-INFINITY), std::signbit(-INFINITY));

  static_assert(value::signbit(std::integral_constant<int, -3>{}));
  static_assert(value::signbit(value::Fixed<double, -3>{}));
  static_assert(not value::signbit(value::Fixed<double, 3>{}));
}


#include "values/math/copysign.hpp"

TEST(values, copysign)
{
  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    static_assert(value::copysign(3.f, -INFINITY) == -3.f);
    static_assert(value::copysign(3.f, INFINITY) == 3.f);
    static_assert(value::copysign(INFINITY, -3.f) == -INFINITY);
    static_assert(value::copysign(-INFINITY, 3.f) == INFINITY);

    static_assert(value::copysign(0., -1.) == 0.);
    static_assert(value::copysign(0., 1.) == 0.);
    static_assert(value::copysign(-0., -1.) == -0.);
    static_assert(value::copysign(-0., 1.) == -0.);

#ifdef __cpp_lib_constexpr_cmath
    static_assert(std::signbit(value::copysign(-1., 0.)) == std::signbit(std::copysign(-1., 0.)));
    static_assert(std::signbit(value::copysign(1., 0.)) == std::signbit(std::copysign(1., 0.)));
    static_assert(std::signbit(value::copysign(-1., -0.)) == std::signbit(std::copysign(-1., -0.)));
    static_assert(std::signbit(value::copysign(1., -0.)) == std::signbit(std::copysign(1., -0.)));

    static_assert(std::signbit(value::copysign(0., -1.)) == std::signbit(std::copysign(0., -1.)));
    static_assert(std::signbit(value::copysign(0., 1.)) == std::signbit(std::copysign(0., 1.)));
    static_assert(std::signbit(value::copysign(-0., -1.)) == std::signbit(std::copysign(-0., -1.)));
    static_assert(std::signbit(value::copysign(-0., 1.)) == std::signbit(std::copysign(-0., 1.)));
#else
    EXPECT_EQ(std::signbit(value::copysign(-1., 0.)), std::signbit(std::copysign(-1., 0.)));
    EXPECT_EQ(std::signbit(value::copysign(1., 0.)), std::signbit(std::copysign(1., 0.)));

    EXPECT_EQ(std::signbit(value::copysign(0., -1.)), std::signbit(std::copysign(0., -1.)));
    EXPECT_EQ(std::signbit(value::copysign(0., 1.)), std::signbit(std::copysign(0., 1.)));
    EXPECT_EQ(std::signbit(value::copysign(-0., -1.)), std::signbit(std::copysign(-0., -1.)));
    EXPECT_EQ(std::signbit(value::copysign(-0., 1.)), std::signbit(std::copysign(-0., 1.)));
#endif

    auto NaN = std::numeric_limits<double>::quiet_NaN();

    EXPECT_TRUE(std::isnan(value::copysign(NaN, -1.)));
    EXPECT_TRUE(std::isnan(value::copysign(NaN, 1.)));
    EXPECT_TRUE(std::isnan(value::copysign(-NaN, -1.)));
    EXPECT_TRUE(std::isnan(value::copysign(-NaN, 1.)));

    EXPECT_TRUE(std::signbit(value::copysign(NaN, -1.)));
    EXPECT_FALSE(std::signbit(value::copysign(NaN, 1.)));
    EXPECT_TRUE(std::signbit(value::copysign(-NaN, -1.)));
    EXPECT_FALSE(std::signbit(value::copysign(-NaN, 1.)));
  }

  static_assert(value::copysign(3., -5.) == -3.);
  static_assert(value::copysign(-3., 5.) == 3.);
  static_assert(value::copysign(-3.f, 5.f) == 3.f);
  static_assert(value::copysign(3.l, -5.l) == -3.l);

  static_assert(value::copysign(5U, 3U) == 5.);
  static_assert(std::is_floating_point_v<decltype(value::copysign(5U, 3U))>);
  static_assert(value::copysign(5, -3) == -5.);
  static_assert(std::is_floating_point_v<decltype(value::copysign(5, -3))>);
  static_assert(value::copysign(5, 3) == 5.);
  static_assert(value::copysign(5U, 3) == 5.);
  static_assert(value::copysign(5U, -3) == -5.);
  static_assert(std::is_floating_point_v<decltype(value::copysign(5U, -3))>);
  static_assert(value::copysign(5, 3U) == 5.);
  static_assert(value::copysign(-5, 3U) == 5.);

  static_assert(value::copysign(value::Fixed<int, 5>{}, value::Fixed<int, -3>{}) == -5);
  static_assert(value::copysign(value::Fixed<int, -5>{}, value::Fixed<int, 3>{}) == 5);
  static_assert(value::copysign(value::Fixed<double, 5>{}, value::Fixed<double, -3>{}) == -5);
  static_assert(value::copysign(value::Fixed<double, -5>{}, value::Fixed<double, 3>{}) == 5);
}


#include "values/math/sqrt.hpp"

TEST(values, sqrt)
{
  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    constexpr auto inf = value::internal::infinity<double>();
    constexpr auto NaN = value::internal::NaN<double>();

    EXPECT_TRUE(std::signbit(value::sqrt(-0.)));
    EXPECT_FALSE(std::signbit(value::sqrt(+0.)));
    EXPECT_TRUE(std::isinf(value::sqrt(inf)));
    EXPECT_EQ(std::isinf(value::sqrt(inf)), std::isinf(std::sqrt(inf)));
    EXPECT_EQ(std::isnan(value::sqrt(-1)), std::isnan(std::sqrt(-1)));
    EXPECT_TRUE(std::isnan(value::sqrt(NaN)));
    EXPECT_EQ(std::isnan(value::sqrt(NaN)), std::isnan(std::sqrt(NaN)));

    EXPECT_FALSE(std::signbit(std::real(value::sqrt(std::complex<double>{+0.0, +0.0}))));
    EXPECT_FALSE(std::signbit(std::imag(value::sqrt(std::complex<double>{+0.0, +0.0}))));
    EXPECT_FALSE(std::signbit(std::real(value::sqrt(std::complex<double>{+0.0, -0.0}))));
    EXPECT_FALSE(std::signbit(std::real(value::sqrt(std::complex<double>{-0.0, +0.0}))));
    EXPECT_FALSE(std::signbit(std::imag(value::sqrt(std::complex<double>{-0.0, +0.0}))));
    EXPECT_FALSE(std::signbit(std::real(value::sqrt(std::complex<double>{-0.0, -0.0}))));
#ifdef __cpp_lib_constexpr_cmath
    EXPECT_TRUE(std::signbit(std::imag(value::sqrt(std::complex<double>{+0.0, -0.0}))));
    EXPECT_TRUE(std::signbit(std::imag(value::sqrt(std::complex<double>{-0.0, -0.0}))));
#endif

    static_assert(value::sqrt(std::complex<double>{+inf, +inf}) == std::complex<double>(+inf, +inf));
    EXPECT_TRUE((value::sqrt(std::complex<double>{+inf, +inf}) == std::sqrt(std::complex<double>{+inf, +inf})));
    static_assert(value::sqrt(std::complex<double>{+inf, -inf}) == std::complex<double>(+inf, -inf));
    EXPECT_TRUE((value::sqrt(std::complex<double>{+inf, -inf}) == std::sqrt(std::complex<double>{+inf, -inf})));
    static_assert(value::sqrt(std::complex<double>{-inf, +inf}) == std::complex<double>(+inf, +inf));
    EXPECT_TRUE((value::sqrt(std::complex<double>{-inf, +inf}) == std::sqrt(std::complex<double>{-inf, +inf})));
    static_assert(value::sqrt(std::complex<double>{-inf, -inf}) == std::complex<double>(+inf, -inf));
    EXPECT_TRUE((value::sqrt(std::complex<double>{-inf, -inf}) == std::sqrt(std::complex<double>{-inf, -inf})));

    static_assert(value::sqrt(std::complex<double>{1, +inf}) == std::complex<double>(inf, inf));
    EXPECT_TRUE((value::sqrt(std::complex<double>{1, +inf}) == std::sqrt(std::complex<double>{1, +inf})));
    static_assert(value::sqrt(std::complex<double>{1, -inf}) == std::complex<double>(inf, -inf));
    EXPECT_TRUE((value::sqrt(std::complex<double>{1, -inf}) == std::sqrt(std::complex<double>{1, -inf})));
    static_assert(value::sqrt(std::complex<double>{-1, +inf}) == std::complex<double>(inf, inf));
    EXPECT_TRUE((value::sqrt(std::complex<double>{-1, +inf}) == std::sqrt(std::complex<double>{-1, +inf})));
    static_assert(value::sqrt(std::complex<double>{-1, -inf}) == std::complex<double>(inf, -inf));
    EXPECT_TRUE((value::sqrt(std::complex<double>{-1, -inf}) == std::sqrt(std::complex<double>{-1, -inf})));

    static_assert(value::sqrt(std::complex<double>{ +inf, 1}) == std::complex<double>(inf, 0));
    EXPECT_TRUE((value::sqrt(std::complex<double>{ +inf, 1}) == std::sqrt(std::complex<double>{+inf, 1})));
    static_assert(value::sqrt(std::complex<double>{ -inf, 1}) == std::complex<double>(0, inf));
    EXPECT_TRUE((value::sqrt(std::complex<double>{ -inf, 1}) == std::sqrt(std::complex<double>{-inf, 1})));
    static_assert(value::sqrt(std::complex<double>{ +inf, -1}) == std::complex<double>(inf, -0.));
    EXPECT_TRUE((value::sqrt(std::complex<double>{ +inf, -1}) == std::sqrt(std::complex<double>{+inf, -1})));
    EXPECT_EQ(std::signbit(std::imag(value::sqrt(std::complex<double>{ +inf, -1}))), std::signbit(std::imag(std::sqrt(std::complex<double>{ +inf, -1}))));
    static_assert(value::sqrt(std::complex<double>{ -inf, -1}) == std::complex<double>(0, -inf));
    EXPECT_TRUE((value::sqrt(std::complex<double>{ -inf, -1}) == std::sqrt(std::complex<double>{-inf, -1})));

    static_assert(std::real(value::sqrt(std::complex<double>{ NaN, +inf})) == inf);
    EXPECT_DOUBLE_EQ(std::real(value::sqrt(std::complex<double>{ NaN, +inf})), std::real(std::sqrt(std::complex<double>{NaN, +inf})));
    static_assert(std::imag(value::sqrt(std::complex<double>{ NaN, +inf})) == inf);
    EXPECT_DOUBLE_EQ(std::imag(value::sqrt(std::complex<double>{ NaN, +inf})), std::imag(std::sqrt(std::complex<double>{NaN, +inf})));
    static_assert(std::real(value::sqrt(std::complex<double>{ NaN, -inf})) == inf);
    EXPECT_DOUBLE_EQ(std::real(value::sqrt(std::complex<double>{ NaN, -inf})), std::real(std::sqrt(std::complex<double>{NaN, -inf})));
    static_assert(std::imag(value::sqrt(std::complex<double>{ NaN, -inf})) == -inf);
    EXPECT_DOUBLE_EQ(std::imag(value::sqrt(std::complex<double>{ NaN, -inf})), std::imag(std::sqrt(std::complex<double>{NaN, -inf})));
    static_assert(std::real(value::sqrt(std::complex<double>{ +inf, NaN})) == inf);
    EXPECT_DOUBLE_EQ(std::real(value::sqrt(std::complex<double>{ +inf, NaN})), std::real(std::sqrt(std::complex<double>{+inf, NaN})));
    EXPECT_EQ(std::isnan(std::imag(value::sqrt(std::complex<double>{ +inf, NaN}))), std::isnan(std::imag(std::sqrt(std::complex<double>{ +inf, NaN}))));
    EXPECT_EQ(std::isnan(std::real(value::sqrt(std::complex<double>{ -inf, NaN}))), std::isnan(std::real(std::sqrt(std::complex<double>{ -inf, NaN}))));
    static_assert(std::imag(value::sqrt(std::complex<double>{ -inf, NaN})) == inf);
    EXPECT_DOUBLE_EQ(std::imag(value::sqrt(std::complex<double>{ -inf, NaN})), std::imag(std::sqrt(std::complex<double>{-inf, NaN})));

    EXPECT_TRUE(std::isnan(std::real(value::sqrt(std::complex<double>{1, NaN}))));
    EXPECT_TRUE(std::isnan(std::real(std::sqrt(std::complex<double>{1, NaN}))));
    EXPECT_TRUE(std::isnan(std::imag(value::sqrt(std::complex<double>{1, NaN}))));
    EXPECT_TRUE(std::isnan(std::imag(std::sqrt(std::complex<double>{1, NaN}))));
    EXPECT_TRUE(std::isnan(std::real(value::sqrt(std::complex<double>{NaN, 1}))));
    EXPECT_TRUE(std::isnan(std::real(std::sqrt(std::complex<double>{NaN, 1}))));
    EXPECT_TRUE(std::isnan(std::imag(value::sqrt(std::complex<double>{NaN, 1}))));
    EXPECT_TRUE(std::isnan(std::imag(std::sqrt(std::complex<double>{NaN, 1}))));

    EXPECT_TRUE(std::isnan(std::real(value::sqrt(std::complex<double>{NaN, NaN}))));
    EXPECT_TRUE(std::isnan(std::imag(value::sqrt(std::complex<double>{NaN, NaN}))));
    EXPECT_TRUE(std::isnan(std::real(value::sqrt(std::complex<double>{1, NaN}))));
    EXPECT_TRUE(std::isnan(std::imag(value::sqrt(std::complex<double>{1, NaN}))));
    EXPECT_TRUE(std::isnan(std::real(value::sqrt(std::complex<double>{NaN, 1}))));
    EXPECT_TRUE(std::isnan(std::imag(value::sqrt(std::complex<double>{NaN, 1}))));
  }

  static_assert(value::sqrt(0) == 0);
  static_assert(value::sqrt(1) == 1);
  static_assert(value::sqrt(4) == 2);
  static_assert(value::sqrt(9) == 3);
  static_assert(value::sqrt(1000000) == 1000);
  static_assert(value::internal::near(value::sqrt(2.), numbers::sqrt2));
  static_assert(value::internal::near(value::sqrt(3.), numbers::sqrt3));
  static_assert(value::internal::near(value::sqrt(4.0e6), 2.0e3));
  static_assert(value::internal::near(value::sqrt(9.0e-2), 3.0e-1));
  static_assert(value::internal::near(value::sqrt(2.5e-11), 5.0e-6));
  EXPECT_NEAR(value::sqrt(5.), std::sqrt(5), 1e-9);
  EXPECT_NEAR(value::sqrt(1.0e20), std::sqrt(1.0e20), 1e-9);
  EXPECT_NEAR(value::sqrt(0.001), std::sqrt(0.001), 1e-9);
  EXPECT_NEAR(value::sqrt(1e-20), std::sqrt(1e-20), 1e-9);

  EXPECT_PRED3(tolerance, value::sqrt(std::complex<double>{-4}), std::sqrt(std::complex<double>{-4}), 1e-9);
  EXPECT_PRED3(tolerance, value::sqrt(std::complex<double>{3, 4}), std::sqrt(std::complex<double>{3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, value::sqrt(std::complex<double>{3, -4}), (std::complex<double>{2, -1}), 1e-9);
  EXPECT_PRED3(tolerance, value::sqrt(std::complex<double>{3, 4}), (std::complex<double>{2, 1}), 1e-9);
  EXPECT_PRED3(tolerance, value::sqrt(std::complex<double>{3, -4}), std::sqrt(std::complex<double>{3, -4}), 1e-9);
  EXPECT_PRED3(tolerance, value::sqrt(std::complex<double>{-3, 4}), std::sqrt(std::complex<double>{-3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, value::sqrt(std::complex<double>{-3, -4}), std::sqrt(std::complex<double>{-3, -4}), 1e-9);
  EXPECT_PRED3(tolerance, value::sqrt(std::complex<double>{-3e10, 4e10}), std::sqrt(std::complex<double>{-3e10, 4e10}), 1e-9);

  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::sqrt(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::sqrt(std::complex<int>{3, 4})));
  COMPLEXINTEXISTS(static_assert(std::is_same_v<decltype(value::sqrt(std::complex<int>{3, -4})), std::complex<int>>));
  COMPLEXINTEXISTS(static_assert(std::is_same_v<decltype(value::sqrt(std::complex<int>{4, -4})), std::complex<int>>));
  COMPLEXINTEXISTS(static_assert(value::sqrt(std::complex<int>{3, 4}) == std::complex<int>{2, 1}));
  COMPLEXINTEXISTS(static_assert(value::sqrt(std::complex<int>{10, 15}) == std::complex<int>{3, 2}));

  static_assert(value::sqrt(std::integral_constant<int, 9>{}) == 3);
  static_assert(value::internal::near(value::sqrt(value::Fixed<double, 9>{}), 3, 1e-6));
}


#include "values/math/hypot.hpp"

TEST(values, hypot)
{
  constexpr auto inf = value::internal::infinity<double>();
  constexpr auto NaN = value::internal::NaN<double>();

  static_assert(value::hypot(3) == 3);
  static_assert(value::hypot(-3) == 3);
  static_assert(value::hypot(3.) == 3);
  static_assert(value::hypot(-3.) == 3);
  static_assert(value::hypot(3, 4) == 5);
  static_assert(value::hypot(4, -3) == 5);
  static_assert(value::internal::near(value::hypot(2, 10, 11), 15, 1e-6));
  static_assert(value::internal::near(value::hypot(2, 4, 5, 10, 72), 73, 1e-6));
  static_assert(value::isnan(value::hypot(NaN)));
  static_assert(value::isnan(value::hypot(1, 2, NaN, 4)));
  static_assert(value::isinf(value::hypot(inf)));
  static_assert(value::isinf(value::hypot(1, 2, 3, inf, 5)));
  static_assert(value::isinf(value::hypot(1, NaN, 3, inf, 5)));
}


#include "values/math/abs.hpp"

TEST(values, abs)
{
  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    static_assert(value::abs(INFINITY) == INFINITY);
    static_assert(value::abs(-INFINITY) == INFINITY);
    EXPECT_TRUE(std::isnan(value::abs(-NAN)));
    EXPECT_FALSE(std::signbit(value::abs(NAN)));
#ifdef __cpp_lib_constexpr_cmath
    EXPECT_FALSE(std::signbit(value::abs(-0.)));
    EXPECT_FALSE(std::signbit(value::abs(-NAN)));
#endif
    EXPECT_TRUE(std::isnan(value::abs(std::complex<double>{3, NAN})));
    EXPECT_EQ(value::abs(std::complex<double>{INFINITY, 0}), INFINITY);
    EXPECT_EQ(value::abs(std::complex<double>{0, INFINITY}), INFINITY);
    EXPECT_EQ(value::abs(std::complex<double>{-INFINITY, 0}), INFINITY);
    EXPECT_EQ(value::abs(std::complex<double>{0, -INFINITY}), INFINITY);
    EXPECT_EQ(value::abs(std::complex<double>{INFINITY, 1}), INFINITY);
    EXPECT_EQ(value::abs(std::complex<double>{-INFINITY, 1}), INFINITY);
    EXPECT_EQ(value::abs(std::complex<double>{1, INFINITY}), INFINITY);
    EXPECT_EQ(value::abs(std::complex<double>{1, -INFINITY}), INFINITY);
    EXPECT_EQ(value::abs(std::complex<double>{INFINITY, NAN}), INFINITY);
    EXPECT_EQ(value::abs(std::complex<double>{-INFINITY, NAN}), INFINITY);
    EXPECT_EQ(value::abs(std::complex<double>{NAN, INFINITY}), INFINITY);
    EXPECT_EQ(value::abs(std::complex<double>{NAN, -INFINITY}), INFINITY);
  }

  static_assert(std::is_integral_v<decltype(value::abs(-3))>);
  static_assert(std::is_floating_point_v<decltype(value::abs(3.))>);
  static_assert(std::is_floating_point_v<decltype(value::abs(std::complex<double>{3., 4.}))>);
  static_assert(value::abs(3) == 3);
  static_assert(value::abs(-3) == 3);
  static_assert(value::abs(3.) == 3);
  static_assert(value::abs(-3.) == 3);
  static_assert(value::abs(-3.f) == 3);
  static_assert(value::abs(-3.l) == 3);

  EXPECT_EQ(value::abs(std::complex<double>{3, -4}), 5);
  EXPECT_EQ(value::abs(std::complex<double>{-3, 4}), 5);

  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::abs(std::complex<int>{3, 4})));
  COMPLEXINTEXISTS(static_assert(std::is_same_v<decltype(value::abs(std::complex<int>{3, -4})), int>));
  COMPLEXINTEXISTS(static_assert(value::abs(std::complex<int>{3, 4}) == 5));
  COMPLEXINTEXISTS(static_assert(value::abs(std::complex<int>{10, 15}) == 18));

  static_assert(value::abs(std::integral_constant<int, -9>{}) == 9);
  static_assert(value::internal::near(value::abs(value::Fixed<double, -9>{}), 9, 1e-6));
  static_assert(value::abs(value::Fixed<std::complex<double>, 3, 4>{}) == 5);
}


#include "values/math/exp.hpp"

TEST(values, exp)
{
  constexpr auto e = numbers::e_v<double>;
  constexpr auto eL = numbers::e_v<long double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::isnan(value::exp(value::internal::NaN<double>())));
    EXPECT_TRUE(value::exp(-value::internal::infinity<double>()) == 0);
    EXPECT_TRUE(value::exp(value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(std::isnan(value::real(value::exp(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}))));
    EXPECT_TRUE(std::isnan(value::imag(value::exp(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}))));
    EXPECT_TRUE(value::exp(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) !=
      value::exp(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::internal::near<10>(value::exp(0), 1));
  static_assert(value::internal::near<10>(value::exp(1), e));
  static_assert(value::internal::near<10>(value::exp(2), e*e));
  static_assert(value::internal::near<100>(value::exp(3), e*e*e));
  static_assert(value::internal::near<10>(value::exp(-1), 1/e));
  static_assert(value::internal::near<10>(value::exp(-2), 1/(e*e)));
  static_assert(value::internal::near<10>(value::exp(1.0), e));
  static_assert(value::internal::near<10>(value::exp(2.0), e*e));
  static_assert(value::internal::near<10>(value::exp(1.0L), eL));
  static_assert(value::internal::near<10>(value::exp(2.0L), eL*eL));
  static_assert(value::internal::near<100>(value::exp(3.0L), eL*eL*eL));
  EXPECT_NEAR(value::exp(5), std::exp(5), 1e-9);
  EXPECT_NEAR(value::exp(-10), std::exp(-10), 1e-9);
  EXPECT_NEAR(value::exp(50), std::exp(50), 1e8);
  EXPECT_NEAR(value::exp(50.5), std::exp(50.5), 1e8);
  EXPECT_NEAR(value::exp(300), std::exp(300), 1e120);
  EXPECT_NEAR(value::exp(300.7), std::exp(300.7), 1e120);
  EXPECT_NEAR(value::exp(1e-5), std::exp(1e-5), 1e-12);
  EXPECT_NEAR(value::exp(1e-10), std::exp(1e-10), 1e-16);

  static_assert(value::internal::near(value::real(value::exp(std::complex<double>{2, 0})), e*e, 1e-6));
  EXPECT_PRED3(tolerance, value::exp(std::complex<double>{3.3, -4.3}), std::exp(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::exp(std::complex<double>{10.4, 3.4}), std::exp(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, value::exp(std::complex<double>{-30.6, 20.6}), std::exp(std::complex<double>{-30.6, 20.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::exp(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(value::exp(std::complex<int>{3, -4}) == std::complex<int>{-13, 15}));

  static_assert(value::internal::near(value::exp(std::integral_constant<int, 2>{}), e*e, 1e-6));
  static_assert(value::internal::near(value::exp(value::Fixed<double, -2>{}), 1/(e*e), 1e-6));
  static_assert(value::internal::near(value::real(value::exp(value::Fixed<std::complex<double>, 2, 0>{})), e*e, 1e-6));
}


#include "values/math/expm1.hpp"

TEST(values, expm1)
{
  constexpr auto e = numbers::e_v<double>;
  constexpr auto eL = numbers::e_v<long double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(value::expm1(value::internal::NaN<double>()) != value::expm1(value::internal::NaN<double>()));
    EXPECT_TRUE(value::expm1(value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(value::expm1(-value::internal::infinity<double>()) == -1);
    EXPECT_TRUE(std::signbit(value::expm1(-0.)));
    EXPECT_FALSE(std::signbit(value::expm1(+0.)));
    EXPECT_TRUE(value::expm1(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::expm1(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::expm1(0) == 0);
  static_assert(value::internal::near(value::expm1(1), e - 1));
  static_assert(value::internal::near(value::expm1(2), e*e - 1, 1e-9));
  static_assert(value::internal::near(value::expm1(3), e*e*e - 1, 1e-9));
  static_assert(value::internal::near(value::expm1(-1), 1 / e - 1, 1e-9));
  static_assert(value::internal::near(value::expm1(3.L), eL*eL*eL - 1, 1e-9));
  static_assert(std::real(value::expm1(std::complex<double>{3e-12, 0})) == value::expm1(3e-12));
  EXPECT_PRED3(tolerance, value::expm1(1e-4), std::expm1(1e-4), 1e-9);
  EXPECT_PRED3(tolerance, value::expm1(1e-8), std::expm1(1e-8), 1e-9);
  EXPECT_PRED3(tolerance, value::expm1(1e-32), std::expm1(1e-32), 1e-9);
  EXPECT_PRED3(tolerance, value::expm1(5.2), std::expm1(5.2), 1e-9);
  EXPECT_PRED3(tolerance, value::expm1(10.2), std::expm1(10.2), 1e-9);
  EXPECT_PRED3(tolerance, value::expm1(-10.2), std::expm1(-10.2), 1e-9);
  EXPECT_PRED3(tolerance, value::expm1(3e-12), std::expm1(3e-12), 1e-9);
  EXPECT_PRED3(tolerance, value::expm1(-3e-12), std::expm1(-3e-12), 1e-9);

  static_assert(value::internal::near(value::real(value::expm1(std::complex<double>{2, 0})), e*e - 1, 1e-6));
  EXPECT_PRED3(tolerance, value::expm1(std::complex<double>{0.001, -0.001}), std::exp(std::complex<double>{0.001, -0.001}) - 1.0, 1e-9);
  EXPECT_PRED3(tolerance, value::expm1(std::complex<double>{3.2, -4.2}), std::exp(std::complex<double>{3.2, -4.2}) - 1.0, 1e-9);
  EXPECT_PRED3(tolerance, value::expm1(std::complex<double>{10.3, 3.3}), std::exp(std::complex<double>{10.3, 3.3}) - 1.0, 1e-9);
  EXPECT_PRED3(tolerance, value::expm1(std::complex<double>{-10.4, 10.4}), std::exp(std::complex<double>{-10.4, 10.4}) - 1.0, 1e-9);
  EXPECT_PRED3(tolerance, std::real(value::expm1(std::complex<double>{3e-12, 0})), std::expm1(3e-12), 1e-20);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::expm1(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(value::expm1(std::complex<int>{3, -4}) == std::complex<int>{-14, 15}));

  static_assert(value::internal::near(value::expm1(std::integral_constant<int, 2>{}), e*e - 1, 1e-6));
  static_assert(value::internal::near(value::expm1(value::Fixed<double, -2>{}), 1/(e*e) - 1, 1e-6));
  static_assert(value::internal::near(value::real(value::expm1(value::Fixed<std::complex<double>, 2, 0>{})), e*e - 1, 1e-6));
}


#include "values/math/sinh.hpp"

TEST(values, sinh)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(value::sinh(value::internal::NaN<double>()) != value::sinh(value::internal::NaN<double>()));
    EXPECT_TRUE(value::sinh(value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(value::sinh(-value::internal::infinity<double>()) == -value::internal::infinity<double>());
    EXPECT_TRUE(std::signbit(value::sinh(-0.)));
    EXPECT_TRUE(not std::signbit(value::sinh(0.)));
    EXPECT_TRUE(value::sinh(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::sinh(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::sinh(0) == 0);
  static_assert(value::internal::near(value::sinh(1), (e - 1/e)/2, 1e-9));
  static_assert(value::internal::near(value::sinh(2), (e*e - 1/e/e)/2, 1e-9));
  static_assert(value::internal::near(value::sinh(3), (e*e*e - 1/e/e/e)/2, 1e-9));
  static_assert(value::internal::near(value::sinh(-1), (1/e - e)/2, 1e-9));
  static_assert(value::internal::near(value::sinh(-2), (1/e/e - e*e)/2, 1e-9));
  static_assert(value::internal::near(value::sinh(-3), (1/e/e/e - e*e*e)/2, 1e-9));
  EXPECT_NEAR(value::sinh(5), std::sinh(5), 1e-9);
  EXPECT_NEAR(value::sinh(-10), std::sinh(-10), 1e-9);

  static_assert(value::internal::near(value::real(value::sinh(std::complex<double>{2, 0})), (e*e - 1/e/e)/2, 1e-9));
  EXPECT_PRED3(tolerance, value::sinh(std::complex<double>{3.3, -4.3}), std::sinh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::sinh(std::complex<double>{10.4, 3.4}), std::sinh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, value::sinh(std::complex<double>{-10.6, 10.6}), std::sinh(std::complex<double>{-10.6, 10.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::sinh(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(value::sinh(std::complex<int>{3, -4}) == std::complex<int>{-6, 7}));

  static_assert(value::internal::near(value::sinh(std::integral_constant<int, 2>{}), (e*e - 1/e/e)/2, 1e-9));
  static_assert(value::internal::near(value::sinh(value::Fixed<double, -2>{}), (1/e/e - e*e)/2, 1e-9));
  static_assert(value::internal::near(value::real(value::sinh(value::Fixed<std::complex<double>, 2, 0>{})), (e*e - 1/e/e)/2, 1e-9));
  static_assert(value::internal::near(value::real(value::sinh(value::Fixed<std::complex<double>, 3, -4>{})), -6.548120040911001647767, 1e-9));
  static_assert(value::internal::near(value::imag(value::sinh(value::Fixed<std::complex<double>, 3, -4>{})), 7.619231720321410208487, 1e-9));
}


#include "values/math/cosh.hpp"

TEST(values, cosh)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(value::cosh(value::internal::NaN<double>()) != value::cosh(value::internal::NaN<double>()));
    EXPECT_TRUE(value::cosh(value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(value::cosh(-value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(value::cosh(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::cosh(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::cosh(0) == 1);
  static_assert(value::internal::near(value::cosh(1), (e + 1/e)/2, 1e-9));
  static_assert(value::internal::near(value::cosh(2), (e*e + 1/e/e)/2, 1e-9));
  static_assert(value::internal::near(value::cosh(3), (e*e*e + 1/e/e/e)/2, 1e-9));
  static_assert(value::internal::near(value::cosh(-1), (1/e + e)/2, 1e-9));
  static_assert(value::internal::near(value::cosh(-2), (1/e/e + e*e)/2, 1e-9));
  static_assert(value::internal::near(value::cosh(-3), (1/e/e/e + e*e*e)/2, 1e-9));
  EXPECT_NEAR(value::cosh(5), std::cosh(5), 1e-9);
  EXPECT_NEAR(value::cosh(-10), std::cosh(-10), 1e-9);

  static_assert(value::internal::near(value::real(value::cosh(std::complex<double>{2, 0})), (e*e + 1/e/e)/2, 1e-9));
  EXPECT_PRED3(tolerance, value::cosh(std::complex<double>{3.3, -4.3}), std::cosh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::cosh(std::complex<double>{10.4, 3.4}), std::cosh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, value::cosh(std::complex<double>{-10.6, 10.6}), std::cosh(std::complex<double>{-10.6, 10.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::cosh(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(value::cosh(std::complex<int>{5, -6}) == std::complex<int>{71, 20}));

  static_assert(value::internal::near(value::cosh(std::integral_constant<int, 2>{}), (e*e + 1/e/e)/2, 1e-9));
  static_assert(value::internal::near(value::cosh(value::Fixed<double, -2>{}), (1/e/e + e*e)/2, 1e-9));
  static_assert(value::internal::near(value::real(value::cosh(value::Fixed<std::complex<double>, 2, 0>{})), (e*e + 1/e/e)/2, 1e-9));
  static_assert(value::internal::near(value::real(value::cosh(value::Fixed<std::complex<double>, 3, -4>{})), -6.580663040551156432561, 1e-9));
  static_assert(value::internal::near(value::imag(value::cosh(value::Fixed<std::complex<double>, 3, -4>{})), 7.581552742746544353716, 1e-9));
}


#include "values/math/tanh.hpp"

TEST(values, tanh)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(value::tanh(value::internal::NaN<double>()) != value::tanh(value::internal::NaN<double>()));
    EXPECT_TRUE(value::tanh(value::internal::infinity<double>()) == 1);
    EXPECT_TRUE(value::tanh(-value::internal::infinity<double>()) == -1);
    EXPECT_TRUE(std::signbit(value::tanh(-0.)));
    EXPECT_TRUE(not std::signbit(value::tanh(0.)));
    EXPECT_TRUE(value::tanh(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::tanh(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::tanh(0) == 0);
  static_assert(value::internal::near(value::tanh(1), (e*e - 1)/(e*e + 1), 1e-9));
  static_assert(value::internal::near(value::tanh(2), (e*e*e*e - 1)/(e*e*e*e + 1), 1e-9));
  static_assert(value::internal::near(value::tanh(3), (e*e*e*e*e*e - 1)/(e*e*e*e*e*e + 1), 1e-9));
  static_assert(value::internal::near(value::tanh(-1), (1 - e*e)/(1 + e*e), 1e-9));
  static_assert(value::internal::near(value::tanh(-2), (1 - e*e*e*e)/(1 + e*e*e*e), 1e-9));
  static_assert(value::internal::near(value::tanh(-3), (1 - e*e*e*e*e*e)/(1 + e*e*e*e*e*e), 1e-9));
  EXPECT_NEAR(value::tanh(5), std::tanh(5), 1e-9);
  EXPECT_NEAR(value::tanh(-10), std::tanh(-10), 1e-9);

  static_assert(value::internal::near(value::real(value::tanh(std::complex<double>{2, 0})), (e*e*e*e - 1)/(e*e*e*e + 1), 1e-9));
  EXPECT_PRED3(tolerance, value::tanh(std::complex<double>{3.3, -4.3}), std::tanh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::tanh(std::complex<double>{10.4, 3.4}), std::tanh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, value::tanh(std::complex<double>{-30.6, 20.6}), std::tanh(std::complex<double>{-30.6, 20.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::tanh(std::complex<int>{3, -4})));

  static_assert(value::internal::near(value::tanh(std::integral_constant<int, 2>{}), (e*e*e*e - 1)/(e*e*e*e + 1), 1e-9));
  static_assert(value::internal::near(value::tanh(value::Fixed<double, -2>{}), (1 - e*e*e*e)/(1 + e*e*e*e), 1e-9));
  static_assert(value::internal::near(value::real(value::tanh(value::Fixed<std::complex<double>, 3, 4>{})), 1.00070953606723293933, 1e-9));
  static_assert(value::internal::near(value::imag(value::tanh(value::Fixed<std::complex<double>, 3, 4>{})), 0.004908258067496060259079, 1e-9));
}


#include "values/math/sin.hpp"

TEST(values, sin)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::isnan(value::sin(value::internal::NaN<double>())));
    EXPECT_TRUE(std::isnan(value::sin(value::internal::infinity<double>())));
    EXPECT_TRUE(std::isnan(value::sin(-value::internal::infinity<double>())));
    EXPECT_FALSE(std::signbit(value::sin(+0.)));
    EXPECT_TRUE(std::signbit(value::sin(-0.)));
    EXPECT_TRUE(value::sin(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::sin(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::internal::near(value::sin(0), 0));
  static_assert(value::internal::near(value::sin(2*pi), 0));
  static_assert(value::internal::near(value::sin(-2*pi), 0));
  static_assert(value::internal::near(value::sin(pi), 0));
  static_assert(value::internal::near(value::sin(-pi), 0));
  static_assert(value::internal::near(value::sin(32*pi), 0, 1e-9));
  static_assert(value::internal::near(value::sin(-32*pi), 0, 1e-9));
  static_assert(value::internal::near(value::sin(0x1p16*pi), 0, 1e-9));
  static_assert(value::internal::near(value::sin(-0x1p16*pi), 0, 1e-9));
  static_assert(value::internal::near(value::sin(0x1p16L*piL), 0, 1e-9));
  static_assert(value::internal::near(value::sin(-0x1p16L*piL), 0, 1e-9));
  static_assert(value::internal::near(value::sin(0x1p16F*piF), 0, 1e-2));
  static_assert(value::internal::near(value::sin(-0x1p16F*piF), 0, 1e-2));
  static_assert(value::internal::near(value::sin(0x1p100L*piL), 0, 1.));
  static_assert(value::internal::near(value::sin(-0x1p100L*piL), 0, 1.));
  static_assert(value::internal::near(value::sin(0x1p180L*piL), 0, 1.));
  static_assert(value::internal::near(value::sin(-0x1p180L*piL), 0, 1.));
  static_assert(value::internal::near(value::sin(0x1p250L*piL), 0, 1.));
  static_assert(value::internal::near(value::sin(-0x1p250L*piL), 0, 1.));
  static_assert(value::internal::near(value::sin(pi/2), 1));
  static_assert(value::internal::near(value::sin(-pi/2), -1));
  static_assert(value::internal::near(value::sin(pi/4), numbers::sqrt2_v<double>/2));
  static_assert(value::internal::near(value::sin(piL/4), numbers::sqrt2_v<long double>/2));
  static_assert(value::internal::near(value::sin(piF/4), numbers::sqrt2_v<float>/2));
  static_assert(value::internal::near(value::sin(pi/4 + 32*pi), numbers::sqrt2_v<double>/2, 1e-9));
  EXPECT_NEAR(value::sin(2), std::sin(2), 1e-9);
  EXPECT_NEAR(value::sin(-32), std::sin(-32), 1e-9);
  EXPECT_NEAR(value::sin(0x1p16), std::sin(0x1p16), 1e-9);

  static_assert(value::internal::near(value::sin(std::complex<double>{pi/2, 0}), 1));
  EXPECT_PRED3(tolerance, value::sin(std::complex<double>{4.1, 3.1}), std::sin(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, value::sin(std::complex<double>{3.2, -4.2}), std::sin(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, value::sin(std::complex<double>{-3.3, 4.3}), std::sin(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::sin(std::complex<double>{-9.3, 10.3}), std::sin(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::sin(std::complex<int>{3, -4})));

  static_assert(value::internal::near(value::sin(std::integral_constant<int, 2>{}), 0.909297426825681695396, 1e-9));
  static_assert(value::internal::near(value::sin(value::Fixed<double, 2>{}), 0.909297426825681695396, 1e-9));
  static_assert(value::internal::near(value::real(value::sin(value::Fixed<std::complex<double>, 2, 0>{})), 0.909297426825681695396, 1e-9));
}


#include "values/math/cos.hpp"

TEST(values, cos)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::isnan(value::cos(value::internal::NaN<double>())));
    EXPECT_TRUE(std::isnan(value::cos(value::internal::infinity<double>())));
    EXPECT_TRUE(std::isnan(value::cos(-value::internal::infinity<double>())));
    EXPECT_TRUE(value::cos(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::cos(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::cos(2*pi) == 1);
  static_assert(value::cos(-2*pi) == 1);
  static_assert(value::cos(0) == 1);
  static_assert(value::internal::near(value::cos(pi), -1));
  static_assert(value::internal::near(value::cos(-pi), -1));
  static_assert(value::internal::near(value::cos(32*pi), 1));
  static_assert(value::internal::near(value::cos(-32*pi), 1));
  static_assert(value::internal::near(value::cos(0x1p16*pi), 1));
  static_assert(value::internal::near(value::cos(-0x1p16*pi), 1));
  static_assert(value::internal::near(value::cos(0x1p16L*piL), 1));
  static_assert(value::internal::near(value::cos(-0x1p16L*piL), 1));
  static_assert(value::internal::near(value::cos(0x1p16F*piF), 1, 1e-4));
  static_assert(value::internal::near(value::cos(-0x1p16F*piF), 1, 1e-4));
  static_assert(value::internal::near(value::cos(0x1p100L*piL), 1, 1.));
  static_assert(value::internal::near(value::cos(-0x1p100L*piL), 1, 1.));
  static_assert(value::internal::near(value::cos(0x1p180L*piL), 1, 1.));
  static_assert(value::internal::near(value::cos(-0x1p180L*piL), 1, 1.));
  static_assert(value::internal::near(value::cos(0x1p250L*piL), 1, 1.));
  static_assert(value::internal::near(value::cos(-0x1p250L*piL), 1, 1.));
  static_assert(value::internal::near(value::cos(pi/2), 0));
  static_assert(value::internal::near(value::cos(-pi/2), 0));
  static_assert(value::internal::near(value::cos(pi/4), numbers::sqrt2_v<double>/2));
  static_assert(value::internal::near(value::cos(piL/4), numbers::sqrt2_v<long double>/2));
  static_assert(value::internal::near(value::cos(piF/4), numbers::sqrt2_v<float>/2));
  EXPECT_NEAR(value::cos(2), std::cos(2), 1e-9);
  EXPECT_NEAR(value::cos(-32), std::cos(-32), 1e-9);
  EXPECT_NEAR(value::cos(0x1p16), std::cos(0x1p16), 1e-9);

  static_assert(value::internal::near(value::cos(std::complex<double>{pi/2, 0}), 0));
  EXPECT_PRED3(tolerance, value::cos(std::complex<double>{4.1, 3.1}), std::cos(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, value::cos(std::complex<double>{3.2, -4.2}), std::cos(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, value::cos(std::complex<double>{-3.3, 4.3}), std::cos(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::cos(std::complex<double>{-9.3, 10.3}), std::cos(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::cos(std::complex<int>{3, -4})));

  static_assert(value::internal::near(value::cos(std::integral_constant<int, 2>{}), -0.4161468365471423869976, 1e-9));
  static_assert(value::internal::near(value::cos(value::Fixed<double, 2>{}), -0.4161468365471423869976, 1e-9));
  static_assert(value::internal::near(value::real(value::cos(value::Fixed<std::complex<double>, 2, 0>{})), -0.4161468365471423869976, 1e-9));
}


#include "values/math/tan.hpp"

TEST(values, tan)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(value::tan(value::internal::NaN<double>()) != value::tan(value::internal::NaN<double>()));
    EXPECT_TRUE(value::tan(value::internal::infinity<double>()) != value::tan(value::internal::infinity<double>()));
    EXPECT_TRUE(value::tan(-value::internal::infinity<double>()) != value::tan(value::internal::infinity<double>()));
    EXPECT_TRUE(value::tan(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::tan(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::tan(0) == 0);
  static_assert(value::internal::near(value::tan(2*pi), 0));
  static_assert(value::internal::near(value::tan(-2*pi), 0));
  static_assert(value::internal::near(value::tan(pi), 0));
  static_assert(value::internal::near(value::tan(-pi), 0));
  static_assert(value::internal::near(value::tan(32*pi), 0, 1e-9));
  static_assert(value::internal::near(value::tan(-32*pi), 0, 1e-9));
  static_assert(value::internal::near(value::tan(0x1p16*pi), 0, 1e-9));
  static_assert(value::internal::near(value::tan(-0x1p16*pi), 0, 1e-9));
  static_assert(value::internal::near(value::tan(0x1p16L*piL), 0, 1e-9));
  static_assert(value::internal::near(value::tan(-0x1p16L*piL), 0, 1e-9));
  static_assert(value::internal::near(value::tan(0x1p16F*piF), 0, 1e-2));
  static_assert(value::internal::near(value::tan(-0x1p16F*piF), 0, 1e-2));
  static_assert(value::internal::near(value::tan(0x1p100L*piL), 0, 1.));
  static_assert(value::internal::near(value::tan(-0x1p100L*piL), 0, 1.));
  static_assert(value::internal::near(value::tan(0x1p180L*piL), 0, 1.));
  static_assert(value::internal::near(value::tan(-0x1p180L*piL), 0, 1.));
  static_assert(value::internal::near(value::tan(0x1p250L*piL), 0, 2.));
  static_assert(value::internal::near(value::tan(-0x1p250L*piL), 0, 2.));
  static_assert(value::internal::near(value::tan(pi/4), 1));
  static_assert(value::internal::near(value::tan(piL/4), 1));
  static_assert(value::internal::near(value::tan(piF/4), 1));
  static_assert(value::internal::near(value::tan(pi/4 + 32*pi), 1, 1e-9));
  EXPECT_NEAR(value::tan(2), std::tan(2), 1e-9);
  EXPECT_NEAR(value::tan(-32), std::tan(-32), 1e-9);
  EXPECT_NEAR(value::tan(0x1p16), std::tan(0x1p16), 1e-9);

  static_assert(value::internal::near(value::tan(std::complex<double>{pi/4, 0}), 1));
  EXPECT_PRED3(tolerance, value::tan(std::complex<double>{4.1, 3.1}), std::tan(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, value::tan(std::complex<double>{3.2, -4.2}), std::tan(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, value::tan(std::complex<double>{-3.3, 4.3}), std::tan(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::tan(std::complex<double>{-30.3, 40.3}), std::tan(std::complex<double>{-30.3, 40.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::tan(std::complex<int>{30, -2})));

  static_assert(value::internal::near(value::tan(std::integral_constant<int, 2>{}), -2.185039863261518991643, 1e-9));
  static_assert(value::internal::near(value::tan(value::Fixed<double, 2>{}), -2.185039863261518991643, 1e-9));
  static_assert(value::internal::near(value::real(value::tan(value::Fixed<std::complex<double>, 3, 4>{})), -1.873462046294784262243E-4, 1e-9));
  static_assert(value::internal::near(value::imag(value::tan(value::Fixed<std::complex<double>, 3, 4>{})), 0.9993559873814731413917, 1e-9));
}


#include "values/math/log.hpp"

TEST(values, log)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(value::log(0) == -value::internal::infinity<double>());
    EXPECT_TRUE(value::log(-0) == -value::internal::infinity<double>());
    EXPECT_TRUE(value::log(+value::internal::infinity<double>()) == +value::internal::infinity<double>());
    EXPECT_TRUE(value::log(-1) != value::log(-1)); // Nan
    EXPECT_FALSE(std::signbit(value::log(1)));
    EXPECT_TRUE(value::log(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::log(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::log(1) == 0);
  static_assert(value::internal::near<10>(value::log(2), numbers::ln2_v<double>));
  static_assert(value::internal::near<10>(value::log(10), numbers::ln10_v<double>));
  static_assert(value::internal::near(value::log(e), 1));
  static_assert(value::internal::near(value::log(e*e), 2));
  static_assert(value::internal::near(value::log(e*e*e), 3));
  static_assert(value::internal::near(value::log(e*e*e*e*e*e*e*e*e*e*e*e*e*e*e*e), 16));
  static_assert(value::internal::near(value::log(1 / e), -1));
  EXPECT_NEAR(value::log(5.0L), std::log(5.0L), 1e-9);
  EXPECT_NEAR(value::log(0.2L), std::log(0.2L), 1e-9);
  EXPECT_NEAR(value::log(5), std::log(5), 1e-9);
  EXPECT_NEAR(value::log(0.2), std::log(0.2), 1e-9);
  EXPECT_NEAR(value::log(20), std::log(20), 1e-9);
  EXPECT_NEAR(value::log(0.05), std::log(0.05), 1e-9);
  EXPECT_NEAR(value::log(100), std::log(100), 1e-9);
  EXPECT_NEAR(value::log(0.01), std::log(0.01), 1e-9);
  EXPECT_NEAR(value::log(1e20), std::log(1e20), 1e-9);
  EXPECT_NEAR(value::log(1e-20), std::log(1e-20), 1e-9);
  EXPECT_NEAR(value::log(1e200), std::log(1e200), 1e-9);
  EXPECT_NEAR(value::log(1e-200), std::log(1e-200), 1e-9);
  EXPECT_NEAR(value::log(1e200L), std::log(1e200L), 1e-9);
  EXPECT_NEAR(value::log(1e-200L), std::log(1e-200L), 1e-9);

  static_assert(value::internal::near(value::log(std::complex<double>{e*e, 0}), 2));
  EXPECT_PRED3(tolerance, value::log(std::complex<double>{-4}), std::log(std::complex<double>{-4}), 1e-9);
  EXPECT_PRED3(tolerance, value::log(std::complex<double>{3, 4}), std::log(std::complex<double>{3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, value::log(std::complex<double>{3, -4}), std::log(std::complex<double>{3, -4}), 1e-9);
  EXPECT_PRED3(tolerance, value::log(std::complex<double>{-3, 4}), std::log(std::complex<double>{-3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, value::log(std::complex<double>{-3, -4}), std::log(std::complex<double>{-3, -4}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::log(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(value::log(std::complex<int>{1, 0}) == std::complex<int>{0, 0}));
  COMPLEXINTEXISTS(static_assert(value::log(std::complex<int>{100, 0}) == std::complex<int>{4, 0}));
  COMPLEXINTEXISTS(static_assert(value::log(std::complex<int>{-100, 0}) == std::complex<int>{4, 3}));
  COMPLEXINTEXISTS(EXPECT_PRED3(tolerance, value::log(std::complex<int>{-3, 0}), std::log(std::complex<int>{-3, 0}), 1e-9));

  static_assert(value::internal::near(value::log(std::integral_constant<int, 2>{}), numbers::ln2_v<double>, 1e-9));
  static_assert(value::internal::near(value::log(value::Fixed<double, 2>{}), numbers::ln2_v<double>, 1e-9));
  static_assert(value::internal::near(value::real(value::log(value::Fixed<std::complex<double>, 3, 4>{})), 1.609437912434100374601, 1e-9));
  static_assert(value::internal::near(value::imag(value::log(value::Fixed<std::complex<double>, 3, 4>{})), 0.9272952180016122324285, 1e-9));
}


#include "values/math/log1p.hpp"

TEST(values, log1p)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::signbit(value::log1p(-0.)));
    EXPECT_FALSE(std::signbit(value::log1p(+0.)));
    EXPECT_EQ(value::log1p(-1), -value::internal::infinity<double>());
    EXPECT_EQ(value::log1p(+value::internal::infinity<double>()), +value::internal::infinity<double>());
    EXPECT_TRUE(value::log1p(-2) != value::log(-2)); // Nan
    EXPECT_TRUE(value::log1p(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::log1p(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::internal::near<10>(value::log1p(-0.), 0));
  static_assert(value::internal::near<10>(value::log1p(1.), numbers::ln2_v<double>));
  static_assert(value::internal::near<10>(value::log1p(9.), numbers::ln10_v<double>));
  static_assert(value::internal::near<10>(value::log1p(e - 1), 1));
  static_assert(value::internal::near<10>(value::log1p(e*e - 1), 2));
  static_assert(value::internal::near<10>(value::log1p(e*e*e - 1), 3));
  static_assert(value::internal::near<10>(value::log1p(e*e*e*e*e*e*e*e*e*e*e*e*e*e*e*e - 1), 16));
  static_assert(value::internal::near<10>(value::log1p(1/e - 1), -1));
  EXPECT_NEAR(value::log1p(5.0L), std::log1p(5.0L), 1e-9);
  EXPECT_NEAR(value::log1p(0.2L), std::log1p(0.2L), 1e-9);
  EXPECT_NEAR(value::log1p(5), std::log1p(5), 1e-9);
  EXPECT_NEAR(value::log1p(0.2), std::log1p(0.2), 1e-9);
  EXPECT_NEAR(value::log1p(20), std::log1p(20), 1e-9);
  EXPECT_NEAR(value::log1p(0.05), std::log1p(0.05), 1e-9);
  EXPECT_NEAR(value::log1p(100), std::log1p(100), 1e-9);
  EXPECT_NEAR(value::log1p(0.01), std::log1p(0.01), 1e-9);
  EXPECT_NEAR(value::log1p(0.001), std::log1p(0.001), 1e-9);
  EXPECT_NEAR(value::log1p(0.0001), std::log1p(0.0001), 1e-9);
  EXPECT_NEAR(value::log1p(0.00001), std::log1p(0.00001), 1e-9);
  EXPECT_NEAR(value::log1p(0.000001), std::log1p(0.000001), 1e-9);
  EXPECT_NEAR(value::log1p(1e-20), std::log1p(1e-20), 1e-9);
  EXPECT_NEAR(value::log1p(1e-200), std::log1p(1e-200), 1e-9);
  EXPECT_NEAR(value::log1p(1e-200L), std::log1p(1e-200L), 1e-9);
  EXPECT_NEAR(value::log1p(1e20), std::log1p(1e20), 1e-9);
  EXPECT_NEAR(value::log1p(1e200), std::log1p(1e200), 1e-9);
  EXPECT_NEAR(value::log1p(1e200L), std::log1p(1e200L), 1e-9);

  static_assert(value::internal::near(value::log1p(std::complex<double>{e*e - 1, 0}), 2));
  EXPECT_PRED3(tolerance, std::real(value::log1p(std::complex<double>{4e-21})), std::log1p(4e-21), 1e-30);
  EXPECT_PRED3(tolerance, value::log1p(std::complex<double>{-4}), std::log(std::complex<double>{-3}), 1e-9);
  EXPECT_PRED3(tolerance, value::log1p(std::complex<double>{3, 4}), std::log(std::complex<double>{4, 4}), 1e-9);
  EXPECT_PRED3(tolerance, value::log1p(std::complex<double>{3, -4}), std::log(std::complex<double>{4, -4}), 1e-9);
  EXPECT_PRED3(tolerance, value::log1p(std::complex<double>{-3, 4}), std::log(std::complex<double>{-2, 4}), 1e-9);
  EXPECT_PRED3(tolerance, value::log1p(std::complex<double>{-3, -4}), std::log(std::complex<double>{-2, -4}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::log1p(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(value::log1p(std::complex<int>{0, 0}) == std::complex<int>{0, 0}));
  COMPLEXINTEXISTS(static_assert(value::log1p(std::complex<int>{99, 0}) == std::complex<int>{4, 0}));
  COMPLEXINTEXISTS(static_assert(value::log1p(std::complex<int>{-101, 0}) == std::complex<int>{4, 3}));

  static_assert(value::internal::near(value::log1p(std::integral_constant<int, 1>{}), numbers::ln2_v<double>, 1e-9));
  static_assert(value::internal::near(value::log1p(value::Fixed<double, 1>{}), numbers::ln2_v<double>, 1e-9));
  static_assert(value::internal::near(value::real(value::log1p(value::Fixed<std::complex<double>, 2, 4>{})), 1.609437912434100374601, 1e-9));
  static_assert(value::internal::near(value::imag(value::log1p(value::Fixed<std::complex<double>, 2, 4>{})), 0.9272952180016122324285, 1e-9));
}


#include "values/math/asinh.hpp"

TEST(values, asinh)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(value::asinh(value::internal::NaN<double>()) != value::asinh(value::internal::NaN<double>()));
    EXPECT_TRUE(value::asinh(value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(value::asinh(-value::internal::infinity<double>()) == -value::internal::infinity<double>());
    EXPECT_FALSE(std::signbit(value::asinh(+0.)));
    EXPECT_TRUE(std::signbit(value::asinh(-0.)));
    EXPECT_TRUE(value::asinh(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::asinh(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::asinh(0) == 0);
  static_assert(value::internal::near(value::asinh((e - 1/e)/2), 1));
  static_assert(value::internal::near(value::asinh((e*e - 1/e/e)/2), 2));
  static_assert(value::internal::near(value::asinh((e*e*e - 1/e/e/e)/2), 3, 1e-9));
  static_assert(value::internal::near(value::asinh((1/e - e)/2), -1));
  static_assert(value::internal::near(value::asinh((1/e/e - e*e)/2), -2, 1e-9));
  static_assert(value::internal::near(value::asinh((1/e/e/e - e*e*e)/2), -3, 1e-9));
  EXPECT_NEAR(value::asinh(5), std::asinh(5), 1e-9);
  EXPECT_NEAR(value::asinh(-10), std::asinh(-10), 1e-9);

  static_assert(value::internal::near(value::asinh(std::complex<double>{(e*e - 1/e/e)/2, 0}), 2));
  EXPECT_PRED3(tolerance, value::asinh(std::complex<double>{3.3, -4.3}), std::asinh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::asinh(std::complex<double>{10.4, 3.4}), std::asinh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, value::asinh(std::complex<double>{-10.6, 10.6}), std::asinh(std::complex<double>{-10.6, 10.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::asinh(std::complex<int>{3, -4})));

  static_assert(value::internal::near(value::asinh(std::integral_constant<int, 2>{}), 1.443635475178810342493, 1e-9));
  static_assert(value::internal::near(value::asinh(value::Fixed<double, 2>{}), 1.443635475178810342493, 1e-9));
  static_assert(value::internal::near(value::real(value::asinh(value::Fixed<std::complex<double>, 3, 4>{})), 2.299914040879269649956, 1e-9));
  static_assert(value::internal::near(value::imag(value::asinh(value::Fixed<std::complex<double>, 3, 4>{})), 0.9176168533514786557599, 1e-9));
}


#include "values/math/acosh.hpp"

TEST(values, acosh)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(value::acosh(value::internal::NaN<double>()) != value::acosh(value::internal::NaN<double>()));
    EXPECT_TRUE(value::acosh(-1) != value::acosh(-1));
    EXPECT_TRUE(value::acosh(value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(value::acosh(-value::internal::infinity<double>()) != value::acosh(-value::internal::infinity<double>()));
    EXPECT_TRUE(value::acosh(0.9) != value::acosh(0.9));
    EXPECT_TRUE(value::acosh(-1) != value::acosh(-1));
    EXPECT_FALSE(std::signbit(value::acosh(1)));
    EXPECT_TRUE(value::acosh(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::acosh(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::acosh(1) == 0);
  static_assert(value::internal::near(value::acosh((e + 1/e)/2), 1));
  static_assert(value::internal::near(value::acosh((e*e + 1/e/e)/2), 2));
  static_assert(value::internal::near(value::acosh((e*e*e + 1/e/e/e)/2), 3, 1e-9));
  EXPECT_NEAR(value::acosh(5), std::acosh(5), 1e-9);
  EXPECT_NEAR(value::acosh(10), std::acosh(10), 1e-9);

  static_assert(value::internal::near(value::acosh(std::complex<double>{(e*e + 1/e/e)/2, 0}), 2));
  EXPECT_PRED3(tolerance, value::acosh(std::complex<double>{-2, 0}), std::acosh(std::complex<double>{-2, 0}), 1e-9);
  EXPECT_PRED3(tolerance, value::acosh(std::complex<double>{3.3, -4.3}), std::acosh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::acosh(std::complex<double>{5.4, 3.4}), std::acosh(std::complex<double>{5.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, value::acosh(std::complex<double>{-5.6, 5.6}), std::acosh(std::complex<double>{-5.6, 5.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::acosh(std::complex<int>{3, -4})));

  static_assert(value::internal::near(value::acosh(std::integral_constant<int, 2>{}), 1.316957896924816708625, 1e-9));
  static_assert(value::internal::near(value::acosh(value::Fixed<double, 2>{}), 1.316957896924816708625, 1e-9));
  static_assert(value::internal::near(value::real(value::acosh(value::Fixed<std::complex<double>, 3, 4>{})), 2.305509031243476942042, 1e-9));
  static_assert(value::internal::near(value::imag(value::acosh(value::Fixed<std::complex<double>, 3, 4>{})), 0.9368124611557199029125, 1e-9));
}


#include "values/math/atanh.hpp"

TEST(values, atanh)
{
  constexpr auto e = numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(value::atanh(value::internal::NaN<double>()) != value::atanh(value::internal::NaN<double>()));
    EXPECT_TRUE(value::atanh(2) != value::atanh(2));
    EXPECT_TRUE(value::atanh(-2) != value::atanh(-2));
    EXPECT_TRUE(value::atanh(1) == value::internal::infinity<double>());
    EXPECT_TRUE(value::atanh(-1) == -value::internal::infinity<double>());
    EXPECT_FALSE(std::signbit(value::atanh(+0.)));
    EXPECT_TRUE(std::signbit(value::atanh(-0.)));
    EXPECT_TRUE(value::atanh(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::atanh(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::atanh(0) == 0);
  static_assert(value::internal::near(value::atanh((e*e - 1)/(e*e + 1)), 1));
  static_assert(value::internal::near(value::atanh((e*e*e*e - 1)/(e*e*e*e + 1)), 2));
  static_assert(value::internal::near(value::atanh((e*e*e*e*e*e - 1)/(e*e*e*e*e*e + 1)), 3, 1e-9));
  static_assert(value::internal::near(value::atanh((1 - e*e)/(1 + e*e)), -1));
  static_assert(value::internal::near(value::atanh((1 - e*e*e*e)/(1 + e*e*e*e)), -2));
  static_assert(value::internal::near(value::atanh((1 - e*e*e*e*e*e)/(1 + e*e*e*e*e*e)), -3, 1e-9));
  EXPECT_NEAR(value::atanh(0.99), std::atanh(0.99), 1e-9);
  EXPECT_NEAR(value::atanh(-0.99), std::atanh(-0.99), 1e-9);

  static_assert(value::internal::near(value::atanh(std::complex<double>{(e*e*e*e - 1)/(e*e*e*e + 1), 0}), 2));
  EXPECT_PRED3(tolerance, value::atanh(std::complex<double>{3.3, -4.3}), std::atanh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::atanh(std::complex<double>{10.4, 3.4}), std::atanh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, value::atanh(std::complex<double>{-30.6, 20.6}), std::atanh(std::complex<double>{-30.6, 20.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::atanh(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(value::atanh(std::complex<int>{3, 4}) == std::complex<int>{0, 1}));
  COMPLEXINTEXISTS(static_assert(value::atanh(std::complex<int>{3, -4}) == std::complex<int>{0, -1}));
  COMPLEXINTEXISTS(static_assert(value::atanh(std::complex<int>{0, 3}) == std::complex<int>{0, 1}));
  COMPLEXINTEXISTS(static_assert(value::atanh(std::complex<int>{0, -7}) == std::complex<int>{0, -1}));

  static_assert(value::internal::near(value::atanh(std::integral_constant<int, 0>{}), 0, 1e-9));
  static_assert(value::internal::near(value::atanh(value::Fixed<double, 0>{}), 0, 1e-9));
  static_assert(value::internal::near(value::real(value::atanh(value::Fixed<std::complex<double>, 3, 4>{})), 0.1175009073114338884127, 1e-9));
  static_assert(value::internal::near(value::imag(value::atanh(value::Fixed<std::complex<double>, 3, 4>{})), 1.409921049596575522531, 1e-9));
}


#include "values/math/asin.hpp"

TEST(values, asin)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(value::asin(value::internal::NaN<double>()) != value::asin(value::internal::NaN<double>()));
    EXPECT_TRUE(value::asin(2.0) != value::asin(2.0));
    EXPECT_TRUE(value::asin(-2.0) != value::asin(-2.0));
    EXPECT_TRUE(std::signbit(value::asin(-0.)));
    EXPECT_TRUE(not std::signbit(value::asin(0.)));
    EXPECT_TRUE(value::asin(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::asin(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::asin(0) == 0);
  static_assert(value::asin(1) == pi/2);
  static_assert(value::asin(1.0L) == piL/2);
  static_assert(value::asin(1.0F) == piF/2);
  static_assert(value::asin(-1) == -pi/2);
  static_assert(value::internal::near(value::asin(numbers::sqrt2_v<double>/2), pi/4));
  static_assert(value::internal::near(value::asin(-numbers::sqrt2_v<double>/2), -pi/4));
  static_assert(value::asin(0.99995) > 0);
  static_assert(value::asin(-0.99995) < 0);
  EXPECT_NEAR(value::asin(numbers::sqrt2_v<double>/2), pi/4, 1e-9);
  EXPECT_NEAR(value::asin(-0.7), std::asin(-0.7), 1e-9);
  EXPECT_NEAR(value::asin(0.9), std::asin(0.9), 1e-9);
  EXPECT_NEAR(value::asin(0.99), std::asin(0.99), 1e-9);
  EXPECT_NEAR(value::asin(0.999), std::asin(0.999), 1e-9);
  EXPECT_NEAR(value::asin(-0.999), std::asin(-0.999), 1e-9);
  EXPECT_NEAR(value::asin(0.99999), std::asin(0.99999), 1e-9);
  EXPECT_NEAR(value::asin(0.99999999), std::asin(0.99999999), 1e-9);

  static_assert(value::internal::near(value::asin(std::complex<double>{numbers::sqrt2_v<double>/2, 0}), pi/4, 1e-9));
  EXPECT_PRED3(tolerance, value::asin(std::complex<double>{4.1, 3.1}), std::asin(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, value::asin(std::complex<double>{3.2, -4.2}), std::asin(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, value::asin(std::complex<double>{-3.3, 4.3}), std::asin(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::asin(std::complex<double>{-9.3, 10.3}), std::asin(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::asin(std::complex<int>{3, -4})));

  static_assert(value::internal::near(value::asin(std::integral_constant<int, 1>{}), pi/2, 1e-9));
  static_assert(value::internal::near(value::asin(value::Fixed<double, 1>{}), pi/2, 1e-9));
  static_assert(value::internal::near(value::real(value::asin(value::Fixed<std::complex<double>, 3, 4>{})), 0.6339838656391767163188, 1e-9));
  static_assert(value::internal::near(value::imag(value::asin(value::Fixed<std::complex<double>, 3, 4>{})), 2.305509031243476942042, 1e-9));
}


#include "values/math/acos.hpp"

TEST(values, acos)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(value::acos(value::internal::NaN<double>()) != value::acos(value::internal::NaN<double>()));
    EXPECT_TRUE(value::acos(-2) != value::acos(-2)); // NaN
    EXPECT_FALSE(std::signbit(value::cos(1)));
    EXPECT_TRUE(value::acos(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::acos(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::acos(0) == pi/2);
  static_assert(value::acos(1) == 0);
  static_assert(value::acos(-1) == pi);
  static_assert(value::acos(-1.0L) == piL);
  static_assert(value::acos(-1.0F) == piF);
  static_assert(value::internal::near(value::acos(0.5), numbers::pi/3));
  static_assert(value::internal::near(value::acos(-0.5), 2*numbers::pi/3));
  static_assert(value::internal::near(value::acos(numbers::sqrt2_v<double>/2), pi/4));
  static_assert(value::internal::near(value::acos(-numbers::sqrt2_v<double>/2), 3*pi/4));
  EXPECT_NEAR(value::acos(-0.7), std::acos(-0.7), 1e-9);
  EXPECT_NEAR(value::acos(0.9), std::acos(0.9), 1e-9);
  EXPECT_NEAR(value::acos(0.999), std::acos(0.999), 1e-9);
  EXPECT_NEAR(value::acos(-0.999), std::acos(-0.999), 1e-9);
  EXPECT_NEAR(value::acos(0.99999), std::acos(0.99999), 1e-9);
  EXPECT_NEAR(value::acos(0.9999999), std::acos(0.9999999), 1e-9);

  static_assert(value::internal::near(value::acos(std::complex<double>{0.5, 0}), pi/3, 1e-9));
  EXPECT_PRED3(tolerance, value::acos(std::complex<double>{4.1, 3.1}), std::acos(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, value::acos(std::complex<double>{3.2, -4.2}), std::acos(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, value::acos(std::complex<double>{-3.3, 4.3}), std::acos(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::acos(std::complex<double>{-9.3, 10.3}), std::acos(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::acos(std::complex<int>{3, -4})));

  static_assert(value::internal::near(value::acos(std::integral_constant<int, -1>{}), pi, 1e-9));
  static_assert(value::internal::near(value::acos(value::Fixed<double, -1>{}), pi, 1e-9));
  static_assert(value::internal::near(value::real(value::acos(value::Fixed<std::complex<double>, 3, 4>{})), 0.9368124611557199029125, 1e-9));
  static_assert(value::internal::near(value::imag(value::acos(value::Fixed<std::complex<double>, 3, 4>{})), -2.305509031243476942042, 1e-9));
}


#include "values/math/atan.hpp"

TEST(values, atan)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(value::atan(value::internal::NaN<double>()) != value::atan(value::internal::NaN<double>()));
    EXPECT_DOUBLE_EQ(value::atan(value::internal::infinity<double>()), pi/2);
    EXPECT_DOUBLE_EQ(value::atan(-value::internal::infinity<double>()), -pi/2);
    EXPECT_TRUE(std::signbit(value::atan(-0.)));
    EXPECT_FALSE(std::signbit(value::atan(+0.)));
    EXPECT_TRUE(value::atan(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::atan(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::atan(0) == 0);
  static_assert(value::internal::near(value::atan(1.), pi/4));
  static_assert(value::internal::near(value::atan(-1.), -pi/4));
  static_assert(value::internal::near(value::atan(-1.L), -piL/4));
  static_assert(value::internal::near(value::atan(-1.F), -piF/4));
  EXPECT_NEAR(value::atan(-0.7), std::atan(-0.7), 1e-9);
  EXPECT_NEAR(value::atan(0.9), std::atan(0.9), 1e-9);
  EXPECT_NEAR(value::atan(5.0), std::atan(5.0), 1e-9);
  EXPECT_NEAR(value::atan(-10.0), std::atan(-10.0), 1e-9);
  EXPECT_NEAR(value::atan(100.0), std::atan(100.0), 1e-9);

  static_assert(value::internal::near(value::atan(std::complex<double>{1, 0}), pi/4, 1e-9));
  EXPECT_PRED3(tolerance, value::atan(std::complex<double>{4.1, 0.}), std::atan(4.1), 1e-9);
  EXPECT_PRED3(tolerance, value::atan(std::complex<double>{4.1, 3.1}), std::atan(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, value::atan(std::complex<double>{3.2, -4.2}), std::atan(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, value::atan(std::complex<double>{-3.3, 4.3}), std::atan(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, value::atan(std::complex<double>{-9.3, 10.3}), std::atan(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::atan(std::complex<int>{3, -4})));

  static_assert(value::internal::near(value::atan(std::integral_constant<int, 1>{}), pi/4, 1e-9));
  static_assert(value::internal::near(value::atan(value::Fixed<double, 1>{}), pi/4, 1e-9));
  static_assert(value::internal::near(value::real(value::atan(value::Fixed<std::complex<double>, 3, 4>{})), 1.448306995231464542145, 1e-9));
  static_assert(value::internal::near(value::imag(value::atan(value::Fixed<std::complex<double>, 3, 4>{})), 0.1589971916799991743648, 1e-9));
}


#include "values/math/atan2.hpp"

TEST(values, atan2)
{
  constexpr auto pi = numbers::pi_v<double>;
  constexpr auto piL = numbers::pi_v<long double>;
  constexpr auto piF = numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_DOUBLE_EQ(value::atan2(value::internal::infinity<double>(), 0.f), pi/2);
    EXPECT_DOUBLE_EQ(value::atan2(-value::internal::infinity<double>(), 0.f), -pi/2);
    EXPECT_DOUBLE_EQ(value::atan2(+0., +value::internal::infinity<double>()), 0);
    EXPECT_DOUBLE_EQ(value::atan2(+0., -value::internal::infinity<double>()), pi);
    EXPECT_DOUBLE_EQ(value::atan2(-0., +value::internal::infinity<double>()), -0.);
    EXPECT_FALSE(std::signbit(value::atan2(+0., value::internal::infinity<double>())));
    EXPECT_DOUBLE_EQ(value::atan2(value::internal::infinity<double>(), value::internal::infinity<double>()), pi/4);
    EXPECT_DOUBLE_EQ(value::atan2(value::internal::infinity<double>(), -value::internal::infinity<double>()), 3*pi/4);
    EXPECT_DOUBLE_EQ(value::atan2(-value::internal::infinity<double>(), value::internal::infinity<double>()), -pi/4);
    EXPECT_DOUBLE_EQ(value::atan2(-value::internal::infinity<double>(), -value::internal::infinity<double>()), -3*pi/4);
#ifdef __cpp_lib_constexpr_cmath
    EXPECT_DOUBLE_EQ(value::atan2(-0., -value::internal::infinity<double>()), -pi);
    EXPECT_TRUE(std::signbit(value::atan2(-0., value::internal::infinity<double>())));
    static_assert(std::signbit(value::atan2(-0., +0.)));
    static_assert(not std::signbit(value::atan2(+0., +0.)));
    static_assert(value::atan2(-0., -0.) == -pi);
    static_assert(value::atan2(-0., -1.) == -pi);
    static_assert(value::atan2(+0., -0.) == pi);
    static_assert(value::atan2(+0., -1.) == pi);
#endif
    //EXPECT_TRUE(std::signbit(value::atan2(-0., +0.))); // This will be inacurate prior to c++23.
    EXPECT_FALSE(std::signbit(value::atan2(+0., +0.)));
    //EXPECT_EQ(value::atan2(-0., -0.), -pi); // This will be inacurate prior to c++23.
    //EXPECT_EQ(value::atan2(-0., -1.), -pi); // This will be inacurate prior to c++23.
    //EXPECT_EQ(value::atan2(+0., -0.), pi); // This will be inacurate prior to c++23.
    EXPECT_EQ(value::atan2(+0., -1.), pi);

    EXPECT_TRUE(value::atan2(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}, std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::atan2(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}, std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::atan2(0, 1) == 0);
  static_assert(value::atan2(0, -1) == pi);
  static_assert(value::atan2(1, 0) == pi/2);
  static_assert(value::atan2(-1, 0) == -pi/2);
  static_assert(value::internal::near(value::atan2(0.5, 0.5), pi/4));
  static_assert(value::internal::near(value::atan2(1., -1.), 3*pi/4));
  static_assert(value::internal::near(value::atan2(-0.5, 0.5), -pi/4));
  static_assert(value::internal::near(value::atan2(-1.L, -1.L), -3*piL/4));
  static_assert(value::internal::near(value::atan2(-1.F, -1.F), -3*piF/4));
  EXPECT_NEAR(value::atan2(-0.7, 4.5), std::atan2(-0.7, 4.5), 1e-9);
  EXPECT_NEAR(value::atan2(0.9, -2.3), std::atan2(0.9, -2.3), 1e-9);
  EXPECT_NEAR(value::atan2(5.0, 3.1), std::atan2(5.0, 3.1), 1e-9);
  EXPECT_NEAR(value::atan2(-10.0, 9.0), std::atan2(-10.0, 9.0), 1e-9);
  EXPECT_NEAR(value::atan2(100.0, 200.0), std::atan2(100.0, 200.0), 1e-9);

  static_assert(value::atan2(std::complex<double>{0, 0}, std::complex<double>{0, 0}) == 0.0);
  static_assert(value::atan2(std::complex<double>{0, 0}, std::complex<double>{1, 0}) == 0.0);
  static_assert(value::internal::near(value::atan2(std::complex<double>{0, 0}, std::complex<double>{-1, 0}), pi, 1e-9));
  static_assert(value::internal::near(value::atan2(std::complex<double>{1, 0}, std::complex<double>{0, 0}), pi/2, 1e-9));
  static_assert(value::internal::near(value::atan2(std::complex<double>{-1, 0}, std::complex<double>{0, 0}), -pi/2, 1e-9));
  static_assert(value::internal::near(value::real(value::atan2(std::complex<double>{3.2, -4.2}, std::complex<double>{-4.1, 3.1})), -0.7993578098204363309621, 1e-9));
  static_assert(value::internal::near(value::imag(value::atan2(std::complex<double>{3.2, -4.2}, std::complex<double>{-4.1, 3.1})), 0.1378262475816170392786, 1e-9));
  EXPECT_PRED3(tolerance, value::atan2(std::complex<double>{4.1, 3.1}, std::complex<double>{2.1, 5.1}), std::atan(std::complex<double>{4.1, 3.1} / std::complex<double>{2.1, 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, value::atan2(std::complex<double>{-3.3, 4.3}, std::complex<double>{2.1, 5.1}), std::atan(std::complex<double>{-3.3, 4.3} / std::complex<double>{2.1, 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, value::atan2(std::complex<double>{-9.3, 10.3}, std::complex<double>{-5.1, 2.1}), std::atan(std::complex<double>{-9.3, 10.3} / std::complex<double>{-5.1, 2.1}), 1e-9);
  EXPECT_PRED3(tolerance, value::atan2(std::complex<double>{0., 0.}, std::complex<double>{0., 0.}), std::complex<double>{0}, 1e-9);
  EXPECT_PRED3(tolerance, value::atan2(std::complex<double>{0., 3.1}, std::complex<double>{2.1, 5.1}), std::atan(std::complex<double>{0., 3.1} / std::complex<double>{2.1, 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, value::atan2(std::complex<double>{0., 3.1}, std::complex<double>{-2.1, 5.1}), std::atan(std::complex<double>{0., 3.1} / std::complex<double>{-2.1, 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, value::atan2(std::complex<double>{4.1, 3.1}, std::complex<double>{0., 5.1}), std::atan(std::complex<double>{4.1, 3.1} / std::complex<double>{0., 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, value::atan2(std::complex<double>{-4.1, 3.1}, std::complex<double>{0., 5.1}), std::atan(std::complex<double>{-4.1, 3.1} / std::complex<double>{0., 5.1}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::atan2(std::complex<int>{3, -4}, std::complex<int>{2, 5})));

  static_assert(value::internal::near(value::atan2(std::integral_constant<int, 1>{}, std::integral_constant<int, 0>{}), pi/2, 1e-9));
  static_assert(value::internal::near(value::atan2(value::Fixed<double, 1>{}, value::Fixed<double, 0>{}), pi/2, 1e-9));
  static_assert(value::internal::near(value::real(value::atan2(value::Fixed<std::complex<double>, 3, 4>{}, value::Fixed<std::complex<double>, 5, 2>{})), 0.7420289940594557537102, 1e-9));
  static_assert(value::internal::near(value::imag(value::atan2(value::Fixed<std::complex<double>, 3, 4>{}, value::Fixed<std::complex<double>, 5, 2>{})), 0.2871556773106927669533, 1e-9));
}


#include "values/math/pow.hpp"

TEST(values, pow)
{
  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_FALSE(std::signbit(value::pow(+0., 3U)));
    EXPECT_TRUE(std::signbit(value::pow(-0., 3U)));
    EXPECT_FALSE(std::signbit(value::pow(+0., 2U)));
    EXPECT_FALSE(std::signbit(value::pow(-0., 2U)));
    EXPECT_TRUE(value::pow(value::internal::NaN<double>(), 0U) == 1);
    EXPECT_TRUE(value::pow(value::internal::NaN<double>(), 1U) != value::pow(value::internal::NaN<double>(), 1U));
    EXPECT_TRUE(value::pow(-value::internal::infinity<double>(), 3U) == -value::internal::infinity<double>());
    EXPECT_TRUE(value::pow(-value::internal::infinity<double>(), 4U) == value::internal::infinity<double>());
    EXPECT_TRUE(value::pow(value::internal::infinity<double>(), 3U) == value::internal::infinity<double>());

    EXPECT_TRUE(value::pow(+0., -3) == value::internal::infinity<double>());
#ifdef __cpp_lib_constexpr_cmath
    EXPECT_TRUE(value::pow(-0., -3) == -value::internal::infinity<double>());
#endif
    EXPECT_TRUE(value::pow(+0., -2) == value::internal::infinity<double>());
    EXPECT_TRUE(value::pow(-0., -2) == value::internal::infinity<double>());
    EXPECT_FALSE(std::signbit(value::pow(+0., 3)));
    EXPECT_TRUE(std::signbit(value::pow(-0., 3)));
    EXPECT_FALSE(std::signbit(value::pow(+0., 2)));
    EXPECT_FALSE(std::signbit(value::pow(-0., 2)));
    EXPECT_TRUE(value::pow(value::internal::NaN<double>(), 0) == 1);
    EXPECT_TRUE(value::pow(value::internal::NaN<double>(), 1) != value::pow(value::internal::NaN<double>(), 1));
    EXPECT_TRUE(value::pow(-value::internal::infinity<double>(), -3) == 0);
    EXPECT_TRUE(std::signbit(value::pow(-value::internal::infinity<double>(), -3)));
    EXPECT_TRUE(value::pow(-value::internal::infinity<double>(), -2) == 0);
    EXPECT_FALSE(std::signbit(value::pow(-value::internal::infinity<double>(), -2)));
    EXPECT_TRUE(value::pow(-value::internal::infinity<double>(), 3) == -value::internal::infinity<double>());
    EXPECT_TRUE(value::pow(-value::internal::infinity<double>(), 4) == value::internal::infinity<double>());
    EXPECT_TRUE(value::pow(value::internal::infinity<double>(), -3) == 0);
    EXPECT_FALSE(std::signbit(value::pow(value::internal::infinity<double>(), -3)));
    EXPECT_TRUE(value::pow(value::internal::infinity<double>(), 3) == value::internal::infinity<double>());

    EXPECT_TRUE(value::pow(+0., -value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(value::pow(-0., -value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(value::pow(+0.5, -value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(value::pow(-0.5, -value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(value::pow(+1.5, -value::internal::infinity<double>()) == 0);
    EXPECT_TRUE(value::pow(-1.5, -value::internal::infinity<double>()) == 0);
    EXPECT_FALSE(std::signbit(value::pow(+1.5, -value::internal::infinity<double>())));
    EXPECT_FALSE(std::signbit(value::pow(-1.5, -value::internal::infinity<double>())));
    EXPECT_TRUE(value::pow(+0.5, value::internal::infinity<double>()) == 0);
    EXPECT_TRUE(value::pow(-0.5, value::internal::infinity<double>()) == 0);
    EXPECT_FALSE(std::signbit(value::pow(+0.5, value::internal::infinity<double>())));
    EXPECT_FALSE(std::signbit(value::pow(-0.5, value::internal::infinity<double>())));
    EXPECT_TRUE(value::pow(+1.5, value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(value::pow(-1.5, value::internal::infinity<double>()) == value::internal::infinity<double>());
    EXPECT_TRUE(value::pow(-value::internal::infinity<double>(), -3.) == 0);

    EXPECT_TRUE(value::pow(value::internal::infinity<double>(), -3.) == 0);
    EXPECT_FALSE(std::signbit(value::pow(-value::internal::infinity<double>(), -3.)));
    EXPECT_EQ(value::pow(-value::internal::infinity<double>(), 3.), value::internal::infinity<double>());
    EXPECT_FALSE(std::signbit(value::pow(value::internal::infinity<double>(), -3.)));
    EXPECT_TRUE(value::pow(value::internal::infinity<double>(), 3.) == value::internal::infinity<double>());

    EXPECT_FALSE(std::signbit(value::pow(+0, 3.)));
    EXPECT_FALSE(std::signbit(value::pow(-0, 3.)));
    EXPECT_TRUE(value::pow(-1., value::internal::infinity<double>()) == 1);
    EXPECT_TRUE(value::pow(-1., -value::internal::infinity<double>()) == 1);
    EXPECT_TRUE(value::pow(+1., value::internal::NaN<double>()) == 1);
    EXPECT_TRUE(value::pow(value::internal::NaN<double>(), +0) == 1);
    EXPECT_TRUE(value::pow(value::internal::NaN<double>(), 1.) != value::pow(value::internal::NaN<double>(), 1.));

    EXPECT_TRUE(value::pow(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}, std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}) != value::pow(std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}, std::complex<double>{value::internal::NaN<double>(), value::internal::NaN<double>()}));
  }

  static_assert(value::pow(+0., 3U) == 0);
  static_assert(value::pow(-0., 3U) == 0);
  static_assert(value::pow(+0., 2U) == 0);
  static_assert(value::pow(-0., 2U) == 0);
  static_assert(value::pow(1, 0U) == 1);
  static_assert(value::pow(0, 1U) == 0);
  static_assert(value::pow(1, 1U) == 1);
  static_assert(value::pow(1, 2U) == 1);
  static_assert(value::pow(2, 1U) == 2);
  static_assert(value::pow(2, 5U) == 32);
  static_assert(value::pow(2, 16U) == 65536);
  static_assert(std::is_floating_point_v<decltype(value::pow(2, 16U))>);
  static_assert(value::pow(2.0, 16U) == 65536);
  static_assert(std::is_floating_point_v<decltype(value::pow(2.0, 16U))>);

  static_assert(value::pow(+0., 3) == 0);
  static_assert(value::pow(-0., 3) == 0);
  static_assert(value::pow(+0., 2) == 0);
  static_assert(value::pow(-0., 2) == 0);
  static_assert(value::pow(2, -4) == 0.0625);
  static_assert(value::pow(2, -5) == 0.03125);
  static_assert(std::is_floating_point_v<decltype(value::pow(2, -4))>);

  static_assert(value::pow(+0., 3.) == +0);
  static_assert(value::pow(-0., 3.) == +0);
  static_assert(value::pow(+1., 5) == 1);
  static_assert(value::pow(-5., +0) == 1);
  EXPECT_TRUE(value::pow(-5., 1.5) != value::pow(-5., 1.5));
  EXPECT_TRUE(value::pow(-7.3, 3.3) != value::pow(-7.3, 3.3));
  static_assert(value::internal::near(value::pow(2, -4.), 0.0625));
  static_assert(value::internal::near(value::pow(10, -4.), 1e-4));
  static_assert(value::internal::near(value::pow(10., 6.), 1e6, 1e-4));
  EXPECT_DOUBLE_EQ(value::pow(5.0L, 4.0L), std::pow(5.0L, 4.0L));
  EXPECT_DOUBLE_EQ(value::pow(5.0L, -4.0L), std::pow(5.0L, -4.0L));
  EXPECT_DOUBLE_EQ(value::pow(1e20L, 2.L), std::pow(1e20L, 2.L));
  EXPECT_DOUBLE_EQ(value::pow(1e20L, -2.L), std::pow(1e20L, -2.L));
  EXPECT_DOUBLE_EQ(value::pow(1e100L, 2.L), std::pow(1e100L, 2.L));
  EXPECT_DOUBLE_EQ(value::pow(1e100L, -2.L), std::pow(1e100L, -2.L));

  EXPECT_PRED3(tolerance, value::pow(2., std::complex<double>{-4}), std::pow(2., std::complex<double>{-4}), 1e-9);
  EXPECT_PRED3(tolerance, value::pow(2, std::complex<double>{-4}), std::pow(2, std::complex<double>{-4}), 1e-9);
  EXPECT_PRED3(tolerance, value::pow(std::complex<double>{3, 4}, 2.), std::pow(std::complex<double>{3, 4}, 2.), 1e-9);
  EXPECT_PRED3(tolerance, value::pow(std::complex<double>{3, 4}, 2), std::pow(std::complex<double>{3, 4}, 2), 1e-9);
  EXPECT_PRED3(tolerance, value::pow(std::complex<double>{3, 4}, -2), std::pow(std::complex<double>{3, 4}, -2), 1e-9);
  EXPECT_PRED3(tolerance, value::pow(std::complex<double>{3, 4}, 3), std::pow(std::complex<double>{3, 4}, 3), 1e-9);
  EXPECT_PRED3(tolerance, value::pow(std::complex<double>{3, 4}, -3), std::pow(std::complex<double>{3, 4}, -3), 1e-9);
  EXPECT_PRED3(tolerance, value::pow(2., std::complex<double>{3, -4}), std::pow(2., std::complex<double>{3, -4}), 1e-9);
  EXPECT_PRED3(tolerance, value::pow(std::complex<double>{1, 2}, std::complex<double>{-3, 4}), std::pow(std::complex<double>{1, 2}, std::complex<double>{-3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, value::pow(std::complex<double>{-3, -4}, std::complex<double>{1, 2}), std::pow(std::complex<double>{-3, -4}, std::complex<double>{1, 2}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(value::pow(std::complex<int>{-3, -4}, std::complex<int>{1, 2})));

  static_assert(value::pow(value::Fixed<double, 2>{}, std::integral_constant<int, 3>{}) == 8);
  static_assert(value::pow(value::Fixed<double, 2>{}, 3) == 8);
  static_assert(value::internal::near(value::pow(2, value::Fixed<double, 3>{}), 8, 1e-6));
}
