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
  auto tolerance = [](const auto& a, const auto& b, const auto& err){ return values::internal::near(a, b, err); };
}


#include "values/math/internal/infinity.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/isinf.hpp"
#include "values/math/isnan.hpp"

TEST(values, infinity_nan)
{
  if (std::numeric_limits<double>::has_infinity)
  {
    static_assert(values::isinf(values::internal::infinity<double>()));
    static_assert(values::isinf(INFINITY));
    static_assert(values::isinf(-INFINITY));
  }
  if (std::numeric_limits<double>::has_quiet_NaN or std::numeric_limits<double>::has_signaling_NaN)
  {
    static_assert(values::isnan(values::internal::NaN<double>()));
    static_assert(values::isnan(NAN));
    static_assert(values::isnan(-NAN));
  }
}


#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/math/conj.hpp"
#include "values/concepts/integral.hpp"

TEST(values, real_imag_conj)
{
  static_assert(std::is_floating_point_v<decltype(values::real(3))>);
  static_assert(std::is_floating_point_v<decltype(values::imag(3))>);
  static_assert(std::is_floating_point_v<decltype(values::real(values::conj(3)))>);
  static_assert(std::is_floating_point_v<decltype(values::real(std::complex<double>{3., 4.}))>);
  COMPLEXINTEXISTS(static_assert(values::integral<decltype(values::real(std::complex<int>{3, 4}))>));
  COMPLEXINTEXISTS(static_assert(values::integral<decltype(values::imag(std::complex<int>{3, 4}))>));
  COMPLEXINTEXISTS(static_assert(values::integral<decltype(values::real(values::conj(std::complex<int>{3, 4})))>));
  static_assert(values::real(3.) == 3);
  static_assert(values::real(3.f) == 3);
  static_assert(values::real(3.l) == 3);
  static_assert(values::imag(3.) == 0);
  static_assert(values::imag(3.f) == 0);
  static_assert(values::imag(3.l) == 0);
  static_assert(values::conj(3.) == 3.);
  static_assert(values::conj(3.f) == 3.f);
  static_assert(values::conj(3.l) == 3.l);
  EXPECT_EQ(values::real(std::complex<double>{3, 4}), 3);
  EXPECT_EQ(values::imag(std::complex<double>{3, 4}), 4);
  EXPECT_TRUE((values::conj(std::complex<double>{3, 4}) == std::complex<double>{3, -4}));

  static_assert(values::real(std::integral_constant<int, 9>{}) == 9);
  static_assert(values::real(values::Fixed<std::complex<double>, 3, 4>{}) == 3);
  static_assert(values::imag(std::integral_constant<int, 9>{}) == 0);
  static_assert(values::imag(values::Fixed<std::complex<double>, 3, 4>{}) == 4);
  static_assert(values::real(values::conj(std::integral_constant<int, 9>{})) == 9);
  static_assert(values::imag(values::conj(std::integral_constant<int, 9>{})) == 0);
  static_assert(values::real(values::conj(values::Fixed<std::complex<double>, 3, 4>{})) == 3);
  static_assert(values::imag(values::conj(values::Fixed<std::complex<double>, 3, 4>{})) == -4);
  static_assert(values::fixed_number_of_v<decltype(values::real(values::conj(values::Fixed<std::complex<double>, 3, 4>{})))> == 3);
  static_assert(values::fixed_number_of_v<decltype(values::imag(values::conj(values::Fixed<std::complex<double>, 3, 4>{})))> == -4);
}


#include "values/math/signbit.hpp"

TEST(values, signbit)
{
  static_assert(not values::signbit(0));
  static_assert(values::signbit(-3));
  static_assert(not values::signbit(3));
  static_assert(not values::signbit(3.));
  static_assert(values::signbit(-3.));
  static_assert(not values::signbit(3.));
  static_assert(values::signbit(-3.f));
  static_assert(not values::signbit(3.f));
  static_assert(values::signbit(-3.l));
  static_assert(not values::signbit(3.l));
  static_assert(not values::signbit(INFINITY));
  static_assert(values::signbit(-INFINITY));
#ifdef __cpp_lib_constexpr_cmath
  static_assert(not values::signbit(0.));
  static_assert(values::signbit(+0.) == std::signbit(+0.));
  static_assert(values::signbit(-0.) == std::signbit(-0.));
  static_assert(values::signbit(NAN) == std::signbit(NAN));
  static_assert(values::signbit(-NAN) == std::signbit(-NAN));
#endif
  EXPECT_EQ(values::signbit(+0.), std::signbit(+0.));
  EXPECT_EQ(values::signbit(NAN), std::signbit(NAN));
  EXPECT_EQ(values::signbit(INFINITY), std::signbit(INFINITY));
  EXPECT_EQ(values::signbit(-INFINITY), std::signbit(-INFINITY));

  struct Op { constexpr auto operator()(const int& a) const { return values::signbit(a); } };
  static_assert(values::fixed_number_of_v<decltype(values::operation(Op{}, std::integral_constant<int, -3>{}))>);

  static_assert(values::fixed_number_of_v<decltype(values::signbit(std::integral_constant<int, -3>{}))>);
  static_assert(values::fixed_number_of_v<decltype(values::signbit(values::Fixed<double, -3>{}))>);
  static_assert(not values::fixed_number_of_v<decltype(values::signbit(values::Fixed<double, 3>{}))>);
}


#include "values/math/copysign.hpp"

TEST(values, copysign)
{
  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    static_assert(values::copysign(3.f, -INFINITY) == -3.f);
    static_assert(values::copysign(3.f, INFINITY) == 3.f);
    static_assert(values::copysign(INFINITY, -3.f) == -INFINITY);
    static_assert(values::copysign(-INFINITY, 3.f) == INFINITY);

    static_assert(values::copysign(0., -1.) == 0.);
    static_assert(values::copysign(0., 1.) == 0.);
    static_assert(values::copysign(-0., -1.) == -0.);
    static_assert(values::copysign(-0., 1.) == -0.);

#ifdef __cpp_lib_constexpr_cmath
    static_assert(std::signbit(values::copysign(-1., 0.)) == std::signbit(std::copysign(-1., 0.)));
    static_assert(std::signbit(values::copysign(1., 0.)) == std::signbit(std::copysign(1., 0.)));
    static_assert(std::signbit(values::copysign(-1., -0.)) == std::signbit(std::copysign(-1., -0.)));
    static_assert(std::signbit(values::copysign(1., -0.)) == std::signbit(std::copysign(1., -0.)));

    static_assert(std::signbit(values::copysign(0., -1.)) == std::signbit(std::copysign(0., -1.)));
    static_assert(std::signbit(values::copysign(0., 1.)) == std::signbit(std::copysign(0., 1.)));
    static_assert(std::signbit(values::copysign(-0., -1.)) == std::signbit(std::copysign(-0., -1.)));
    static_assert(std::signbit(values::copysign(-0., 1.)) == std::signbit(std::copysign(-0., 1.)));
#else
    EXPECT_EQ(std::signbit(values::copysign(-1., 0.)), std::signbit(std::copysign(-1., 0.)));
    EXPECT_EQ(std::signbit(values::copysign(1., 0.)), std::signbit(std::copysign(1., 0.)));

    EXPECT_EQ(std::signbit(values::copysign(0., -1.)), std::signbit(std::copysign(0., -1.)));
    EXPECT_EQ(std::signbit(values::copysign(0., 1.)), std::signbit(std::copysign(0., 1.)));
    EXPECT_EQ(std::signbit(values::copysign(-0., -1.)), std::signbit(std::copysign(-0., -1.)));
    EXPECT_EQ(std::signbit(values::copysign(-0., 1.)), std::signbit(std::copysign(-0., 1.)));
#endif

    auto NaN = std::numeric_limits<double>::quiet_NaN();

    EXPECT_TRUE(std::isnan(values::copysign(NaN, -1.)));
    EXPECT_TRUE(std::isnan(values::copysign(NaN, 1.)));
    EXPECT_TRUE(std::isnan(values::copysign(-NaN, -1.)));
    EXPECT_TRUE(std::isnan(values::copysign(-NaN, 1.)));

    EXPECT_TRUE(std::signbit(values::copysign(NaN, -1.)));
    EXPECT_FALSE(std::signbit(values::copysign(NaN, 1.)));
    EXPECT_TRUE(std::signbit(values::copysign(-NaN, -1.)));
    EXPECT_FALSE(std::signbit(values::copysign(-NaN, 1.)));
  }

  static_assert(values::copysign(3., -5.) == -3.);
  static_assert(values::copysign(-3., 5.) == 3.);
  static_assert(values::copysign(-3.f, 5.f) == 3.f);
  static_assert(values::copysign(3.l, -5.l) == -3.l);

  static_assert(values::copysign(5U, 3U) == 5.);
  static_assert(std::is_floating_point_v<decltype(values::copysign(5U, 3U))>);
  static_assert(values::copysign(5, -3) == -5.);
  static_assert(std::is_floating_point_v<decltype(values::copysign(5, -3))>);
  static_assert(values::copysign(5, 3) == 5.);
  static_assert(values::copysign(5U, 3) == 5.);
  static_assert(values::copysign(5U, -3) == -5.);
  static_assert(std::is_floating_point_v<decltype(values::copysign(5U, -3))>);
  static_assert(values::copysign(5, 3U) == 5.);
  static_assert(values::copysign(-5, 3U) == 5.);

  static_assert(values::fixed_number_of_v<decltype(values::copysign(values::Fixed<int, 5>{}, values::Fixed<int, -3>{}))> == -5);
  static_assert(values::fixed_number_of_v<decltype(values::copysign(values::Fixed<int, -5>{}, values::Fixed<int, 3>{}))> == 5);
  static_assert(values::fixed_number_of_v<decltype(values::copysign(values::Fixed<int, 5>{}, values::Fixed<int, -3>{}))> == -5);
  static_assert(values::fixed_number_of_v<decltype(values::copysign(values::Fixed<int, -5>{}, values::Fixed<int, 3>{}))> == 5);
}


#include "values/math/fmod.hpp"

TEST(values, fmod)
{
  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    constexpr auto inf = values::internal::infinity<double>();
    constexpr auto NaN = values::internal::NaN<double>();
    EXPECT_TRUE(std::signbit(std::fmod(-0., 1.)));
    EXPECT_FALSE(std::signbit(std::fmod(+0., 1.)));
    EXPECT_TRUE(std::isnan(std::fmod(+inf, 1.)));
    EXPECT_TRUE(std::isnan(std::fmod(-inf, 1.)));
    EXPECT_TRUE(std::isnan(std::fmod(1., +0.)));
    EXPECT_TRUE(std::isnan(std::fmod(1., -0.)));
    EXPECT_EQ(std::fmod(3., +inf), 3.);
    EXPECT_EQ(std::fmod(3., -inf), 3.);
    EXPECT_TRUE(std::isnan(std::fmod(NaN, 1.)));
    EXPECT_TRUE(std::isnan(std::fmod(1., NaN)));
    EXPECT_TRUE(std::isnan(std::fmod(NaN, NaN)));
  }
  static_assert(values::fmod(0., 5.) == 0.);
  static_assert(values::fmod(12., 5.) == 2.);
  static_assert(values::fmod(16., 3.) == 1.);
  static_assert(values::fmod(-12., 5.) == -2.);
  static_assert(values::fmod(-16., 3.) == -1.);
  static_assert(values::fmod(12., -5.) == 2.);
  static_assert(values::fmod(16., -3.) == 1.);
  static_assert(values::fmod(-12., -5.) == -2.);
  static_assert(values::fmod(-16., -3.) == -1.);
#ifndef __cpp_lib_constexpr_cmath
  EXPECT_ANY_THROW(values::fmod(1.e300, 7.));
#endif

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::fmod(values::Fixed<double, -12>{}, values::Fixed<double, 5>{}))>, -2, 1e-6));

}


#include "values/math/sqrt.hpp"

TEST(values, sqrt)
{
  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    constexpr auto inf = values::internal::infinity<double>();
    constexpr auto NaN = values::internal::NaN<double>();

    EXPECT_TRUE(std::signbit(values::sqrt(-0.)));
    EXPECT_FALSE(std::signbit(values::sqrt(+0.)));
    EXPECT_TRUE(std::isinf(values::sqrt(inf)));
    EXPECT_EQ(std::isinf(values::sqrt(inf)), std::isinf(std::sqrt(inf)));
    EXPECT_EQ(std::isnan(values::sqrt(-1)), std::isnan(std::sqrt(-1)));
    EXPECT_TRUE(std::isnan(values::sqrt(NaN)));
    EXPECT_EQ(std::isnan(values::sqrt(NaN)), std::isnan(std::sqrt(NaN)));

    EXPECT_FALSE(std::signbit(std::real(values::sqrt(std::complex<double>{+0.0, +0.0}))));
    EXPECT_FALSE(std::signbit(std::imag(values::sqrt(std::complex<double>{+0.0, +0.0}))));
    EXPECT_FALSE(std::signbit(std::real(values::sqrt(std::complex<double>{+0.0, -0.0}))));
    EXPECT_FALSE(std::signbit(std::real(values::sqrt(std::complex<double>{-0.0, +0.0}))));
    EXPECT_FALSE(std::signbit(std::imag(values::sqrt(std::complex<double>{-0.0, +0.0}))));
    EXPECT_FALSE(std::signbit(std::real(values::sqrt(std::complex<double>{-0.0, -0.0}))));
#ifdef __cpp_lib_constexpr_cmath
    EXPECT_TRUE(std::signbit(std::imag(values::sqrt(std::complex<double>{+0.0, -0.0}))));
    EXPECT_TRUE(std::signbit(std::imag(values::sqrt(std::complex<double>{-0.0, -0.0}))));
#endif

    static_assert(values::sqrt(std::complex<double>{+inf, +inf}) == std::complex<double>(+inf, +inf));
    EXPECT_TRUE((values::sqrt(std::complex<double>{+inf, +inf}) == std::sqrt(std::complex<double>{+inf, +inf})));
    static_assert(values::sqrt(std::complex<double>{+inf, -inf}) == std::complex<double>(+inf, -inf));
    EXPECT_TRUE((values::sqrt(std::complex<double>{+inf, -inf}) == std::sqrt(std::complex<double>{+inf, -inf})));
    static_assert(values::sqrt(std::complex<double>{-inf, +inf}) == std::complex<double>(+inf, +inf));
    EXPECT_TRUE((values::sqrt(std::complex<double>{-inf, +inf}) == std::sqrt(std::complex<double>{-inf, +inf})));
    static_assert(values::sqrt(std::complex<double>{-inf, -inf}) == std::complex<double>(+inf, -inf));
    EXPECT_TRUE((values::sqrt(std::complex<double>{-inf, -inf}) == std::sqrt(std::complex<double>{-inf, -inf})));

    static_assert(values::sqrt(std::complex<double>{1, +inf}) == std::complex<double>(inf, inf));
    EXPECT_TRUE((values::sqrt(std::complex<double>{1, +inf}) == std::sqrt(std::complex<double>{1, +inf})));
    static_assert(values::sqrt(std::complex<double>{1, -inf}) == std::complex<double>(inf, -inf));
    EXPECT_TRUE((values::sqrt(std::complex<double>{1, -inf}) == std::sqrt(std::complex<double>{1, -inf})));
    static_assert(values::sqrt(std::complex<double>{-1, +inf}) == std::complex<double>(inf, inf));
    EXPECT_TRUE((values::sqrt(std::complex<double>{-1, +inf}) == std::sqrt(std::complex<double>{-1, +inf})));
    static_assert(values::sqrt(std::complex<double>{-1, -inf}) == std::complex<double>(inf, -inf));
    EXPECT_TRUE((values::sqrt(std::complex<double>{-1, -inf}) == std::sqrt(std::complex<double>{-1, -inf})));

    static_assert(values::sqrt(std::complex<double>{ +inf, 1}) == std::complex<double>(inf, 0));
    EXPECT_TRUE((values::sqrt(std::complex<double>{ +inf, 1}) == std::sqrt(std::complex<double>{+inf, 1})));
    static_assert(values::sqrt(std::complex<double>{ -inf, 1}) == std::complex<double>(0, inf));
    EXPECT_TRUE((values::sqrt(std::complex<double>{ -inf, 1}) == std::sqrt(std::complex<double>{-inf, 1})));
    static_assert(values::sqrt(std::complex<double>{ +inf, -1}) == std::complex<double>(inf, -0.));
    EXPECT_TRUE((values::sqrt(std::complex<double>{ +inf, -1}) == std::sqrt(std::complex<double>{+inf, -1})));
    EXPECT_EQ(std::signbit(std::imag(values::sqrt(std::complex<double>{ +inf, -1}))), std::signbit(std::imag(std::sqrt(std::complex<double>{ +inf, -1}))));
    static_assert(values::sqrt(std::complex<double>{ -inf, -1}) == std::complex<double>(0, -inf));
    EXPECT_TRUE((values::sqrt(std::complex<double>{ -inf, -1}) == std::sqrt(std::complex<double>{-inf, -1})));

    static_assert(std::real(values::sqrt(std::complex<double>{ NaN, +inf})) == inf);
    EXPECT_DOUBLE_EQ(std::real(values::sqrt(std::complex<double>{ NaN, +inf})), std::real(std::sqrt(std::complex<double>{NaN, +inf})));
    static_assert(std::imag(values::sqrt(std::complex<double>{ NaN, +inf})) == inf);
    EXPECT_DOUBLE_EQ(std::imag(values::sqrt(std::complex<double>{ NaN, +inf})), std::imag(std::sqrt(std::complex<double>{NaN, +inf})));
    static_assert(std::real(values::sqrt(std::complex<double>{ NaN, -inf})) == inf);
    EXPECT_DOUBLE_EQ(std::real(values::sqrt(std::complex<double>{ NaN, -inf})), std::real(std::sqrt(std::complex<double>{NaN, -inf})));
    static_assert(std::imag(values::sqrt(std::complex<double>{ NaN, -inf})) == -inf);
    EXPECT_DOUBLE_EQ(std::imag(values::sqrt(std::complex<double>{ NaN, -inf})), std::imag(std::sqrt(std::complex<double>{NaN, -inf})));
    static_assert(std::real(values::sqrt(std::complex<double>{ +inf, NaN})) == inf);
    EXPECT_DOUBLE_EQ(std::real(values::sqrt(std::complex<double>{ +inf, NaN})), std::real(std::sqrt(std::complex<double>{+inf, NaN})));
    EXPECT_EQ(std::isnan(std::imag(values::sqrt(std::complex<double>{ +inf, NaN}))), std::isnan(std::imag(std::sqrt(std::complex<double>{ +inf, NaN}))));
    EXPECT_EQ(std::isnan(std::real(values::sqrt(std::complex<double>{ -inf, NaN}))), std::isnan(std::real(std::sqrt(std::complex<double>{ -inf, NaN}))));
    static_assert(std::imag(values::sqrt(std::complex<double>{ -inf, NaN})) == inf);
    EXPECT_DOUBLE_EQ(std::imag(values::sqrt(std::complex<double>{ -inf, NaN})), std::imag(std::sqrt(std::complex<double>{-inf, NaN})));

    EXPECT_TRUE(std::isnan(std::real(values::sqrt(std::complex<double>{1, NaN}))));
    EXPECT_TRUE(std::isnan(std::real(std::sqrt(std::complex<double>{1, NaN}))));
    EXPECT_TRUE(std::isnan(std::imag(values::sqrt(std::complex<double>{1, NaN}))));
    EXPECT_TRUE(std::isnan(std::imag(std::sqrt(std::complex<double>{1, NaN}))));
    EXPECT_TRUE(std::isnan(std::real(values::sqrt(std::complex<double>{NaN, 1}))));
    EXPECT_TRUE(std::isnan(std::real(std::sqrt(std::complex<double>{NaN, 1}))));
    EXPECT_TRUE(std::isnan(std::imag(values::sqrt(std::complex<double>{NaN, 1}))));
    EXPECT_TRUE(std::isnan(std::imag(std::sqrt(std::complex<double>{NaN, 1}))));

    EXPECT_TRUE(std::isnan(std::real(values::sqrt(std::complex<double>{NaN, NaN}))));
    EXPECT_TRUE(std::isnan(std::imag(values::sqrt(std::complex<double>{NaN, NaN}))));
    EXPECT_TRUE(std::isnan(std::real(values::sqrt(std::complex<double>{1, NaN}))));
    EXPECT_TRUE(std::isnan(std::imag(values::sqrt(std::complex<double>{1, NaN}))));
    EXPECT_TRUE(std::isnan(std::real(values::sqrt(std::complex<double>{NaN, 1}))));
    EXPECT_TRUE(std::isnan(std::imag(values::sqrt(std::complex<double>{NaN, 1}))));
  }

  static_assert(values::sqrt(0) == 0);
  static_assert(values::sqrt(1) == 1);
  static_assert(values::sqrt(4) == 2);
  static_assert(values::sqrt(9) == 3);
  static_assert(values::sqrt(1000000) == 1000);
  static_assert(values::internal::near(values::sqrt(2.), stdcompat::numbers::sqrt2));
  static_assert(values::internal::near(values::sqrt(3.), stdcompat::numbers::sqrt3));
  static_assert(values::internal::near(values::sqrt(4.0e6), 2.0e3));
  static_assert(values::internal::near(values::sqrt(9.0e-2), 3.0e-1));
  static_assert(values::internal::near(values::sqrt(2.5e-11), 5.0e-6));
  EXPECT_NEAR(values::sqrt(5.), std::sqrt(5), 1e-9);
  EXPECT_NEAR(values::sqrt(1.0e20), std::sqrt(1.0e20), 1e-9);
  EXPECT_NEAR(values::sqrt(0.001), std::sqrt(0.001), 1e-9);
  EXPECT_NEAR(values::sqrt(1e-20), std::sqrt(1e-20), 1e-9);

  EXPECT_PRED3(tolerance, values::sqrt(std::complex<double>{-4}), std::sqrt(std::complex<double>{-4}), 1e-9);
  EXPECT_PRED3(tolerance, values::sqrt(std::complex<double>{3, 4}), std::sqrt(std::complex<double>{3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, values::sqrt(std::complex<double>{3, -4}), (std::complex<double>{2, -1}), 1e-9);
  EXPECT_PRED3(tolerance, values::sqrt(std::complex<double>{3, 4}), (std::complex<double>{2, 1}), 1e-9);
  EXPECT_PRED3(tolerance, values::sqrt(std::complex<double>{3, -4}), std::sqrt(std::complex<double>{3, -4}), 1e-9);
  EXPECT_PRED3(tolerance, values::sqrt(std::complex<double>{-3, 4}), std::sqrt(std::complex<double>{-3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, values::sqrt(std::complex<double>{-3, -4}), std::sqrt(std::complex<double>{-3, -4}), 1e-9);
  EXPECT_PRED3(tolerance, values::sqrt(std::complex<double>{-3e10, 4e10}), std::sqrt(std::complex<double>{-3e10, 4e10}), 1e-9);

  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::sqrt(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::sqrt(std::complex<int>{3, 4})));
  COMPLEXINTEXISTS(static_assert(std::is_same_v<decltype(values::sqrt(std::complex<int>{3, -4})), std::complex<int>>));
  COMPLEXINTEXISTS(static_assert(std::is_same_v<decltype(values::sqrt(std::complex<int>{4, -4})), std::complex<int>>));
  COMPLEXINTEXISTS(static_assert(values::sqrt(std::complex<int>{3, 4}) == std::complex<int>{2, 1}));
  COMPLEXINTEXISTS(static_assert(values::sqrt(std::complex<int>{10, 15}) == std::complex<int>{3, 2}));

  static_assert(values::sqrt(std::integral_constant<int, 9>{}) == 3);
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::sqrt(values::Fixed<double, 9>{}))>, 3, 1e-6));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::sqrt(values::Fixed<double, 2>{}))>, stdcompat::numbers::sqrt2, 1e-6));
}


#include "values/math/hypot.hpp"

TEST(values, hypot)
{
  constexpr auto inf = values::internal::infinity<double>();
  constexpr auto NaN = values::internal::NaN<double>();

  static_assert(values::hypot(3) == 3);
  static_assert(values::hypot(-3) == 3);
  static_assert(values::hypot(3.) == 3);
  static_assert(values::hypot(-3.) == 3);
  static_assert(values::hypot(3, 4) == 5);
  static_assert(values::hypot(4, -3) == 5);
  static_assert(values::internal::near(values::hypot(2, 10, 11), 15, 1e-6));
  static_assert(values::internal::near(values::hypot(2, 4, 5, 10, 72), 73, 1e-6));
  static_assert(values::isnan(values::hypot(NaN)));
  static_assert(values::isnan(values::hypot(1, 2, NaN, 4)));
  static_assert(values::isinf(values::hypot(inf)));
  static_assert(values::isinf(values::hypot(1, 2, 3, inf, 5)));
  static_assert(values::isinf(values::hypot(1, NaN, 3, inf, 5)));

  constexpr auto a = std::complex<double>{3., 4.};
  constexpr auto b = std::complex<double>{5., 6.};
  static_assert(values::internal::near(values::hypot(a, b), values::sqrt(std::complex<double>{-18., 84.}), 1e-6));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::hypot(values::Fixed<double, 3>{}))>, 3, 1e-6));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::hypot(values::Fixed<double, 3>{}, values::Fixed<double, 4>{}))>, 5, 1e-6));
}


#include "values/math/abs.hpp"

TEST(values, abs)
{
  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    static_assert(values::abs(INFINITY) == INFINITY);
    static_assert(values::abs(-INFINITY) == INFINITY);
    EXPECT_TRUE(std::isnan(values::abs(-NAN)));
    EXPECT_FALSE(std::signbit(values::abs(NAN)));
#ifdef __cpp_lib_constexpr_cmath
    EXPECT_FALSE(std::signbit(values::abs(-0.)));
    EXPECT_FALSE(std::signbit(values::abs(-NAN)));
#endif
    EXPECT_TRUE(std::isnan(values::abs(std::complex<double>{3, NAN})));
    EXPECT_EQ(values::abs(std::complex<double>{INFINITY, 0}), INFINITY);
    EXPECT_EQ(values::abs(std::complex<double>{0, INFINITY}), INFINITY);
    EXPECT_EQ(values::abs(std::complex<double>{-INFINITY, 0}), INFINITY);
    EXPECT_EQ(values::abs(std::complex<double>{0, -INFINITY}), INFINITY);
    EXPECT_EQ(values::abs(std::complex<double>{INFINITY, 1}), INFINITY);
    EXPECT_EQ(values::abs(std::complex<double>{-INFINITY, 1}), INFINITY);
    EXPECT_EQ(values::abs(std::complex<double>{1, INFINITY}), INFINITY);
    EXPECT_EQ(values::abs(std::complex<double>{1, -INFINITY}), INFINITY);
    EXPECT_EQ(values::abs(std::complex<double>{INFINITY, NAN}), INFINITY);
    EXPECT_EQ(values::abs(std::complex<double>{-INFINITY, NAN}), INFINITY);
    EXPECT_EQ(values::abs(std::complex<double>{NAN, INFINITY}), INFINITY);
    EXPECT_EQ(values::abs(std::complex<double>{NAN, -INFINITY}), INFINITY);
  }

  static_assert(std::is_integral_v<decltype(values::abs(-3))>);
  static_assert(std::is_floating_point_v<decltype(values::abs(3.))>);
  static_assert(std::is_floating_point_v<decltype(values::abs(std::complex<double>{3., 4.}))>);
  static_assert(values::abs(3) == 3);
  static_assert(values::abs(-3) == 3);
  static_assert(values::abs(3.) == 3);
  static_assert(values::abs(-3.) == 3);
  static_assert(values::abs(-3.f) == 3);
  static_assert(values::abs(-3.l) == 3);

  EXPECT_EQ(values::abs(std::complex<double>{3, -4}), 5);
  EXPECT_EQ(values::abs(std::complex<double>{-3, 4}), 5);

  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::abs(std::complex<int>{3, 4})));
  COMPLEXINTEXISTS(static_assert(std::is_same_v<decltype(values::abs(std::complex<int>{3, -4})), int>));
  COMPLEXINTEXISTS(static_assert(values::abs(std::complex<int>{3, 4}) == 5));
  COMPLEXINTEXISTS(static_assert(values::abs(std::complex<int>{10, 15}) == 18));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::abs(std::integral_constant<int, -9>{}))>, 9, 1e-6));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::abs(values::Fixed<double, -9>{}))>, 9, 1e-6));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::abs(values::Fixed<std::complex<double>, 3, 4>{}))>, 5, 1e-6));
}


#include "values/math/exp.hpp"

TEST(values, exp)
{
  constexpr auto e = stdcompat::numbers::e_v<double>;
  constexpr auto eL = stdcompat::numbers::e_v<long double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::isnan(values::exp(values::internal::NaN<double>())));
    EXPECT_TRUE(values::exp(-values::internal::infinity<double>()) == 0);
    EXPECT_TRUE(values::exp(values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(std::isnan(values::real(values::exp(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}))));
    EXPECT_TRUE(std::isnan(values::imag(values::exp(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}))));
    EXPECT_TRUE(values::exp(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) !=
      values::exp(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::internal::near<10>(values::exp(0), 1));
  static_assert(values::internal::near<10>(values::exp(1), e));
  static_assert(values::internal::near<10>(values::exp(2), e*e));
  static_assert(values::internal::near<100>(values::exp(3), e*e*e));
  static_assert(values::internal::near<10>(values::exp(-1), 1/e));
  static_assert(values::internal::near<10>(values::exp(-2), 1/(e*e)));
  static_assert(values::internal::near<10>(values::exp(1.0), e));
  static_assert(values::internal::near<10>(values::exp(2.0), e*e));
  static_assert(values::internal::near<10>(values::exp(1.0L), eL));
  static_assert(values::internal::near<10>(values::exp(2.0L), eL*eL));
  static_assert(values::internal::near<100>(values::exp(3.0L), eL*eL*eL));
  EXPECT_NEAR(values::exp(5), std::exp(5), 1e-9);
  EXPECT_NEAR(values::exp(-10), std::exp(-10), 1e-9);
  EXPECT_NEAR(values::exp(50), std::exp(50), 1e8);
  EXPECT_NEAR(values::exp(50.5), std::exp(50.5), 1e8);
  EXPECT_NEAR(values::exp(300), std::exp(300), 1e120);
  EXPECT_NEAR(values::exp(300.7), std::exp(300.7), 1e120);
  EXPECT_NEAR(values::exp(1e-5), std::exp(1e-5), 1e-12);
  EXPECT_NEAR(values::exp(1e-10), std::exp(1e-10), 1e-16);

  static_assert(values::internal::near(values::real(values::exp(std::complex<double>{2, 0})), e*e, 1e-6));
  EXPECT_PRED3(tolerance, values::exp(std::complex<double>{3.3, -4.3}), std::exp(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::exp(std::complex<double>{10.4, 3.4}), std::exp(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, values::exp(std::complex<double>{-30.6, 20.6}), std::exp(std::complex<double>{-30.6, 20.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::exp(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(values::exp(std::complex<int>{3, -4}) == std::complex<int>{-13, 15}));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::exp(std::integral_constant<int, 2>{}))>, e*e, 1e-6));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::exp(values::Fixed<double, -2>{}))>, 1/(e*e), 1e-6));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::exp(values::Fixed<std::complex<double>, 2, 0>{}))>, e*e, 1e-6));
}


#include "values/math/expm1.hpp"

TEST(values, expm1)
{
  constexpr auto e = stdcompat::numbers::e_v<double>;
  constexpr auto eL = stdcompat::numbers::e_v<long double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(values::expm1(values::internal::NaN<double>()) != values::expm1(values::internal::NaN<double>()));
    EXPECT_TRUE(values::expm1(values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(values::expm1(-values::internal::infinity<double>()) == -1);
    EXPECT_TRUE(std::signbit(values::expm1(-0.)));
    EXPECT_FALSE(std::signbit(values::expm1(+0.)));
    EXPECT_TRUE(values::expm1(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::expm1(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::expm1(0) == 0);
  static_assert(values::internal::near(values::expm1(1), e - 1));
  static_assert(values::internal::near(values::expm1(2), e*e - 1, 1e-9));
  static_assert(values::internal::near(values::expm1(3), e*e*e - 1, 1e-9));
  static_assert(values::internal::near(values::expm1(-1), 1 / e - 1, 1e-9));
  static_assert(values::internal::near(values::expm1(3.L), eL*eL*eL - 1, 1e-9));
  static_assert(std::real(values::expm1(std::complex<double>{3e-12, 0})) == values::expm1(3e-12));
  EXPECT_PRED3(tolerance, values::expm1(1e-4), std::expm1(1e-4), 1e-9);
  EXPECT_PRED3(tolerance, values::expm1(1e-8), std::expm1(1e-8), 1e-9);
  EXPECT_PRED3(tolerance, values::expm1(1e-32), std::expm1(1e-32), 1e-9);
  EXPECT_PRED3(tolerance, values::expm1(5.2), std::expm1(5.2), 1e-9);
  EXPECT_PRED3(tolerance, values::expm1(10.2), std::expm1(10.2), 1e-9);
  EXPECT_PRED3(tolerance, values::expm1(-10.2), std::expm1(-10.2), 1e-9);
  EXPECT_PRED3(tolerance, values::expm1(3e-12), std::expm1(3e-12), 1e-9);
  EXPECT_PRED3(tolerance, values::expm1(-3e-12), std::expm1(-3e-12), 1e-9);

  static_assert(values::internal::near(values::real(values::expm1(std::complex<double>{2, 0})), e*e - 1, 1e-6));
  EXPECT_PRED3(tolerance, values::expm1(std::complex<double>{0.001, -0.001}), std::exp(std::complex<double>{0.001, -0.001}) - 1.0, 1e-9);
  EXPECT_PRED3(tolerance, values::expm1(std::complex<double>{3.2, -4.2}), std::exp(std::complex<double>{3.2, -4.2}) - 1.0, 1e-9);
  EXPECT_PRED3(tolerance, values::expm1(std::complex<double>{10.3, 3.3}), std::exp(std::complex<double>{10.3, 3.3}) - 1.0, 1e-9);
  EXPECT_PRED3(tolerance, values::expm1(std::complex<double>{-10.4, 10.4}), std::exp(std::complex<double>{-10.4, 10.4}) - 1.0, 1e-9);
  EXPECT_PRED3(tolerance, std::real(values::expm1(std::complex<double>{3e-12, 0})), std::expm1(3e-12), 1e-20);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::expm1(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(values::expm1(std::complex<int>{3, -4}) == std::complex<int>{-14, 15}));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::expm1(std::integral_constant<int, 2>{}))>, e*e - 1, 1e-6));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::expm1(values::Fixed<double, -2>{}))>, 1/(e*e) - 1, 1e-6));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::expm1(values::Fixed<std::complex<double>, 2, 0>{}))>, e*e - 1, 1e-6));
}


#include "values/math/sinh.hpp"

TEST(values, sinh)
{
  constexpr auto e = stdcompat::numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(values::sinh(values::internal::NaN<double>()) != values::sinh(values::internal::NaN<double>()));
    EXPECT_TRUE(values::sinh(values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(values::sinh(-values::internal::infinity<double>()) == -values::internal::infinity<double>());
    EXPECT_TRUE(std::signbit(values::sinh(-0.)));
    EXPECT_TRUE(not std::signbit(values::sinh(0.)));
    EXPECT_TRUE(values::sinh(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::sinh(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::sinh(0) == 0);
  static_assert(values::internal::near(values::sinh(1), (e - 1/e)/2, 1e-9));
  static_assert(values::internal::near(values::sinh(2), (e*e - 1/e/e)/2, 1e-9));
  static_assert(values::internal::near(values::sinh(3), (e*e*e - 1/e/e/e)/2, 1e-9));
  static_assert(values::internal::near(values::sinh(-1), (1/e - e)/2, 1e-9));
  static_assert(values::internal::near(values::sinh(-2), (1/e/e - e*e)/2, 1e-9));
  static_assert(values::internal::near(values::sinh(-3), (1/e/e/e - e*e*e)/2, 1e-9));
  EXPECT_NEAR(values::sinh(5), std::sinh(5), 1e-9);
  EXPECT_NEAR(values::sinh(-10), std::sinh(-10), 1e-9);

  static_assert(values::internal::near(values::real(values::sinh(std::complex<double>{2, 0})), (e*e - 1/e/e)/2, 1e-9));
  EXPECT_PRED3(tolerance, values::sinh(std::complex<double>{3.3, -4.3}), std::sinh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::sinh(std::complex<double>{10.4, 3.4}), std::sinh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, values::sinh(std::complex<double>{-10.6, 10.6}), std::sinh(std::complex<double>{-10.6, 10.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::sinh(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(values::sinh(std::complex<int>{3, -4}) == std::complex<int>{-6, 7}));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::sinh(std::integral_constant<int, 2>{}))>, (e*e - 1/e/e)/2, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::sinh(values::Fixed<double, -2>{}))>, (1/e/e - e*e)/2, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::sinh(values::Fixed<std::complex<double>, 2, 0>{})))>, (e*e - 1/e/e)/2, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::sinh(values::Fixed<std::complex<double>, 3, -4>{})))>, -6.548120040911001647767, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::sinh(values::Fixed<std::complex<double>, 3, -4>{})))>, 7.619231720321410208487, 1e-9));
}


#include "values/math/cosh.hpp"

TEST(values, cosh)
{
  constexpr auto e = stdcompat::numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(values::cosh(values::internal::NaN<double>()) != values::cosh(values::internal::NaN<double>()));
    EXPECT_TRUE(values::cosh(values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(values::cosh(-values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(values::cosh(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::cosh(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::cosh(0) == 1);
  static_assert(values::internal::near(values::cosh(1), (e + 1/e)/2, 1e-9));
  static_assert(values::internal::near(values::cosh(2), (e*e + 1/e/e)/2, 1e-9));
  static_assert(values::internal::near(values::cosh(3), (e*e*e + 1/e/e/e)/2, 1e-9));
  static_assert(values::internal::near(values::cosh(-1), (1/e + e)/2, 1e-9));
  static_assert(values::internal::near(values::cosh(-2), (1/e/e + e*e)/2, 1e-9));
  static_assert(values::internal::near(values::cosh(-3), (1/e/e/e + e*e*e)/2, 1e-9));
  EXPECT_NEAR(values::cosh(5), std::cosh(5), 1e-9);
  EXPECT_NEAR(values::cosh(-10), std::cosh(-10), 1e-9);

  static_assert(values::internal::near(values::real(values::cosh(std::complex<double>{2, 0})), (e*e + 1/e/e)/2, 1e-9));
  EXPECT_PRED3(tolerance, values::cosh(std::complex<double>{3.3, -4.3}), std::cosh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::cosh(std::complex<double>{10.4, 3.4}), std::cosh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, values::cosh(std::complex<double>{-10.6, 10.6}), std::cosh(std::complex<double>{-10.6, 10.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::cosh(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(values::cosh(std::complex<int>{5, -6}) == std::complex<int>{71, 20}));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::cosh(std::integral_constant<int, 2>{}))>, (e*e + 1/e/e)/2, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::cosh(values::Fixed<double, -2>{}))>, (1/e/e + e*e)/2, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::cosh(values::Fixed<std::complex<double>, 2, 0>{})))>, (e*e + 1/e/e)/2, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::cosh(values::Fixed<std::complex<double>, 3, -4>{})))>, -6.580663040551156432561, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::cosh(values::Fixed<std::complex<double>, 3, -4>{})))>, 7.581552742746544353716, 1e-9));
}


#include "values/math/tanh.hpp"

TEST(values, tanh)
{
  constexpr auto e = stdcompat::numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(values::tanh(values::internal::NaN<double>()) != values::tanh(values::internal::NaN<double>()));
    EXPECT_TRUE(values::tanh(values::internal::infinity<double>()) == 1);
    EXPECT_TRUE(values::tanh(-values::internal::infinity<double>()) == -1);
    EXPECT_TRUE(std::signbit(values::tanh(-0.)));
    EXPECT_TRUE(not std::signbit(values::tanh(0.)));
    EXPECT_TRUE(values::tanh(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::tanh(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::tanh(0) == 0);
  static_assert(values::internal::near(values::tanh(1), (e*e - 1)/(e*e + 1), 1e-9));
  static_assert(values::internal::near(values::tanh(2), (e*e*e*e - 1)/(e*e*e*e + 1), 1e-9));
  static_assert(values::internal::near(values::tanh(3), (e*e*e*e*e*e - 1)/(e*e*e*e*e*e + 1), 1e-9));
  static_assert(values::internal::near(values::tanh(-1), (1 - e*e)/(1 + e*e), 1e-9));
  static_assert(values::internal::near(values::tanh(-2), (1 - e*e*e*e)/(1 + e*e*e*e), 1e-9));
  static_assert(values::internal::near(values::tanh(-3), (1 - e*e*e*e*e*e)/(1 + e*e*e*e*e*e), 1e-9));
  EXPECT_NEAR(values::tanh(5), std::tanh(5), 1e-9);
  EXPECT_NEAR(values::tanh(-10), std::tanh(-10), 1e-9);

  static_assert(values::internal::near(values::real(values::tanh(std::complex<double>{2, 0})), (e*e*e*e - 1)/(e*e*e*e + 1), 1e-9));
  EXPECT_PRED3(tolerance, values::tanh(std::complex<double>{3.3, -4.3}), std::tanh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::tanh(std::complex<double>{10.4, 3.4}), std::tanh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, values::tanh(std::complex<double>{-30.6, 20.6}), std::tanh(std::complex<double>{-30.6, 20.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::tanh(std::complex<int>{3, -4})));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::tanh(std::integral_constant<int, 2>{}))>, (e*e*e*e - 1)/(e*e*e*e + 1), 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::tanh(values::Fixed<double, -2>{}))>, (1 - e*e*e*e)/(1 + e*e*e*e), 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::tanh(values::Fixed<std::complex<double>, 3, 4>{})))>, 1.00070953606723293933, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::tanh(values::Fixed<std::complex<double>, 3, 4>{})))>, 0.004908258067496060259079, 1e-9));
}


#include "values/math/sin.hpp"

TEST(values, sin)
{
  constexpr auto pi = stdcompat::numbers::pi_v<double>;
  constexpr auto piL = stdcompat::numbers::pi_v<long double>;
  constexpr auto piF = stdcompat::numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::isnan(values::sin(values::internal::NaN<double>())));
    EXPECT_TRUE(std::isnan(values::sin(values::internal::infinity<double>())));
    EXPECT_TRUE(std::isnan(values::sin(-values::internal::infinity<double>())));
    EXPECT_FALSE(std::signbit(values::sin(+0.)));
    EXPECT_TRUE(std::signbit(values::sin(-0.)));
    EXPECT_TRUE(values::sin(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::sin(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::internal::near(values::sin(0), 0));
  static_assert(values::internal::near(values::sin(2*pi), 0));
  static_assert(values::internal::near(values::sin(-2*pi), 0));
  static_assert(values::internal::near(values::sin(pi), 0));
  static_assert(values::internal::near(values::sin(-pi), 0));
  static_assert(values::internal::near(values::sin(32*pi), 0, 1e-9));
  static_assert(values::internal::near(values::sin(-32*pi), 0, 1e-9));
  static_assert(values::internal::near(values::sin(0x1p16*pi), 0, 1e-9));
  static_assert(values::internal::near(values::sin(-0x1p16*pi), 0, 1e-9));
  static_assert(values::internal::near(values::sin(0x1p16L*piL), 0, 1e-9));
  static_assert(values::internal::near(values::sin(-0x1p16L*piL), 0, 1e-9));
  static_assert(values::internal::near(values::sin(0x1p16F*piF), 0, 1e-2));
  static_assert(values::internal::near(values::sin(-0x1p16F*piF), 0, 1e-2));
  static_assert(values::internal::near(values::sin(0x1p100L*piL), 0, 1.));
  static_assert(values::internal::near(values::sin(-0x1p100L*piL), 0, 1.));
  static_assert(values::internal::near(values::sin(0x1p180L*piL), 0, 1.));
  static_assert(values::internal::near(values::sin(-0x1p180L*piL), 0, 1.));
  static_assert(values::internal::near(values::sin(0x1p250L*piL), 0, 1.));
  static_assert(values::internal::near(values::sin(-0x1p250L*piL), 0, 1.));
  static_assert(values::internal::near(values::sin(pi/2), 1));
  static_assert(values::internal::near(values::sin(-pi/2), -1));
  static_assert(values::internal::near(values::sin(pi/4), stdcompat::numbers::sqrt2_v<double>/2));
  static_assert(values::internal::near(values::sin(piL/4), stdcompat::numbers::sqrt2_v<long double>/2));
  static_assert(values::internal::near(values::sin(piF/4), stdcompat::numbers::sqrt2_v<float>/2));
  static_assert(values::internal::near(values::sin(pi/4 + 32*pi), stdcompat::numbers::sqrt2_v<double>/2, 1e-9));
  EXPECT_NEAR(values::sin(2), std::sin(2), 1e-9);
  EXPECT_NEAR(values::sin(-32), std::sin(-32), 1e-9);
  EXPECT_NEAR(values::sin(0x1p16), std::sin(0x1p16), 1e-9);

  static_assert(values::internal::near(values::sin(std::complex<double>{pi/2, 0}), 1));
  EXPECT_PRED3(tolerance, values::sin(std::complex<double>{4.1, 3.1}), std::sin(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, values::sin(std::complex<double>{3.2, -4.2}), std::sin(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, values::sin(std::complex<double>{-3.3, 4.3}), std::sin(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::sin(std::complex<double>{-9.3, 10.3}), std::sin(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::sin(std::complex<int>{3, -4})));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::sin(std::integral_constant<int, 2>{}))>, 0.909297426825681695396, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::sin(values::Fixed<double, 2>{}))>, 0.909297426825681695396, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::sin(values::Fixed<std::complex<double>, 2, 0>{})))>, 0.909297426825681695396, 1e-9));
}


#include "values/math/cos.hpp"

TEST(values, cos)
{
  constexpr auto pi = stdcompat::numbers::pi_v<double>;
  constexpr auto piL = stdcompat::numbers::pi_v<long double>;
  constexpr auto piF = stdcompat::numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::isnan(values::cos(values::internal::NaN<double>())));
    EXPECT_TRUE(std::isnan(values::cos(values::internal::infinity<double>())));
    EXPECT_TRUE(std::isnan(values::cos(-values::internal::infinity<double>())));
    EXPECT_TRUE(values::cos(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::cos(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::cos(2*pi) == 1);
  static_assert(values::cos(-2*pi) == 1);
  static_assert(values::cos(0) == 1);
  static_assert(values::internal::near(values::cos(pi), -1));
  static_assert(values::internal::near(values::cos(-pi), -1));
  static_assert(values::internal::near(values::cos(32*pi), 1));
  static_assert(values::internal::near(values::cos(-32*pi), 1));
  static_assert(values::internal::near(values::cos(0x1p16*pi), 1));
  static_assert(values::internal::near(values::cos(-0x1p16*pi), 1));
  static_assert(values::internal::near(values::cos(0x1p16L*piL), 1));
  static_assert(values::internal::near(values::cos(-0x1p16L*piL), 1));
  static_assert(values::internal::near(values::cos(0x1p16F*piF), 1, 1e-4));
  static_assert(values::internal::near(values::cos(-0x1p16F*piF), 1, 1e-4));
  static_assert(values::internal::near(values::cos(0x1p100L*piL), 1, 1.));
  static_assert(values::internal::near(values::cos(-0x1p100L*piL), 1, 1.));
  static_assert(values::internal::near(values::cos(0x1p180L*piL), 1, 1.));
  static_assert(values::internal::near(values::cos(-0x1p180L*piL), 1, 1.));
  static_assert(values::internal::near(values::cos(0x1p250L*piL), 1, 1.));
  static_assert(values::internal::near(values::cos(-0x1p250L*piL), 1, 1.));
  static_assert(values::internal::near(values::cos(pi/2), 0));
  static_assert(values::internal::near(values::cos(-pi/2), 0));
  static_assert(values::internal::near(values::cos(pi/4), stdcompat::numbers::sqrt2_v<double>/2));
  static_assert(values::internal::near(values::cos(piL/4), stdcompat::numbers::sqrt2_v<long double>/2));
  static_assert(values::internal::near(values::cos(piF/4), stdcompat::numbers::sqrt2_v<float>/2));
  EXPECT_NEAR(values::cos(2), std::cos(2), 1e-9);
  EXPECT_NEAR(values::cos(-32), std::cos(-32), 1e-9);
  EXPECT_NEAR(values::cos(0x1p16), std::cos(0x1p16), 1e-9);

  static_assert(values::internal::near(values::cos(std::complex<double>{pi/2, 0}), 0));
  EXPECT_PRED3(tolerance, values::cos(std::complex<double>{4.1, 3.1}), std::cos(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, values::cos(std::complex<double>{3.2, -4.2}), std::cos(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, values::cos(std::complex<double>{-3.3, 4.3}), std::cos(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::cos(std::complex<double>{-9.3, 10.3}), std::cos(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::cos(std::complex<int>{3, -4})));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::cos(std::integral_constant<int, 2>{}))>, -0.4161468365471423869976, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::cos(values::Fixed<double, 2>{}))>, -0.4161468365471423869976, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::cos(values::Fixed<std::complex<double>, 2, 0>{})))>, -0.4161468365471423869976, 1e-9));
}


#include "values/math/tan.hpp"

TEST(values, tan)
{
  constexpr auto pi = stdcompat::numbers::pi_v<double>;
  constexpr auto piL = stdcompat::numbers::pi_v<long double>;
  constexpr auto piF = stdcompat::numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(values::tan(values::internal::NaN<double>()) != values::tan(values::internal::NaN<double>()));
    EXPECT_TRUE(values::tan(values::internal::infinity<double>()) != values::tan(values::internal::infinity<double>()));
    EXPECT_TRUE(values::tan(-values::internal::infinity<double>()) != values::tan(values::internal::infinity<double>()));
    EXPECT_TRUE(values::tan(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::tan(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::tan(0) == 0);
  static_assert(values::internal::near(values::tan(2*pi), 0));
  static_assert(values::internal::near(values::tan(-2*pi), 0));
  static_assert(values::internal::near(values::tan(pi), 0));
  static_assert(values::internal::near(values::tan(-pi), 0));
  static_assert(values::internal::near(values::tan(32*pi), 0, 1e-9));
  static_assert(values::internal::near(values::tan(-32*pi), 0, 1e-9));
  static_assert(values::internal::near(values::tan(0x1p16*pi), 0, 1e-9));
  static_assert(values::internal::near(values::tan(-0x1p16*pi), 0, 1e-9));
  static_assert(values::internal::near(values::tan(0x1p16L*piL), 0, 1e-9));
  static_assert(values::internal::near(values::tan(-0x1p16L*piL), 0, 1e-9));
  static_assert(values::internal::near(values::tan(0x1p16F*piF), 0, 1e-2));
  static_assert(values::internal::near(values::tan(-0x1p16F*piF), 0, 1e-2));
  static_assert(values::internal::near(values::tan(0x1p100L*piL), 0, 1.));
  static_assert(values::internal::near(values::tan(-0x1p100L*piL), 0, 1.));
  static_assert(values::internal::near(values::tan(0x1p180L*piL), 0, 1.));
  static_assert(values::internal::near(values::tan(-0x1p180L*piL), 0, 1.));
  static_assert(values::internal::near(values::tan(0x1p250L*piL), 0, 2.));
  static_assert(values::internal::near(values::tan(-0x1p250L*piL), 0, 2.));
  static_assert(values::internal::near(values::tan(pi/4), 1));
  static_assert(values::internal::near(values::tan(piL/4), 1));
  static_assert(values::internal::near(values::tan(piF/4), 1));
  static_assert(values::internal::near(values::tan(pi/4 + 32*pi), 1, 1e-9));
  EXPECT_NEAR(values::tan(2), std::tan(2), 1e-9);
  EXPECT_NEAR(values::tan(-32), std::tan(-32), 1e-9);
  EXPECT_NEAR(values::tan(0x1p16), std::tan(0x1p16), 1e-9);

  static_assert(values::internal::near(values::tan(std::complex<double>{pi/4, 0}), 1));
  EXPECT_PRED3(tolerance, values::tan(std::complex<double>{4.1, 3.1}), std::tan(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, values::tan(std::complex<double>{3.2, -4.2}), std::tan(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, values::tan(std::complex<double>{-3.3, 4.3}), std::tan(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::tan(std::complex<double>{-30.3, 40.3}), std::tan(std::complex<double>{-30.3, 40.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::tan(std::complex<int>{30, -2})));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::tan(std::integral_constant<int, 2>{}))>, -2.185039863261518991643, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::tan(values::Fixed<double, 2>{}))>, -2.185039863261518991643, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::tan(values::Fixed<std::complex<double>, 3, 4>{})))>, -1.873462046294784262243E-4, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::tan(values::Fixed<std::complex<double>, 3, 4>{})))>, 0.9993559873814731413917, 1e-9));
}


#include "values/math/log.hpp"

TEST(values, log)
{
  constexpr auto e = stdcompat::numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(values::log(0) == -values::internal::infinity<double>());
    EXPECT_TRUE(values::log(-0) == -values::internal::infinity<double>());
    EXPECT_TRUE(values::log(+values::internal::infinity<double>()) == +values::internal::infinity<double>());
    EXPECT_TRUE(values::log(-1) != values::log(-1)); // Nan
    EXPECT_FALSE(std::signbit(values::log(1)));
    EXPECT_TRUE(values::log(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::log(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::log(1) == 0);
  static_assert(values::internal::near<10>(values::log(2), stdcompat::numbers::ln2_v<double>));
  static_assert(values::internal::near<10>(values::log(10), stdcompat::numbers::ln10_v<double>));
  static_assert(values::internal::near(values::log(e), 1));
  static_assert(values::internal::near(values::log(e*e), 2));
  static_assert(values::internal::near(values::log(e*e*e), 3));
  static_assert(values::internal::near(values::log(e*e*e*e*e*e*e*e*e*e*e*e*e*e*e*e), 16));
  static_assert(values::internal::near(values::log(1 / e), -1));
  EXPECT_NEAR(values::log(5.0L), std::log(5.0L), 1e-9);
  EXPECT_NEAR(values::log(0.2L), std::log(0.2L), 1e-9);
  EXPECT_NEAR(values::log(5), std::log(5), 1e-9);
  EXPECT_NEAR(values::log(0.2), std::log(0.2), 1e-9);
  EXPECT_NEAR(values::log(20), std::log(20), 1e-9);
  EXPECT_NEAR(values::log(0.05), std::log(0.05), 1e-9);
  EXPECT_NEAR(values::log(100), std::log(100), 1e-9);
  EXPECT_NEAR(values::log(0.01), std::log(0.01), 1e-9);
  EXPECT_NEAR(values::log(1e20), std::log(1e20), 1e-9);
  EXPECT_NEAR(values::log(1e-20), std::log(1e-20), 1e-9);
  EXPECT_NEAR(values::log(1e200), std::log(1e200), 1e-9);
  EXPECT_NEAR(values::log(1e-200), std::log(1e-200), 1e-9);
  EXPECT_NEAR(values::log(1e200L), std::log(1e200L), 1e-9);
  EXPECT_NEAR(values::log(1e-200L), std::log(1e-200L), 1e-9);

  static_assert(values::internal::near(values::log(std::complex<double>{e*e, 0}), 2));
  EXPECT_PRED3(tolerance, values::log(std::complex<double>{-4}), std::log(std::complex<double>{-4}), 1e-9);
  EXPECT_PRED3(tolerance, values::log(std::complex<double>{3, 4}), std::log(std::complex<double>{3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, values::log(std::complex<double>{3, -4}), std::log(std::complex<double>{3, -4}), 1e-9);
  EXPECT_PRED3(tolerance, values::log(std::complex<double>{-3, 4}), std::log(std::complex<double>{-3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, values::log(std::complex<double>{-3, -4}), std::log(std::complex<double>{-3, -4}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::log(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(values::log(std::complex<int>{1, 0}) == std::complex<int>{0, 0}));
  COMPLEXINTEXISTS(static_assert(values::log(std::complex<int>{100, 0}) == std::complex<int>{4, 0}));
  COMPLEXINTEXISTS(static_assert(values::log(std::complex<int>{-100, 0}) == std::complex<int>{4, 3}));
  COMPLEXINTEXISTS(EXPECT_PRED3(tolerance, values::log(std::complex<int>{-3, 0}), std::log(std::complex<int>{-3, 0}), 1e-9));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::log(std::integral_constant<int, 2>{}))>, stdcompat::numbers::ln2_v<double>, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::log(values::Fixed<double, 2>{}))>, stdcompat::numbers::ln2_v<double>, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::log(values::Fixed<std::complex<double>, 3, 4>{})))>, 1.609437912434100374601, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::log(values::Fixed<std::complex<double>, 3, 4>{})))>, 0.9272952180016122324285, 1e-9));
}


#include "values/math/log1p.hpp"

TEST(values, log1p)
{
  constexpr auto e = stdcompat::numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(std::signbit(values::log1p(-0.)));
    EXPECT_FALSE(std::signbit(values::log1p(+0.)));
    EXPECT_EQ(values::log1p(-1), -values::internal::infinity<double>());
    EXPECT_EQ(values::log1p(+values::internal::infinity<double>()), +values::internal::infinity<double>());
    EXPECT_TRUE(values::log1p(-2) != values::log(-2)); // Nan
    EXPECT_TRUE(values::log1p(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::log1p(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::internal::near<10>(values::log1p(-0.), 0));
  static_assert(values::internal::near<10>(values::log1p(1.), stdcompat::numbers::ln2_v<double>));
  static_assert(values::internal::near<10>(values::log1p(9.), stdcompat::numbers::ln10_v<double>));
  static_assert(values::internal::near<10>(values::log1p(e - 1), 1));
  static_assert(values::internal::near<10>(values::log1p(e*e - 1), 2));
  static_assert(values::internal::near<10>(values::log1p(e*e*e - 1), 3));
  static_assert(values::internal::near<10>(values::log1p(e*e*e*e*e*e*e*e*e*e*e*e*e*e*e*e - 1), 16));
  static_assert(values::internal::near<10>(values::log1p(1/e - 1), -1));
  EXPECT_NEAR(values::log1p(5.0L), std::log1p(5.0L), 1e-9);
  EXPECT_NEAR(values::log1p(0.2L), std::log1p(0.2L), 1e-9);
  EXPECT_NEAR(values::log1p(5), std::log1p(5), 1e-9);
  EXPECT_NEAR(values::log1p(0.2), std::log1p(0.2), 1e-9);
  EXPECT_NEAR(values::log1p(20), std::log1p(20), 1e-9);
  EXPECT_NEAR(values::log1p(0.05), std::log1p(0.05), 1e-9);
  EXPECT_NEAR(values::log1p(100), std::log1p(100), 1e-9);
  EXPECT_NEAR(values::log1p(0.01), std::log1p(0.01), 1e-9);
  EXPECT_NEAR(values::log1p(0.001), std::log1p(0.001), 1e-9);
  EXPECT_NEAR(values::log1p(0.0001), std::log1p(0.0001), 1e-9);
  EXPECT_NEAR(values::log1p(0.00001), std::log1p(0.00001), 1e-9);
  EXPECT_NEAR(values::log1p(0.000001), std::log1p(0.000001), 1e-9);
  EXPECT_NEAR(values::log1p(1e-20), std::log1p(1e-20), 1e-9);
  EXPECT_NEAR(values::log1p(1e-200), std::log1p(1e-200), 1e-9);
  EXPECT_NEAR(values::log1p(1e-200L), std::log1p(1e-200L), 1e-9);
  EXPECT_NEAR(values::log1p(1e20), std::log1p(1e20), 1e-9);
  EXPECT_NEAR(values::log1p(1e200), std::log1p(1e200), 1e-9);
  EXPECT_NEAR(values::log1p(1e200L), std::log1p(1e200L), 1e-9);

  static_assert(values::internal::near(values::log1p(std::complex<double>{e*e - 1, 0}), 2));
  EXPECT_PRED3(tolerance, std::real(values::log1p(std::complex<double>{4e-21})), std::log1p(4e-21), 1e-30);
  EXPECT_PRED3(tolerance, values::log1p(std::complex<double>{-4}), std::log(std::complex<double>{-3}), 1e-9);
  EXPECT_PRED3(tolerance, values::log1p(std::complex<double>{3, 4}), std::log(std::complex<double>{4, 4}), 1e-9);
  EXPECT_PRED3(tolerance, values::log1p(std::complex<double>{3, -4}), std::log(std::complex<double>{4, -4}), 1e-9);
  EXPECT_PRED3(tolerance, values::log1p(std::complex<double>{-3, 4}), std::log(std::complex<double>{-2, 4}), 1e-9);
  EXPECT_PRED3(tolerance, values::log1p(std::complex<double>{-3, -4}), std::log(std::complex<double>{-2, -4}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::log1p(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(values::log1p(std::complex<int>{0, 0}) == std::complex<int>{0, 0}));
  COMPLEXINTEXISTS(static_assert(values::log1p(std::complex<int>{99, 0}) == std::complex<int>{4, 0}));
  COMPLEXINTEXISTS(static_assert(values::log1p(std::complex<int>{-101, 0}) == std::complex<int>{4, 3}));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::log1p(std::integral_constant<int, 1>{}))>, stdcompat::numbers::ln2_v<double>, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::log1p(values::Fixed<double, 1>{}))>, stdcompat::numbers::ln2_v<double>, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::log1p(values::Fixed<std::complex<double>, 2, 4>{})))>, 1.609437912434100374601, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::log1p(values::Fixed<std::complex<double>, 2, 4>{})))>, 0.9272952180016122324285, 1e-9));
}


#include "values/math/asinh.hpp"

TEST(values, asinh)
{
  constexpr auto e = stdcompat::numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(values::asinh(values::internal::NaN<double>()) != values::asinh(values::internal::NaN<double>()));
    EXPECT_TRUE(values::asinh(values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(values::asinh(-values::internal::infinity<double>()) == -values::internal::infinity<double>());
    EXPECT_FALSE(std::signbit(values::asinh(+0.)));
    EXPECT_TRUE(std::signbit(values::asinh(-0.)));
    EXPECT_TRUE(values::asinh(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::asinh(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::asinh(0) == 0);
  static_assert(values::internal::near(values::asinh((e - 1/e)/2), 1));
  static_assert(values::internal::near(values::asinh((e*e - 1/e/e)/2), 2));
  static_assert(values::internal::near(values::asinh((e*e*e - 1/e/e/e)/2), 3, 1e-9));
  static_assert(values::internal::near(values::asinh((1/e - e)/2), -1));
  static_assert(values::internal::near(values::asinh((1/e/e - e*e)/2), -2, 1e-9));
  static_assert(values::internal::near(values::asinh((1/e/e/e - e*e*e)/2), -3, 1e-9));
  EXPECT_NEAR(values::asinh(5), std::asinh(5), 1e-9);
  EXPECT_NEAR(values::asinh(-10), std::asinh(-10), 1e-9);

  static_assert(values::internal::near(values::asinh(std::complex<double>{(e*e - 1/e/e)/2, 0}), 2));
  EXPECT_PRED3(tolerance, values::asinh(std::complex<double>{3.3, -4.3}), std::asinh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::asinh(std::complex<double>{10.4, 3.4}), std::asinh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, values::asinh(std::complex<double>{-10.6, 10.6}), std::asinh(std::complex<double>{-10.6, 10.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::asinh(std::complex<int>{3, -4})));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::asinh(std::integral_constant<int, 2>{}))>, 1.443635475178810342493, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::asinh(values::Fixed<double, 2>{}))>, 1.443635475178810342493, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::asinh(values::Fixed<std::complex<double>, 3, 4>{})))>, 2.299914040879269649956, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::asinh(values::Fixed<std::complex<double>, 3, 4>{})))>, 0.9176168533514786557599, 1e-9));
}


#include "values/math/acosh.hpp"

TEST(values, acosh)
{
  constexpr auto e = stdcompat::numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(values::acosh(values::internal::NaN<double>()) != values::acosh(values::internal::NaN<double>()));
    EXPECT_TRUE(values::acosh(-1) != values::acosh(-1));
    EXPECT_TRUE(values::acosh(values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(values::acosh(-values::internal::infinity<double>()) != values::acosh(-values::internal::infinity<double>()));
    EXPECT_TRUE(values::acosh(0.9) != values::acosh(0.9));
    EXPECT_TRUE(values::acosh(-1) != values::acosh(-1));
    EXPECT_FALSE(std::signbit(values::acosh(1)));
    EXPECT_TRUE(values::acosh(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::acosh(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::acosh(1) == 0);
  static_assert(values::internal::near(values::acosh((e + 1/e)/2), 1));
  static_assert(values::internal::near(values::acosh((e*e + 1/e/e)/2), 2));
  static_assert(values::internal::near(values::acosh((e*e*e + 1/e/e/e)/2), 3, 1e-9));
  EXPECT_NEAR(values::acosh(5), std::acosh(5), 1e-9);
  EXPECT_NEAR(values::acosh(10), std::acosh(10), 1e-9);

  static_assert(values::internal::near(values::acosh(std::complex<double>{(e*e + 1/e/e)/2, 0}), 2));
  EXPECT_PRED3(tolerance, values::acosh(std::complex<double>{-2, 0}), std::acosh(std::complex<double>{-2, 0}), 1e-9);
  EXPECT_PRED3(tolerance, values::acosh(std::complex<double>{3.3, -4.3}), std::acosh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::acosh(std::complex<double>{5.4, 3.4}), std::acosh(std::complex<double>{5.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, values::acosh(std::complex<double>{-5.6, 5.6}), std::acosh(std::complex<double>{-5.6, 5.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::acosh(std::complex<int>{3, -4})));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::acosh(std::integral_constant<int, 2>{}))>, 1.316957896924816708625, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::acosh(values::Fixed<double, 2>{}))>, 1.316957896924816708625, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::acosh(values::Fixed<std::complex<double>, 3, 4>{})))>, 2.305509031243476942042, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::acosh(values::Fixed<std::complex<double>, 3, 4>{})))>, 0.9368124611557199029125, 1e-9));
}


#include "values/math/atanh.hpp"

TEST(values, atanh)
{
  constexpr auto e = stdcompat::numbers::e_v<double>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(values::atanh(values::internal::NaN<double>()) != values::atanh(values::internal::NaN<double>()));
    EXPECT_TRUE(values::atanh(2) != values::atanh(2));
    EXPECT_TRUE(values::atanh(-2) != values::atanh(-2));
    EXPECT_TRUE(values::atanh(1) == values::internal::infinity<double>());
    EXPECT_TRUE(values::atanh(-1) == -values::internal::infinity<double>());
    EXPECT_FALSE(std::signbit(values::atanh(+0.)));
    EXPECT_TRUE(std::signbit(values::atanh(-0.)));
    EXPECT_TRUE(values::atanh(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::atanh(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::atanh(0) == 0);
  static_assert(values::internal::near(values::atanh((e*e - 1)/(e*e + 1)), 1));
  static_assert(values::internal::near(values::atanh((e*e*e*e - 1)/(e*e*e*e + 1)), 2));
  static_assert(values::internal::near(values::atanh((e*e*e*e*e*e - 1)/(e*e*e*e*e*e + 1)), 3, 1e-9));
  static_assert(values::internal::near(values::atanh((1 - e*e)/(1 + e*e)), -1));
  static_assert(values::internal::near(values::atanh((1 - e*e*e*e)/(1 + e*e*e*e)), -2));
  static_assert(values::internal::near(values::atanh((1 - e*e*e*e*e*e)/(1 + e*e*e*e*e*e)), -3, 1e-9));
  EXPECT_NEAR(values::atanh(0.99), std::atanh(0.99), 1e-9);
  EXPECT_NEAR(values::atanh(-0.99), std::atanh(-0.99), 1e-9);

  static_assert(values::internal::near(values::atanh(std::complex<double>{(e*e*e*e - 1)/(e*e*e*e + 1), 0}), 2));
  EXPECT_PRED3(tolerance, values::atanh(std::complex<double>{3.3, -4.3}), std::atanh(std::complex<double>{3.3, -4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::atanh(std::complex<double>{10.4, 3.4}), std::atanh(std::complex<double>{10.4, 3.4}), 1e-9);
  EXPECT_PRED3(tolerance, values::atanh(std::complex<double>{-30.6, 20.6}), std::atanh(std::complex<double>{-30.6, 20.6}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::atanh(std::complex<int>{3, -4})));
  COMPLEXINTEXISTS(static_assert(values::atanh(std::complex<int>{3, 4}) == std::complex<int>{0, 1}));
  COMPLEXINTEXISTS(static_assert(values::atanh(std::complex<int>{3, -4}) == std::complex<int>{0, -1}));
  COMPLEXINTEXISTS(static_assert(values::atanh(std::complex<int>{0, 3}) == std::complex<int>{0, 1}));
  COMPLEXINTEXISTS(static_assert(values::atanh(std::complex<int>{0, -7}) == std::complex<int>{0, -1}));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::atanh(std::integral_constant<int, 0>{}))>, 0, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::atanh(values::Fixed<double, 0>{}))>, 0, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::atanh(values::Fixed<std::complex<double>, 3, 4>{})))>, 0.1175009073114338884127, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::atanh(values::Fixed<std::complex<double>, 3, 4>{})))>, 1.409921049596575522531, 1e-9));
}


#include "values/math/asin.hpp"

TEST(values, asin)
{
  constexpr auto pi = stdcompat::numbers::pi_v<double>;
  constexpr auto piL = stdcompat::numbers::pi_v<long double>;
  constexpr auto piF = stdcompat::numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(values::asin(values::internal::NaN<double>()) != values::asin(values::internal::NaN<double>()));
    EXPECT_TRUE(values::asin(2.0) != values::asin(2.0));
    EXPECT_TRUE(values::asin(-2.0) != values::asin(-2.0));
    EXPECT_TRUE(std::signbit(values::asin(-0.)));
    EXPECT_TRUE(not std::signbit(values::asin(0.)));
    EXPECT_TRUE(values::asin(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::asin(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::asin(0) == 0);
  static_assert(values::asin(1) == pi/2);
  static_assert(values::asin(1.0L) == piL/2);
  static_assert(values::asin(1.0F) == piF/2);
  static_assert(values::asin(-1) == -pi/2);
  static_assert(values::internal::near(values::asin(stdcompat::numbers::sqrt2_v<double>/2), pi/4));
  static_assert(values::internal::near(values::asin(-stdcompat::numbers::sqrt2_v<double>/2), -pi/4));
  static_assert(values::asin(0.99995) > 0);
  static_assert(values::asin(-0.99995) < 0);
  EXPECT_NEAR(values::asin(stdcompat::numbers::sqrt2_v<double>/2), pi/4, 1e-9);
  EXPECT_NEAR(values::asin(-0.7), std::asin(-0.7), 1e-9);
  EXPECT_NEAR(values::asin(0.9), std::asin(0.9), 1e-9);
  EXPECT_NEAR(values::asin(0.99), std::asin(0.99), 1e-9);
  EXPECT_NEAR(values::asin(0.999), std::asin(0.999), 1e-9);
  EXPECT_NEAR(values::asin(-0.999), std::asin(-0.999), 1e-9);
  EXPECT_NEAR(values::asin(0.99999), std::asin(0.99999), 1e-9);
  EXPECT_NEAR(values::asin(0.99999999), std::asin(0.99999999), 1e-9);

  static_assert(values::internal::near(values::asin(std::complex<double>{stdcompat::numbers::sqrt2_v<double>/2, 0}), pi/4, 1e-9));
  EXPECT_PRED3(tolerance, values::asin(std::complex<double>{4.1, 3.1}), std::asin(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, values::asin(std::complex<double>{3.2, -4.2}), std::asin(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, values::asin(std::complex<double>{-3.3, 4.3}), std::asin(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::asin(std::complex<double>{-9.3, 10.3}), std::asin(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::asin(std::complex<int>{3, -4})));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::asin(std::integral_constant<int, 1>{}))>, pi/2, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::asin(values::Fixed<double, 1>{}))>, pi/2, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::asin(values::Fixed<std::complex<double>, 3, 4>{})))>, 0.6339838656391767163188, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::asin(values::Fixed<std::complex<double>, 3, 4>{})))>, 2.305509031243476942042, 1e-9));
}


#include "values/math/acos.hpp"

TEST(values, acos)
{
  constexpr auto pi = stdcompat::numbers::pi_v<double>;
  constexpr auto piL = stdcompat::numbers::pi_v<long double>;
  constexpr auto piF = stdcompat::numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(values::acos(values::internal::NaN<double>()) != values::acos(values::internal::NaN<double>()));
    EXPECT_TRUE(values::acos(-2) != values::acos(-2)); // NaN
    EXPECT_FALSE(std::signbit(values::cos(1)));
    EXPECT_TRUE(values::acos(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::acos(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::acos(0) == pi/2);
  static_assert(values::acos(1) == 0);
  static_assert(values::acos(-1) == pi);
  static_assert(values::acos(-1.0L) == piL);
  static_assert(values::acos(-1.0F) == piF);
  static_assert(values::internal::near(values::acos(0.5), stdcompat::numbers::pi/3));
  static_assert(values::internal::near(values::acos(-0.5), 2*stdcompat::numbers::pi/3));
  static_assert(values::internal::near(values::acos(stdcompat::numbers::sqrt2_v<double>/2), pi/4));
  static_assert(values::internal::near(values::acos(-stdcompat::numbers::sqrt2_v<double>/2), 3*pi/4));
  EXPECT_NEAR(values::acos(-0.7), std::acos(-0.7), 1e-9);
  EXPECT_NEAR(values::acos(0.9), std::acos(0.9), 1e-9);
  EXPECT_NEAR(values::acos(0.999), std::acos(0.999), 1e-9);
  EXPECT_NEAR(values::acos(-0.999), std::acos(-0.999), 1e-9);
  EXPECT_NEAR(values::acos(0.99999), std::acos(0.99999), 1e-9);
  EXPECT_NEAR(values::acos(0.9999999), std::acos(0.9999999), 1e-9);

  static_assert(values::internal::near(values::acos(std::complex<double>{0.5, 0}), pi/3, 1e-9));
  EXPECT_PRED3(tolerance, values::acos(std::complex<double>{4.1, 3.1}), std::acos(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, values::acos(std::complex<double>{3.2, -4.2}), std::acos(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, values::acos(std::complex<double>{-3.3, 4.3}), std::acos(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::acos(std::complex<double>{-9.3, 10.3}), std::acos(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::acos(std::complex<int>{3, -4})));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::acos(std::integral_constant<int, -1>{}))>, pi, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::acos(values::Fixed<double, -1>{}))>, pi, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::acos(values::Fixed<std::complex<double>, 3, 4>{})))>, 0.9368124611557199029125, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::acos(values::Fixed<std::complex<double>, 3, 4>{})))>, -2.305509031243476942042, 1e-9));
}


#include "values/math/atan.hpp"

TEST(values, atan)
{
  constexpr auto pi = stdcompat::numbers::pi_v<double>;
  constexpr auto piL = stdcompat::numbers::pi_v<long double>;
  constexpr auto piF = stdcompat::numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_TRUE(values::atan(values::internal::NaN<double>()) != values::atan(values::internal::NaN<double>()));
    EXPECT_DOUBLE_EQ(values::atan(values::internal::infinity<double>()), pi/2);
    EXPECT_DOUBLE_EQ(values::atan(-values::internal::infinity<double>()), -pi/2);
    EXPECT_TRUE(std::signbit(values::atan(-0.)));
    EXPECT_FALSE(std::signbit(values::atan(+0.)));
    EXPECT_TRUE(values::atan(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::atan(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::atan(0) == 0);
  static_assert(values::internal::near(values::atan(1.), pi/4));
  static_assert(values::internal::near(values::atan(-1.), -pi/4));
  static_assert(values::internal::near(values::atan(-1.L), -piL/4));
  static_assert(values::internal::near(values::atan(-1.F), -piF/4));
  EXPECT_NEAR(values::atan(-0.7), std::atan(-0.7), 1e-9);
  EXPECT_NEAR(values::atan(0.9), std::atan(0.9), 1e-9);
  EXPECT_NEAR(values::atan(5.0), std::atan(5.0), 1e-9);
  EXPECT_NEAR(values::atan(-10.0), std::atan(-10.0), 1e-9);
  EXPECT_NEAR(values::atan(100.0), std::atan(100.0), 1e-9);

  static_assert(values::internal::near(values::atan(std::complex<double>{1, 0}), pi/4, 1e-9));
  EXPECT_PRED3(tolerance, values::atan(std::complex<double>{4.1, 0.}), std::atan(4.1), 1e-9);
  EXPECT_PRED3(tolerance, values::atan(std::complex<double>{4.1, 3.1}), std::atan(std::complex<double>{4.1, 3.1}), 1e-9);
  EXPECT_PRED3(tolerance, values::atan(std::complex<double>{3.2, -4.2}), std::atan(std::complex<double>{3.2, -4.2}), 1e-9);
  EXPECT_PRED3(tolerance, values::atan(std::complex<double>{-3.3, 4.3}), std::atan(std::complex<double>{-3.3, 4.3}), 1e-9);
  EXPECT_PRED3(tolerance, values::atan(std::complex<double>{-9.3, 10.3}), std::atan(std::complex<double>{-9.3, 10.3}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::atan(std::complex<int>{3, -4})));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::atan(std::integral_constant<int, 1>{}))>, pi/4, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::atan(values::Fixed<double, 1>{}))>, pi/4, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::atan(values::Fixed<std::complex<double>, 3, 4>{})))>, 1.448306995231464542145, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::atan(values::Fixed<std::complex<double>, 3, 4>{})))>, 0.1589971916799991743648, 1e-9));
}


#include "values/math/atan2.hpp"

TEST(values, atan2)
{
  constexpr auto pi = stdcompat::numbers::pi_v<double>;
  constexpr auto piL = stdcompat::numbers::pi_v<long double>;
  constexpr auto piF = stdcompat::numbers::pi_v<float>;

  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_DOUBLE_EQ(values::atan2(values::internal::infinity<double>(), 0.f), pi/2);
    EXPECT_DOUBLE_EQ(values::atan2(-values::internal::infinity<double>(), 0.f), -pi/2);
    EXPECT_DOUBLE_EQ(values::atan2(+0., +values::internal::infinity<double>()), 0);
    EXPECT_DOUBLE_EQ(values::atan2(+0., -values::internal::infinity<double>()), pi);
    EXPECT_DOUBLE_EQ(values::atan2(-0., +values::internal::infinity<double>()), -0.);
    EXPECT_FALSE(std::signbit(values::atan2(+0., values::internal::infinity<double>())));
    EXPECT_DOUBLE_EQ(values::atan2(values::internal::infinity<double>(), values::internal::infinity<double>()), pi/4);
    EXPECT_DOUBLE_EQ(values::atan2(values::internal::infinity<double>(), -values::internal::infinity<double>()), 3*pi/4);
    EXPECT_DOUBLE_EQ(values::atan2(-values::internal::infinity<double>(), values::internal::infinity<double>()), -pi/4);
    EXPECT_DOUBLE_EQ(values::atan2(-values::internal::infinity<double>(), -values::internal::infinity<double>()), -3*pi/4);
#ifdef __cpp_lib_constexpr_cmath
    EXPECT_DOUBLE_EQ(values::atan2(-0., -values::internal::infinity<double>()), -pi);
    EXPECT_TRUE(std::signbit(values::atan2(-0., values::internal::infinity<double>())));
    static_assert(std::signbit(values::atan2(-0., +0.)));
    static_assert(not std::signbit(values::atan2(+0., +0.)));
    static_assert(values::atan2(-0., -0.) == -pi);
    static_assert(values::atan2(-0., -1.) == -pi);
    static_assert(values::atan2(+0., -0.) == pi);
    static_assert(values::atan2(+0., -1.) == pi);
#endif
    //EXPECT_TRUE(std::signbit(values::atan2(-0., +0.))); // This will be inacurate prior to c++23.
    EXPECT_FALSE(std::signbit(values::atan2(+0., +0.)));
    //EXPECT_EQ(values::atan2(-0., -0.), -pi); // This will be inacurate prior to c++23.
    //EXPECT_EQ(values::atan2(-0., -1.), -pi); // This will be inacurate prior to c++23.
    //EXPECT_EQ(values::atan2(+0., -0.), pi); // This will be inacurate prior to c++23.
    EXPECT_EQ(values::atan2(+0., -1.), pi);

    EXPECT_TRUE(values::atan2(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}, std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::atan2(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}, std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::atan2(0, 1) == 0);
  static_assert(values::atan2(0, -1) == pi);
  static_assert(values::atan2(1, 0) == pi/2);
  static_assert(values::atan2(-1, 0) == -pi/2);
  static_assert(values::internal::near(values::atan2(0.5, 0.5), pi/4));
  static_assert(values::internal::near(values::atan2(1., -1.), 3*pi/4));
  static_assert(values::internal::near(values::atan2(-0.5, 0.5), -pi/4));
  static_assert(values::internal::near(values::atan2(-1.L, -1.L), -3*piL/4));
  static_assert(values::internal::near(values::atan2(-1.F, -1.F), -3*piF/4));
  EXPECT_NEAR(values::atan2(-0.7, 4.5), std::atan2(-0.7, 4.5), 1e-9);
  EXPECT_NEAR(values::atan2(0.9, -2.3), std::atan2(0.9, -2.3), 1e-9);
  EXPECT_NEAR(values::atan2(5.0, 3.1), std::atan2(5.0, 3.1), 1e-9);
  EXPECT_NEAR(values::atan2(-10.0, 9.0), std::atan2(-10.0, 9.0), 1e-9);
  EXPECT_NEAR(values::atan2(100.0, 200.0), std::atan2(100.0, 200.0), 1e-9);

  static_assert(values::atan2(std::complex<double>{0, 0}, std::complex<double>{0, 0}) == 0.0);
  static_assert(values::atan2(std::complex<double>{0, 0}, std::complex<double>{1, 0}) == 0.0);
  static_assert(values::internal::near(values::atan2(std::complex<double>{0, 0}, std::complex<double>{-1, 0}), pi, 1e-9));
  static_assert(values::internal::near(values::atan2(std::complex<double>{1, 0}, std::complex<double>{0, 0}), pi/2, 1e-9));
  static_assert(values::internal::near(values::atan2(std::complex<double>{-1, 0}, std::complex<double>{0, 0}), -pi/2, 1e-9));
  static_assert(values::internal::near(values::real(values::atan2(std::complex<double>{3.2, -4.2}, std::complex<double>{-4.1, 3.1})), -0.7993578098204363309621, 1e-9));
  static_assert(values::internal::near(values::imag(values::atan2(std::complex<double>{3.2, -4.2}, std::complex<double>{-4.1, 3.1})), 0.1378262475816170392786, 1e-9));
  EXPECT_PRED3(tolerance, values::atan2(std::complex<double>{4.1, 3.1}, std::complex<double>{2.1, 5.1}), std::atan(std::complex<double>{4.1, 3.1} / std::complex<double>{2.1, 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, values::atan2(std::complex<double>{-3.3, 4.3}, std::complex<double>{2.1, 5.1}), std::atan(std::complex<double>{-3.3, 4.3} / std::complex<double>{2.1, 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, values::atan2(std::complex<double>{-9.3, 10.3}, std::complex<double>{-5.1, 2.1}), std::atan(std::complex<double>{-9.3, 10.3} / std::complex<double>{-5.1, 2.1}), 1e-9);
  EXPECT_PRED3(tolerance, values::atan2(std::complex<double>{0., 0.}, std::complex<double>{0., 0.}), std::complex<double>{0}, 1e-9);
  EXPECT_PRED3(tolerance, values::atan2(std::complex<double>{0., 3.1}, std::complex<double>{2.1, 5.1}), std::atan(std::complex<double>{0., 3.1} / std::complex<double>{2.1, 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, values::atan2(std::complex<double>{0., 3.1}, std::complex<double>{-2.1, 5.1}), std::atan(std::complex<double>{0., 3.1} / std::complex<double>{-2.1, 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, values::atan2(std::complex<double>{4.1, 3.1}, std::complex<double>{0., 5.1}), std::atan(std::complex<double>{4.1, 3.1} / std::complex<double>{0., 5.1}), 1e-9);
  EXPECT_PRED3(tolerance, values::atan2(std::complex<double>{-4.1, 3.1}, std::complex<double>{0., 5.1}), std::atan(std::complex<double>{-4.1, 3.1} / std::complex<double>{0., 5.1}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::atan2(std::complex<int>{3, -4}, std::complex<int>{2, 5})));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::atan2(std::integral_constant<int, 1>{}, std::integral_constant<int, 0>{}))>, pi/2, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::atan2(values::Fixed<double, 1>{}, values::Fixed<double, 0>{}))>, pi/2, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::real(values::atan2(values::Fixed<std::complex<double>, 3, 4>{}, values::Fixed<std::complex<double>, 5, 2>{})))>, 0.7420289940594557537102, 1e-9));
  static_assert(values::internal::near(values::fixed_number_of_v<decltype(values::imag(values::atan2(values::Fixed<std::complex<double>, 3, 4>{}, values::Fixed<std::complex<double>, 5, 2>{})))>, 0.2871556773106927669533, 1e-9));
}


#include "values/math/pow.hpp"

TEST(values, pow)
{
  if constexpr (std::numeric_limits<double>::is_iec559)
  {
    EXPECT_FALSE(std::signbit(values::pow(+0., 3U)));
    EXPECT_TRUE(std::signbit(values::pow(-0., 3U)));
    EXPECT_FALSE(std::signbit(values::pow(+0., 2U)));
    EXPECT_FALSE(std::signbit(values::pow(-0., 2U)));
    EXPECT_TRUE(values::pow(values::internal::NaN<double>(), 0U) == 1);
    EXPECT_TRUE(values::pow(values::internal::NaN<double>(), 1U) != values::pow(values::internal::NaN<double>(), 1U));
    EXPECT_TRUE(values::pow(-values::internal::infinity<double>(), 3U) == -values::internal::infinity<double>());
    EXPECT_TRUE(values::pow(-values::internal::infinity<double>(), 4U) == values::internal::infinity<double>());
    EXPECT_TRUE(values::pow(values::internal::infinity<double>(), 3U) == values::internal::infinity<double>());

    EXPECT_TRUE(values::pow(+0., -3) == values::internal::infinity<double>());
#ifdef __cpp_lib_constexpr_cmath
    EXPECT_TRUE(values::pow(-0., -3) == -values::internal::infinity<double>());
#endif
    EXPECT_TRUE(values::pow(+0., -2) == values::internal::infinity<double>());
    EXPECT_TRUE(values::pow(-0., -2) == values::internal::infinity<double>());
    EXPECT_FALSE(std::signbit(values::pow(+0., 3)));
    EXPECT_TRUE(std::signbit(values::pow(-0., 3)));
    EXPECT_FALSE(std::signbit(values::pow(+0., 2)));
    EXPECT_FALSE(std::signbit(values::pow(-0., 2)));
    EXPECT_TRUE(values::pow(values::internal::NaN<double>(), 0) == 1);
    EXPECT_TRUE(values::pow(values::internal::NaN<double>(), 1) != values::pow(values::internal::NaN<double>(), 1));
    EXPECT_TRUE(values::pow(-values::internal::infinity<double>(), -3) == 0);
    EXPECT_TRUE(std::signbit(values::pow(-values::internal::infinity<double>(), -3)));
    EXPECT_TRUE(values::pow(-values::internal::infinity<double>(), -2) == 0);
    EXPECT_FALSE(std::signbit(values::pow(-values::internal::infinity<double>(), -2)));
    EXPECT_TRUE(values::pow(-values::internal::infinity<double>(), 3) == -values::internal::infinity<double>());
    EXPECT_TRUE(values::pow(-values::internal::infinity<double>(), 4) == values::internal::infinity<double>());
    EXPECT_TRUE(values::pow(values::internal::infinity<double>(), -3) == 0);
    EXPECT_FALSE(std::signbit(values::pow(values::internal::infinity<double>(), -3)));
    EXPECT_TRUE(values::pow(values::internal::infinity<double>(), 3) == values::internal::infinity<double>());

    EXPECT_TRUE(values::pow(+0., -values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(values::pow(-0., -values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(values::pow(+0.5, -values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(values::pow(-0.5, -values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(values::pow(+1.5, -values::internal::infinity<double>()) == 0);
    EXPECT_TRUE(values::pow(-1.5, -values::internal::infinity<double>()) == 0);
    EXPECT_FALSE(std::signbit(values::pow(+1.5, -values::internal::infinity<double>())));
    EXPECT_FALSE(std::signbit(values::pow(-1.5, -values::internal::infinity<double>())));
    EXPECT_TRUE(values::pow(+0.5, values::internal::infinity<double>()) == 0);
    EXPECT_TRUE(values::pow(-0.5, values::internal::infinity<double>()) == 0);
    EXPECT_FALSE(std::signbit(values::pow(+0.5, values::internal::infinity<double>())));
    EXPECT_FALSE(std::signbit(values::pow(-0.5, values::internal::infinity<double>())));
    EXPECT_TRUE(values::pow(+1.5, values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(values::pow(-1.5, values::internal::infinity<double>()) == values::internal::infinity<double>());
    EXPECT_TRUE(values::pow(-values::internal::infinity<double>(), -3.) == 0);

    EXPECT_TRUE(values::pow(values::internal::infinity<double>(), -3.) == 0);
    EXPECT_FALSE(std::signbit(values::pow(-values::internal::infinity<double>(), -3.)));
    EXPECT_EQ(values::pow(-values::internal::infinity<double>(), 3.), values::internal::infinity<double>());
    EXPECT_FALSE(std::signbit(values::pow(values::internal::infinity<double>(), -3.)));
    EXPECT_TRUE(values::pow(values::internal::infinity<double>(), 3.) == values::internal::infinity<double>());

    EXPECT_FALSE(std::signbit(values::pow(+0, 3.)));
    EXPECT_FALSE(std::signbit(values::pow(-0, 3.)));
    EXPECT_TRUE(values::pow(-1., values::internal::infinity<double>()) == 1);
    EXPECT_TRUE(values::pow(-1., -values::internal::infinity<double>()) == 1);
    EXPECT_TRUE(values::pow(+1., values::internal::NaN<double>()) == 1);
    EXPECT_TRUE(values::pow(values::internal::NaN<double>(), +0) == 1);
    EXPECT_TRUE(values::pow(values::internal::NaN<double>(), 1.) != values::pow(values::internal::NaN<double>(), 1.));

    EXPECT_TRUE(values::pow(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}, std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}) != values::pow(std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}, std::complex<double>{values::internal::NaN<double>(), values::internal::NaN<double>()}));
  }

  static_assert(values::pow(+0., 3U) == 0);
  static_assert(values::pow(-0., 3U) == 0);
  static_assert(values::pow(+0., 2U) == 0);
  static_assert(values::pow(-0., 2U) == 0);
  static_assert(values::pow(1, 0U) == 1);
  static_assert(values::pow(0, 1U) == 0);
  static_assert(values::pow(1, 1U) == 1);
  static_assert(values::pow(1, 2U) == 1);
  static_assert(values::pow(2, 1U) == 2);
  static_assert(values::pow(2, 5U) == 32);
  static_assert(values::pow(2, 16U) == 65536);
  static_assert(std::is_floating_point_v<decltype(values::pow(2, 16U))>);
  static_assert(values::pow(2.0, 16U) == 65536);
  static_assert(std::is_floating_point_v<decltype(values::pow(2.0, 16U))>);

  static_assert(values::pow(+0., 3) == 0);
  static_assert(values::pow(-0., 3) == 0);
  static_assert(values::pow(+0., 2) == 0);
  static_assert(values::pow(-0., 2) == 0);
  static_assert(values::pow(2, -4) == 0.0625);
  static_assert(values::pow(2, -5) == 0.03125);
  static_assert(std::is_floating_point_v<decltype(values::pow(2, -4))>);

  static_assert(values::pow(+0., 3.) == +0);
  static_assert(values::pow(-0., 3.) == +0);
  static_assert(values::pow(+1., 5) == 1);
  static_assert(values::pow(-5., +0) == 1);
  EXPECT_TRUE(values::pow(-5., 1.5) != values::pow(-5., 1.5));
  EXPECT_TRUE(values::pow(-7.3, 3.3) != values::pow(-7.3, 3.3));
  static_assert(values::internal::near(values::pow(2, -4.), 0.0625));
  static_assert(values::internal::near(values::pow(10, -4.), 1e-4));
  static_assert(values::internal::near(values::pow(10., 6.), 1e6, 1e-4));
  EXPECT_DOUBLE_EQ(values::pow(5.0L, 4.0L), std::pow(5.0L, 4.0L));
  EXPECT_DOUBLE_EQ(values::pow(5.0L, -4.0L), std::pow(5.0L, -4.0L));
  EXPECT_DOUBLE_EQ(values::pow(1e20L, 2.L), std::pow(1e20L, 2.L));
  EXPECT_DOUBLE_EQ(values::pow(1e20L, -2.L), std::pow(1e20L, -2.L));
  EXPECT_DOUBLE_EQ(values::pow(1e100L, 2.L), std::pow(1e100L, 2.L));
  EXPECT_DOUBLE_EQ(values::pow(1e100L, -2.L), std::pow(1e100L, -2.L));

  EXPECT_PRED3(tolerance, values::pow(2., std::complex<double>{-4}), std::pow(2., std::complex<double>{-4}), 1e-9);
  EXPECT_PRED3(tolerance, values::pow(2, std::complex<double>{-4}), std::pow(2, std::complex<double>{-4}), 1e-9);
  EXPECT_PRED3(tolerance, values::pow(std::complex<double>{3, 4}, 2.), std::pow(std::complex<double>{3, 4}, 2.), 1e-9);
  EXPECT_PRED3(tolerance, values::pow(std::complex<double>{3, 4}, 2), std::pow(std::complex<double>{3, 4}, 2), 1e-9);
  EXPECT_PRED3(tolerance, values::pow(std::complex<double>{3, 4}, -2), std::pow(std::complex<double>{3, 4}, -2), 1e-9);
  EXPECT_PRED3(tolerance, values::pow(std::complex<double>{3, 4}, 3), std::pow(std::complex<double>{3, 4}, 3), 1e-9);
  EXPECT_PRED3(tolerance, values::pow(std::complex<double>{3, 4}, -3), std::pow(std::complex<double>{3, 4}, -3), 1e-9);
  EXPECT_PRED3(tolerance, values::pow(2., std::complex<double>{3, -4}), std::pow(2., std::complex<double>{3, -4}), 1e-9);
  EXPECT_PRED3(tolerance, values::pow(std::complex<double>{1, 2}, std::complex<double>{-3, 4}), std::pow(std::complex<double>{1, 2}, std::complex<double>{-3, 4}), 1e-9);
  EXPECT_PRED3(tolerance, values::pow(std::complex<double>{-3, -4}, std::complex<double>{1, 2}), std::pow(std::complex<double>{-3, -4}, std::complex<double>{1, 2}), 1e-9);
  COMPLEXINTEXISTS(EXPECT_NO_THROW(values::pow(std::complex<int>{-3, -4}, std::complex<int>{1, 2})));

  static_assert(values::pow(values::Fixed<double, 2>{}, 3) == 8);
  static_assert(values::internal::near(values::pow(2, values::Fixed<double, 3>{}), 8, 1e-6));
  static_assert(values::fixed_number_of_v<decltype(values::pow(values::Fixed<double, 2>{}, std::integral_constant<int, 3>{}))> == 8);
}
