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

#include <type_traits>
#include "basics/tests/tests.hpp"

using namespace OpenKalman;


#if defined(__GNUC__) or defined(__clang__)
#define COMPLEXINTEXISTS(F) F
#else
#define COMPLEXINTEXISTS(F)
#endif

#include "linear-algebra/values/concepts/complex_number.hpp"

TEST(values, complex)
{
  static_assert(value::complex_number<std::complex<double>>);
  static_assert(value::complex_number<std::complex<float>>);
  static_assert(not value::complex_number<double>);
  COMPLEXINTEXISTS(static_assert(value::complex_number<std::complex<int>>));
}

#include "linear-algebra/values/concepts/number.hpp"

TEST(values, number)
{
  static_assert(value::number<int>);
  static_assert(value::number<double>);
  static_assert(value::number<std::complex<double>>);
  static_assert(value::number<std::complex<float>>);
  static_assert(not value::number<std::integral_constant<std::size_t, 3>>);
  COMPLEXINTEXISTS(static_assert(value::number<std::complex<int>>));
}

#include "linear-algebra/values/concepts/floating_number.hpp"

TEST(values, floating_number)
{
  static_assert(value::floating_number<float>);
  static_assert(value::floating_number<double>);
  static_assert(value::floating_number<long double>);
  static_assert(not value::floating_number<int>);
  static_assert(not value::floating_number<std::complex<double>>);
  static_assert(not value::floating_number<std::complex<float>>);
  static_assert(not value::floating_number<std::integral_constant<std::size_t, 3>>);
  COMPLEXINTEXISTS(static_assert(not value::floating_number<std::complex<int>>));
}

#include "linear-algebra/values/concepts/static_scalar.hpp"

TEST(values, static_scalar)
{
  static_assert(value::static_scalar<std::integral_constant<std::size_t, 3>>);
  static_assert(value::static_scalar<std::integral_constant<int, 5>>);
  struct return8 { constexpr auto operator()() { return 8; } };
  static_assert(value::static_scalar<return8>);
}

#include "linear-algebra/values/concepts/dynamic_scalar.hpp"

TEST(values, dynamic_scalar)
{
  static_assert(value::dynamic_scalar<int>);
  static_assert(value::dynamic_scalar<double>);
  static_assert(value::dynamic_scalar<std::complex<double>>);
  static_assert(value::dynamic_scalar<std::complex<float>>);
  struct return8r { auto operator()() { return 8; } };
  static_assert(value::dynamic_scalar<return8r>);
}

#include "linear-algebra/values/concepts/scalar.hpp"

TEST(values, scalar)
{
  static_assert(value::scalar<double>);
  static_assert(value::scalar<std::integral_constant<int, 6>>);
}

#include "linear-algebra/values/functions/to_number.hpp"

TEST(values, to_number)
{

  EXPECT_EQ(value::to_number(7), 7);
  EXPECT_EQ(value::to_number(std::integral_constant<int, 7>{}), 7);
  EXPECT_EQ(value::to_number([](){ return 8; }), 8);
}

#include "linear-algebra/values/concepts/real_scalar.hpp"

TEST(values, real_scalar_constant)
{
  static_assert(value::real_scalar<int>);
  static_assert(value::real_scalar<double>);
  static_assert(not value::real_scalar<std::complex<double>>);
}

#include "linear-algebra/values/internal-classes/static_scalar_operation.hpp"

TEST(values, static_scalar_operation)
{
  struct NullaryFunc { constexpr auto operator()() { return 5.5; } };

  static_assert(value::static_scalar<value::static_scalar_operation<NullaryFunc>>);

  static_assert(value::dynamic_scalar<value::static_scalar_operation<std::negate<>, double>>);
  static_assert(value::dynamic_scalar<value::static_scalar_operation<std::multiplies<>, double, double>>);
  static_assert(value::static_scalar_operation{std::plus{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 5>{}}() == 9);

  static_assert(value::to_number(value::static_scalar_operation<NullaryFunc>{}) == 5.5);
  EXPECT_EQ(value::to_number(value::static_scalar_operation{[](){ return 9; }}), 9);
  int k = 9; EXPECT_EQ(value::to_number(value::static_scalar_operation{[&k](){ return k; }}), 9);
  EXPECT_EQ(value::to_number(value::static_scalar_operation{std::plus{}, 4, 5}), 9);
}

#include "linear-algebra/values/internal-classes/StaticScalar.hpp"

TEST(values, StaticScalar)
{
  static_assert(not value::complex_number<value::StaticScalar<std::complex<double>, 3, 0>>);

  static_assert(not value::number<value::StaticScalar<double, 3>>);

  static_assert(not value::floating_number<value::StaticScalar<double, 3>>);

  static_assert(value::static_scalar<value::StaticScalar<double, 3>>);

  static_assert(value::to_number(value::StaticScalar<double, 3>{}) == 3);
  static_assert(std::real(value::to_number(value::StaticScalar<std::complex<double>, 3, 4>{})) == 3);

  static_assert(value::real_scalar<value::StaticScalar<std::complex<double>, 3, 0>>);

  static_assert(value::real_scalar<value::StaticScalar<std::complex<double>, 3, 0>>);
  static_assert(not value::real_scalar<value::StaticScalar<std::complex<double>, 3, 1>>);
  static_assert(value::StaticScalar<double, 3>{}() == 3);
  static_assert(value::StaticScalar<std::complex<double>, 3, 4>{}() == std::complex<double>{3, 4});
  static_assert(value::StaticScalar<std::integral_constant<int, 7>>{}() == 7);
  static_assert(value::StaticScalar{std::integral_constant<int, 7>{}}.value == 7);
  static_assert(value::StaticScalar{3}() == 3);
  static_assert(value::StaticScalar{3.} == 3.);
  static_assert(std::is_same_v<decltype(value::StaticScalar{std::integral_constant<int, 7>{}})::value_type, int>);
  static_assert(std::is_same_v<decltype(value::StaticScalar{3})::value_type, int>);
  static_assert(std::is_same_v<decltype(value::StaticScalar{3.})::value_type, double>);
}

#include "linear-algebra/values/functions/internal/scalar-arithmetic.hpp"

TEST(values, scalar_arithmetic)
{
  static_assert(std::decay_t<decltype(+value::StaticScalar<double, 3>{})>::value == 3);
  static_assert(std::decay_t<decltype(-value::StaticScalar<double, 3>{})>::value == -3);
  static_assert(std::decay_t<decltype(value::StaticScalar<double, 3>{} + std::integral_constant<int, 2>{})>::value == 5);
  static_assert(std::decay_t<decltype(value::StaticScalar<double, 3>{} - std::integral_constant<int, 2>{})>::value == 1);
  static_assert(std::decay_t<decltype(value::StaticScalar<double, 3>{} * std::integral_constant<int, 2>{})>::value == 6);
  static_assert(std::decay_t<decltype(value::StaticScalar<double, 3>{} / std::integral_constant<int, 2>{})>::value == 1.5);

  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value + value::StaticScalar<double, 3>{} == 5);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value - value::StaticScalar<double, 3>{} == -1);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value * value::StaticScalar<double, 3>{} == 6);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 3>{})>::value / value::StaticScalar<double, 2>{} == 1.5);

  auto sc3 = value::static_scalar_operation{std::minus<>{}, value::StaticScalar<double, 7>{}, std::integral_constant<int, 4>{}};

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

#include "linear-algebra/values/functions/internal/make_complex_number.hpp"

TEST(values, make_complex_number)
{
  static_assert(std::real(internal::make_complex_number(3., 4.)) == 3);
  static_assert(std::imag(internal::make_complex_number(3., 4.)) == 4);
}

#include "linear-algebra/values/functions/internal/are_within_tolerance.hpp"

TEST(values, are_within_tolerance)
{
  static_assert(internal::are_within_tolerance(10., 10.));
  static_assert(internal::are_within_tolerance<2>(0., 0. + std::numeric_limits<double>::epsilon()));
  static_assert(not internal::are_within_tolerance<2>(0., 0. + 3 * std::numeric_limits<double>::epsilon()));
}

#include "linear-algebra/values/functions/internal/update_real_part.hpp"

TEST(values, update_real_part)
{
  static_assert(internal::update_real_part(std::complex{3.5, 4.5}, 5.5) == std::complex{5.5, 4.5});
  static_assert(internal::update_real_part(std::complex{3, 4}, 5.2) == std::complex{5, 4}); // truncation occurs
}

#include "linear-algebra/values/functions/internal/index_to_scalar_constant.hpp"

TEST(values, index_to_scalar_constant)
{
  static_assert(internal::index_to_scalar_constant<double>(std::integral_constant<int, 4>{}) == 4);
  static_assert(std::real(internal::index_to_scalar_constant<std::complex<double>>(3)) == 3);
}

