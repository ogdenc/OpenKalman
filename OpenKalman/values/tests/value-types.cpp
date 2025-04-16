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
#include "values/tests/tests.hpp"

using namespace OpenKalman;

#include "values/concepts/number.hpp"

TEST(values, number)
{
  static_assert(value::number<int>);
  static_assert(value::number<double>);
  static_assert(not value::number<std::integral_constant<std::size_t, 3>>);
}

#include "values/concepts/floating.hpp"

TEST(values, floating_number)
{
  static_assert(value::floating<float>);
  static_assert(value::floating<double>);
  static_assert(value::floating<long double>);
  static_assert(not value::floating<int>);
  static_assert(not value::floating<std::integral_constant<std::size_t, 3>>);
}

#include "values/concepts/fixed.hpp"

TEST(values, fixed)
{
  static_assert(value::fixed<std::integral_constant<std::size_t, 3>>);
  static_assert(value::fixed<std::integral_constant<int, 5>>);
  struct return8 { constexpr auto operator()() { return 8; } };
  static_assert(value::fixed<return8>);
#ifdef __cpp_concepts
  constexpr auto f8 = []{ return 8; };
  static_assert(value::fixed<decltype(f8)>);
  constexpr auto f8d = []{ return 8.; };
  static_assert(value::fixed<decltype(f8d)>);
#endif
}

#include "values/concepts/dynamic.hpp"

TEST(values, dynamic)
{
  static_assert(value::dynamic<int>);
  static_assert(value::dynamic<double>);
  struct return8r { auto operator()() { return 8; } };
  static_assert(value::dynamic<return8r>);
}

#include "values/concepts/value.hpp"

TEST(values, scalar)
{
  static_assert(value::value<double>);
  static_assert(value::value<std::integral_constant<int, 6>>);
}

#include "values/functions/to_number.hpp"

TEST(values, to_number)
{

  EXPECT_EQ(value::to_number(7), 7);
  static_assert(value::to_number(std::integral_constant<int, 7>{}) == 7);
  static_assert(value::to_number([]{ return 8; }) == 8);
}

#include "values/traits/fixed_number_of.hpp"

TEST(values, fixed_number_of)
{
  static_assert(value::fixed_number_of_v<std::integral_constant<int, 7>> == 7);
  struct return8 { constexpr auto operator()() { return 8; } };
  static_assert(value::fixed_number_of_v<return8> == 8);
#ifdef __cpp_concepts
  constexpr auto f8 = []{ return 8; };
  static_assert(value::fixed_number_of_v<decltype(f8)> == 8);
  constexpr auto f8d = []{ return 8.; };
  static_assert(value::fixed_number_of_v<decltype(f8d)> == 8.);
#endif
}

#include "values/classes/operation.hpp"

TEST(values, operation)
{
  struct NullaryFunc { constexpr auto operator()() { return 5.5; } };

  static_assert(value::fixed<value::operation<NullaryFunc>>);

  static_assert(value::dynamic<value::operation<std::negate<>, double>>);
  static_assert(value::dynamic<value::operation<std::multiplies<>, double, double>>);
  static_assert(value::operation{std::plus{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 5>{}}() == 9);

  static_assert(value::to_number(value::operation<NullaryFunc>{}) == 5.5);
  static_assert(value::to_number(value::operation{std::plus{}, 4, 5}) == 9);
  static_assert(value::to_number(value::operation{[](){ return 4 + 5; }}) == 9);
  int k = 9; EXPECT_EQ(value::to_number(value::operation{[&k]{ return k; }}), 9);
}

#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/classes/Fixed.hpp"

TEST(values, Fixed)
{
  static_assert(not value::number<value::Fixed<double, 3>>);

  static_assert(value::floating<value::Fixed<double, 3>>);

  static_assert(value::fixed<value::Fixed<double, 3>>);
  static_assert(value::fixed<decltype(value::Fixed{value::Fixed<double, 7>{}})>);

  static_assert(value::to_number(value::Fixed<double, 3>{}) == 3);
  static_assert(value::Fixed<double, 3>{}() == 3);
  static_assert(value::Fixed<std::integral_constant<int, 7>>{}() == 7);
  static_assert(decltype(value::Fixed{std::integral_constant<int, 7>{}})::value == 7);
  static_assert(value::Fixed{value::Fixed<double, 7>{}} == 7);
  static_assert(std::is_same_v<decltype(value::Fixed{std::integral_constant<int, 7>{}})::value_type, int>);
  static_assert(std::is_same_v<value::Fixed<int, 3>::value_type, int>);
  static_assert(std::is_same_v<value::Fixed<double, 3>::value_type, double>);

  static_assert(std::is_same_v<value::number_type_of_t<value::Fixed<double, 3>>, double>);
  static_assert(std::is_same_v<value::number_type_of_t<value::real_type_of_t<value::Fixed<double, 3>>>, double>);
}

#include "values/functions/value-arithmetic.hpp"

TEST(values, scalar_arithmetic)
{
  static_assert(std::decay_t<decltype(+value::Fixed<double, 3>{})>::value == 3);
  static_assert(std::decay_t<decltype(-value::Fixed<double, 3>{})>::value == -3);
  static_assert(std::decay_t<decltype(value::Fixed<double, 3>{} + std::integral_constant<int, 2>{})>::value == 5);
  static_assert(std::decay_t<decltype(value::Fixed<double, 3>{} - std::integral_constant<int, 2>{})>::value == 1);
  static_assert(std::decay_t<decltype(value::Fixed<double, 3>{} * std::integral_constant<int, 2>{})>::value == 6);
  static_assert(std::decay_t<decltype(value::Fixed<double, 3>{} / std::integral_constant<int, 2>{})>::value == 1.5);

  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value + value::Fixed<double, 3>{} == 5);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value - value::Fixed<double, 3>{} == -1);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value * value::Fixed<double, 3>{} == 6);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 3>{})>::value / value::Fixed<double, 2>{} == 1.5);

  static_assert(value::Fixed<double, 3>{} == value::Fixed<double, 3>{});
  static_assert(value::Fixed<double, 3>{} == std::integral_constant<int, 3>{});
  static_assert(value::Fixed<double, 3>{} != std::integral_constant<int, 4>{});
  static_assert(value::Fixed<double, 3>{} < std::integral_constant<int, 4>{});
  static_assert(std::integral_constant<int, 4>{} <= value::Fixed<double, 7>{});
  static_assert(value::Fixed<double, 8>{} > std::integral_constant<int, 4>{});
  static_assert(std::integral_constant<int, 9>{} >= value::Fixed<double, 7>{});

  auto sc3 = value::operation{std::minus{}, value::Fixed<double, 7>{}, std::integral_constant<int, 4>{}};

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

#include "values/functions/cast_to.hpp"

TEST(values, cast_to)
{
  static_assert(value::cast_to<double>(std::integral_constant<int, 4>{}) == 4);
  static_assert(std::is_same_v<decltype(value::cast_to<double>(std::integral_constant<int, 4>{}))::value_type, double>);
  static_assert(std::is_same_v<decltype(value::cast_to<int>(std::integral_constant<int, 4>{})), std::integral_constant<int, 4>&&>);
  static_assert(value::cast_to<double>(value::Fixed<float, 4>{}) == 4);
  static_assert(std::is_same_v<decltype(value::cast_to<double>(value::Fixed<double, 4>{})), value::Fixed<double, 4>&&>);
  static_assert(std::is_same_v<typename value::fixed_number_of<decltype(value::cast_to<double>(value::Fixed<double, 4>{}))>::value_type, double>);
  static_assert(std::is_same_v<decltype(value::cast_to<double>(value::Fixed<int, 4>{}))::value_type, double>);
  static_assert(value::cast_to<double>(value::Fixed<float, 4>{}) == 4);
  static_assert(std::is_same_v<value::number_type_of_t<decltype(value::cast_to<double>(value::Fixed<float, 4>{}))>, double>);
}

#include "values/functions/internal/near.hpp"

TEST(values, near)
{
  static_assert(value::internal::near(10., 10.));
  static_assert(value::internal::near<2>(0., 0. + std::numeric_limits<double>::epsilon()));
  static_assert(not value::internal::near<2>(0., 0. + 3 * std::numeric_limits<double>::epsilon()));

  static_assert(value::internal::near(value::Fixed<double, 4>{}, value::Fixed<double, 5>{}, 2));
  static_assert(value::internal::near(std::integral_constant<int, 4>{}, std::integral_constant<int, 5>{}, 2));
  static_assert(value::internal::near(std::integral_constant<int, 4>{}, 5, 2));
  static_assert(not value::internal::near(std::integral_constant<int, 4>{}, 6, 1));
  static_assert(value::internal::near(value::Fixed<std::complex<double>, 3, 4>{}, value::Fixed<std::complex<double>, 4, 5>{}, 2));
  static_assert(value::internal::near(value::Fixed<std::complex<double>, 3, 4>{}, value::Fixed<std::complex<double>, 4, 5>{}, value::Fixed<std::complex<double>, 2, 2>{}));
  static_assert(value::internal::near(std::integral_constant<int, 4>{}, value::Fixed<std::complex<double>, 4, 1>{}, 2));
}

