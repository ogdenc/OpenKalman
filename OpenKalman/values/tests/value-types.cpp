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
  static_assert(values::number<int>);
  static_assert(values::number<double>);
  static_assert(not values::number<std::integral_constant<std::size_t, 3>>);
}

#include "values/concepts/floating.hpp"

TEST(values, floating_number)
{
  static_assert(values::floating<float>);
  static_assert(values::floating<double>);
  static_assert(values::floating<long double>);
  static_assert(not values::floating<int>);
  static_assert(not values::floating<std::integral_constant<std::size_t, 3>>);
}

#include "values/concepts/fixed.hpp"

TEST(values, fixed)
{
  static_assert(values::fixed<std::integral_constant<std::size_t, 3>>);
  static_assert(values::fixed<std::integral_constant<int, 5>>);
  struct return8 { constexpr auto operator()() { return 8; } };
  static_assert(values::fixed<return8>);
#ifdef __cpp_concepts
  constexpr auto f8 = []{ return 8; };
  static_assert(values::fixed<decltype(f8)>);
  constexpr auto f8d = []{ return 8.; };
  static_assert(values::fixed<decltype(f8d)>);
#endif
}

#include "values/concepts/dynamic.hpp"

TEST(values, dynamic)
{
  static_assert(values::dynamic<int>);
  static_assert(values::dynamic<double>);
  struct return8r { auto operator()() { return 8; } };
  static_assert(values::dynamic<return8r>);
}

#include "values/concepts/value.hpp"

TEST(values, scalar)
{
  static_assert(values::value<double>);
  static_assert(values::value<std::integral_constant<int, 6>>);
}

#include "values/functions/to_number.hpp"

TEST(values, to_number)
{

  EXPECT_EQ(values::to_number(7), 7);
  static_assert(values::to_number(std::integral_constant<int, 7>{}) == 7);
  static_assert(values::to_number([]{ return 8; }) == 8);
}

#include "values/traits/fixed_number_of.hpp"

TEST(values, fixed_number_of)
{
  static_assert(values::fixed_number_of_v<std::integral_constant<int, 7>> == 7);
  struct return8 { constexpr auto operator()() { return 8; } };
  static_assert(values::fixed_number_of_v<return8> == 8);
#ifdef __cpp_concepts
  constexpr auto f8 = []{ return 8; };
  static_assert(values::fixed_number_of_v<decltype(f8)> == 8);
  constexpr auto f8d = []{ return 8.; };
  static_assert(values::fixed_number_of_v<decltype(f8d)> == 8.);
#endif
}

#include "values/classes/operation.hpp"

TEST(values, operation)
{
  struct NullaryFunc { constexpr auto operator()() { return 5.5; } };

  static_assert(values::fixed<values::operation<NullaryFunc>>);

  static_assert(values::dynamic<values::operation<std::negate<>, double>>);
  static_assert(values::dynamic<values::operation<std::multiplies<>, double, double>>);
  static_assert(values::operation{std::plus{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 5>{}}() == 9);

  static_assert(values::to_number(values::operation<NullaryFunc>{}) == 5.5);
  static_assert(values::to_number(values::operation{std::plus{}, 4, 5}) == 9);
  static_assert(values::to_number(values::operation{[](){ return 4 + 5; }}) == 9);
  int k = 9; EXPECT_EQ(values::to_number(values::operation{[&k]{ return k; }}), 9);
}

#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/classes/Fixed.hpp"

TEST(values, Fixed)
{
  static_assert(not values::number<values::Fixed<double, 3>>);

  static_assert(values::floating<values::Fixed<double, 3>>);

  static_assert(values::fixed<values::Fixed<double, 3>>);
  static_assert(values::fixed<decltype(values::Fixed{values::Fixed<double, 7>{}})>);

  static_assert(values::to_number(values::Fixed<double, 3>{}) == 3);
  static_assert(values::Fixed<double, 3>{}() == 3);
  static_assert(values::Fixed<std::integral_constant<int, 7>>{}() == 7);
  static_assert(decltype(values::Fixed{std::integral_constant<int, 7>{}})::value == 7);
  static_assert(values::Fixed{values::Fixed<double, 7>{}} == 7);
  static_assert(std::is_same_v<decltype(values::Fixed{std::integral_constant<int, 7>{}})::value_type, int>);
  static_assert(std::is_same_v<values::Fixed<int, 3>::value_type, int>);
  static_assert(std::is_same_v<values::Fixed<double, 3>::value_type, double>);

  static_assert(std::is_same_v<values::number_type_of_t<values::Fixed<double, 3>>, double>);
  static_assert(std::is_same_v<values::number_type_of_t<values::real_type_of_t<values::Fixed<double, 3>>>, double>);
}

#include "values/functions/value-arithmetic.hpp"

TEST(values, scalar_arithmetic)
{
  static_assert(std::decay_t<decltype(+values::Fixed<double, 3>{})>::value == 3);
  static_assert(std::decay_t<decltype(-values::Fixed<double, 3>{})>::value == -3);
  static_assert(std::decay_t<decltype(values::Fixed<double, 3>{} + std::integral_constant<int, 2>{})>::value == 5);
  static_assert(std::decay_t<decltype(values::Fixed<double, 3>{} - std::integral_constant<int, 2>{})>::value == 1);
  static_assert(std::decay_t<decltype(values::Fixed<double, 3>{} * std::integral_constant<int, 2>{})>::value == 6);
  static_assert(std::decay_t<decltype(values::Fixed<double, 3>{} / std::integral_constant<int, 2>{})>::value == 1.5);

  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value + values::Fixed<double, 3>{} == 5);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value - values::Fixed<double, 3>{} == -1);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value * values::Fixed<double, 3>{} == 6);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 3>{})>::value / values::Fixed<double, 2>{} == 1.5);

  static_assert(values::Fixed<double, 3>{} == values::Fixed<double, 3>{});
  static_assert(values::Fixed<double, 3>{} == std::integral_constant<int, 3>{});
  static_assert(values::Fixed<double, 3>{} != std::integral_constant<int, 4>{});
  static_assert(values::Fixed<double, 3>{} < std::integral_constant<int, 4>{});
  static_assert(std::integral_constant<int, 4>{} <= values::Fixed<double, 7>{});
  static_assert(values::Fixed<double, 8>{} > std::integral_constant<int, 4>{});
  static_assert(std::integral_constant<int, 9>{} >= values::Fixed<double, 7>{});

  auto sc3 = values::operation{std::minus{}, values::Fixed<double, 7>{}, std::integral_constant<int, 4>{}};

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
  static_assert(values::cast_to<double>(std::integral_constant<int, 4>{}) == 4);
  static_assert(std::is_same_v<decltype(values::cast_to<double>(std::integral_constant<int, 4>{}))::value_type, double>);
  static_assert(std::is_same_v<decltype(values::cast_to<int>(std::integral_constant<int, 4>{})), std::integral_constant<int, 4>&&>);
  static_assert(values::cast_to<double>(values::Fixed<float, 4>{}) == 4);
  static_assert(std::is_same_v<decltype(values::cast_to<double>(values::Fixed<double, 4>{})), values::Fixed<double, 4>&&>);
  static_assert(std::is_same_v<typename values::fixed_number_of<decltype(values::cast_to<double>(values::Fixed<double, 4>{}))>::value_type, double>);
  static_assert(std::is_same_v<decltype(values::cast_to<double>(values::Fixed<int, 4>{}))::value_type, double>);
  static_assert(values::cast_to<double>(values::Fixed<float, 4>{}) == 4);
  static_assert(std::is_same_v<values::number_type_of_t<decltype(values::cast_to<double>(values::Fixed<float, 4>{}))>, double>);
}

#include "values/functions/internal/near.hpp"

TEST(values, near)
{
  static_assert(values::internal::near(10., 10.));
  static_assert(values::internal::near<2>(0., 0. + std::numeric_limits<double>::epsilon()));
  static_assert(not values::internal::near<2>(0., 0. + 3 * std::numeric_limits<double>::epsilon()));

  static_assert(values::internal::near(values::Fixed<double, 4>{}, values::Fixed<double, 5>{}, 2));
  static_assert(values::internal::near(std::integral_constant<int, 4>{}, std::integral_constant<int, 5>{}, 2));
  static_assert(values::internal::near(std::integral_constant<int, 4>{}, 5, 2));
  static_assert(not values::internal::near(std::integral_constant<int, 4>{}, 6, 1));
  static_assert(values::internal::near(values::Fixed<std::complex<double>, 3, 4>{}, values::Fixed<std::complex<double>, 4, 5>{}, 2));
  static_assert(values::internal::near(values::Fixed<std::complex<double>, 3, 4>{}, values::Fixed<std::complex<double>, 4, 5>{}, values::Fixed<std::complex<double>, 2, 2>{}));
  static_assert(values::internal::near(std::integral_constant<int, 4>{}, values::Fixed<std::complex<double>, 4, 1>{}, 2));
}

