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

namespace
{
  struct return8 { constexpr auto operator()() { return 8; } };
  struct return8r { auto operator()() { return 8; } };
#ifdef __cpp_concepts
  constexpr auto f8 = []{ return 8; };
  constexpr auto f8d = []{ return 8.; };
#endif
}


#include "values/concepts/fixed.hpp"

TEST(values, fixed)
{
  static_assert(values::fixed<std::integral_constant<std::size_t, 3>>);
  static_assert(values::fixed<std::integral_constant<int, 5>>);
  static_assert(values::fixed<return8>);
  static_assert(not values::fixed<return8r>);
#ifdef __cpp_concepts
  static_assert(values::fixed<decltype(f8)>);
  static_assert(values::fixed<decltype(f8d)>);
#endif
  static_assert(not values::fixed<stdcompat::ranges::repeat_view<std::monostate>>);
}

#include "values/concepts/dynamic.hpp"

TEST(values, dynamic)
{
  static_assert(values::dynamic<int>);
  static_assert(values::dynamic<double>);
  static_assert(not values::dynamic<return8>);
  static_assert(values::dynamic<return8r>);
#ifdef __cpp_concepts
  static_assert(not values::dynamic<decltype(f8)>);
  static_assert(not values::dynamic<decltype(f8d)>);
#endif
  static_assert(values::dynamic<stdcompat::ranges::repeat_view<std::monostate>>);
}

#include "values/functions/to_value_type.hpp"

TEST(values, to_value_type)
{
  EXPECT_EQ(values::to_value_type(7), 7);
  static_assert(values::to_value_type(std::integral_constant<int, 7>{}) == 7);
  static_assert(values::to_value_type([]{ return 8; }) == 8);
}

#include "values/traits/fixed_value_of.hpp"

TEST(values, fixed_value_of)
{
  static_assert(values::fixed_value_of_v<std::integral_constant<int, 7>> == 7);
  struct return8 { constexpr auto operator()() { return 8; } };
  static_assert(values::fixed_value_of_v<return8> == 8);
#ifdef __cpp_concepts
  constexpr auto f8 = []{ return 8; };
  static_assert(values::fixed_value_of_v<decltype(f8)> == 8);
  constexpr auto f8d = []{ return 8.; };
  static_assert(values::fixed_value_of_v<decltype(f8d)> == 8.);
#endif
}

#include "values/concepts/fixed_value_compares_with.hpp"

TEST(values, fixed_value_compares_with)
{
  static_assert(values::fixed_value_compares_with<std::integral_constant<int, 7>, 7>);
  static_assert(not values::fixed_value_compares_with<std::integral_constant<int, 7>, 6>);
  static_assert(values::fixed_value_compares_with<std::integral_constant<int, 6>, 7, std::less<>>);
  static_assert(not values::fixed_value_compares_with<std::integral_constant<int, 6>, 7, std::greater<>>);
}

#include "values/concepts/value.hpp"

TEST(values, value)
{
  static_assert(values::value<double>);
  static_assert(values::value<std::integral_constant<int, 6>>);
}

#include "values/functions/operation.hpp"
#include "values/classes/fixed-constants.hpp"

TEST(values, operation)
{
  struct NullaryFunc { constexpr auto operator()() const { return 5.5; } };

  static_assert(values::fixed<values::consteval_operation<NullaryFunc>>);
  struct add_op { constexpr auto operator()(int a) const { return a + 1; } };
  struct take_real { constexpr auto operator()(int a) const { return std::real(a); } };

  static_assert(values::fixed<values::consteval_operation<std::negate<>, std::integral_constant<int, 3>>>);
  static_assert(values::fixed<values::consteval_operation<std::multiplies<>, std::integral_constant<int, 2>, std::integral_constant<int, 3>>>);
  static_assert(values::operation(std::plus{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 5>{}) == 9);

  static_assert(values::fixed_value_of_v<decltype(values::operation(std::plus{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 5>{}))> == 9);
  static_assert(values::fixed_value_of_v<decltype(values::operation(add_op{}, std::integral_constant<int, 4>{}))> == 5);
  static_assert(values::fixed_value_of_v<decltype(values::operation(take_real{}, std::integral_constant<int, 4>{}))> == 4);
  static_assert(values::fixed_value_of_v<decltype(values::operation(NullaryFunc{}))> == 5.5);

  static_assert(values::to_value_type(values::consteval_operation<NullaryFunc>{}) == 5.5);
  static_assert(values::to_value_type(values::operation(std::plus{}, 4, 5)) == 9);
  EXPECT_EQ(values::to_value_type(values::operation([](){ return 4 + 5; })), 9);
  int k = 9; EXPECT_EQ(values::to_value_type(values::operation([&k]{ return k; })), 9);

  static_assert(values::fixed_value_of_v<decltype(values::operation(std::equal_to{}, values::fixed_partial_ordering_equivalent{}, values::fixed_partial_ordering_equivalent{}))>);
  static_assert(values::fixed_value_of_v<decltype(values::operation(std::equal_to{}, values::fixed_partial_ordering_less{}, values::fixed_partial_ordering_less{}))>);
  static_assert(values::fixed_value_of_v<decltype(values::operation(std::equal_to{}, values::fixed_partial_ordering_greater{}, values::fixed_partial_ordering_greater{}))>);
  static_assert(values::fixed_value_of_v<decltype(values::operation(std::equal_to{}, values::fixed_partial_ordering_unordered{}, values::fixed_partial_ordering_unordered{}))>);
  static_assert(values::fixed_value_of_v<decltype(values::operation(std::not_equal_to{}, values::fixed_partial_ordering_equivalent{}, values::fixed_partial_ordering_less{}))>);
  static_assert(values::fixed_value_of_v<decltype(values::operation(std::not_equal_to{}, values::fixed_partial_ordering_equivalent{}, values::fixed_partial_ordering_greater{}))>);
  static_assert(values::fixed_value_of_v<decltype(values::operation(std::not_equal_to{}, values::fixed_partial_ordering_equivalent{}, values::fixed_partial_ordering_unordered{}))>);

  static_assert(values::fixed_value_of_v<decltype(values::operation(stdcompat::compare_three_way{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{}))> == stdcompat::partial_ordering::equivalent);
  static_assert(values::fixed_value_of_v<decltype(values::operation(stdcompat::compare_three_way{}, std::integral_constant<int, 3>{}, std::integral_constant<int, 4>{}))> == stdcompat::partial_ordering::less);
  static_assert(values::fixed_value_of_v<decltype(values::operation(stdcompat::compare_three_way{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 3>{}))> == stdcompat::partial_ordering::greater);
  static_assert(values::operation(stdcompat::compare_three_way{}, std::integral_constant<int, 4>{}, 3) == stdcompat::partial_ordering::greater);
}

#include "values/traits/value_type_of.hpp"
#include "values/traits/real_type_of.hpp"
#include "values/classes/fixed_value.hpp"

TEST(values, fixed_value)
{
  static_assert(values::fixed<values::fixed_value<double, 3>>);
  static_assert(not values::number<values::fixed_value<double, 3>>);
  static_assert(values::floating<values::fixed_value<double, 3>>);
  static_assert(values::fixed<decltype(values::fixed_value{values::fixed_value<double, 7>{}})>);

  static_assert(values::to_value_type(values::fixed_value<double, 3>{}) == 3);
  static_assert(values::fixed_value<double, 3>{}() == 3);
  static_assert(values::fixed_value {std::integral_constant<int, 7>{}}() == 7);
  static_assert(decltype(values::fixed_value{std::integral_constant<int, 7>{}})::value == 7);
  static_assert(values::fixed_value{values::fixed_value<double, 7>{}} == 7);
  static_assert(stdcompat::same_as<decltype(values::fixed_value{std::integral_constant<int, 7>{}})::value_type, int>);
  static_assert(stdcompat::same_as<values::fixed_value<int, 3>::value_type, int>);
  static_assert(stdcompat::same_as<values::fixed_value<double, 3>::value_type, double>);

  static_assert(stdcompat::same_as<values::value_type_of_t<values::fixed_value<double, 3>>, double>);
  static_assert(stdcompat::same_as<values::value_type_of_t<values::real_type_of_t<values::fixed_value<double, 3>>>, double>);
}

#include "values/functions/cast_to.hpp"

TEST(values, cast_to)
{
  static_assert(values::cast_to<double>(std::integral_constant<int, 4>{}) == 4);
  static_assert(stdcompat::same_as<values::fixed_value_of<decltype(values::cast_to<int>(std::integral_constant<int, 4>{}))>::value_type, int>);
  static_assert(stdcompat::same_as<values::fixed_value_of<decltype(values::cast_to<double>(std::integral_constant<int, 4>{}))>::value_type, double>);
  static_assert(values::cast_to<double>(values::fixed_value<float, 4>{}) == 4);
  static_assert(stdcompat::same_as<decltype(values::cast_to<double>(values::fixed_value<double, 4>{})), values::fixed_value<double, 4>&&>);
  static_assert(stdcompat::same_as<typename values::fixed_value_of<decltype(values::cast_to<double>(values::fixed_value<double, 4>{}))>::value_type, double>);
  static_assert(stdcompat::same_as<values::fixed_value_of<decltype(values::cast_to<int>(values::fixed_value<int, 4>{}))>::value_type, int>);
  static_assert(stdcompat::same_as<values::fixed_value_of<decltype(values::cast_to<double>(values::fixed_value<int, 4>{}))>::value_type, double>);
  static_assert(values::cast_to<double>(values::fixed_value<float, 4>{}) == 4);
  static_assert(stdcompat::same_as<values::value_type_of_t<decltype(values::cast_to<double>(values::fixed_value<float, 4>{}))>, double>);

  static_assert(values::to_value_type(values::cast_to<stdcompat::partial_ordering>(values::operation(stdcompat::compare_three_way{}, std::integral_constant<int, 3>{}, std::integral_constant<int, 4>{}))) == stdcompat::partial_ordering::less);
}

#include "values/functions/internal/near.hpp"

TEST(values, near)
{
  static_assert(values::internal::near(10., 10.));
  static_assert(values::internal::near<2>(0., 0. + std::numeric_limits<double>::epsilon()));
  static_assert(not values::internal::near<2>(0., 0. + 3 * std::numeric_limits<double>::epsilon()));

  static_assert(values::internal::near(values::fixed_value<double, 4>{}, values::fixed_value<double, 5>{}, 2));
  static_assert(values::internal::near(std::integral_constant<int, 4>{}, std::integral_constant<int, 5>{}, 2));
  static_assert(values::internal::near(std::integral_constant<int, 4>{}, 5, 2));
  static_assert(not values::internal::near(std::integral_constant<int, 4>{}, 6, 1));
}

