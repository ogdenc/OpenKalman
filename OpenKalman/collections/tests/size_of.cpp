/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for \ref collections::size_of
 */

#include "tests.hpp"
#include "basics/basics.hpp"
#include "values/values.hpp"
#include "collections/concepts/sized.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

TEST(collections, sized)
{
  static_assert(sized<std::tuple<int, double>>);
  static_assert(sized<const std::tuple<int, double>>);
  static_assert(sized<std::tuple<>>);
  static_assert(sized<std::array<int, 7>>);
  static_assert(sized<const std::array<int, 7>&>);
  static_assert(sized<std::vector<int>>);
  static_assert(sized<const std::vector<int>&>);
  static_assert(sized<int(&)[5]>);
  static_assert(sized<const int(&)[6]>);
  static_assert(sized<stdcompat::ranges::views::all_t<int(&)[7]>>);
  static_assert(sized<stdcompat::ranges::views::all_t<decltype(stdcompat::ranges::views::reverse(stdcompat::ranges::views::all(std::declval<int(&)[5]>())))>>);
}


#include "collections/functions/get_size.hpp"

TEST(collections, get_size)
{
  constexpr auto tup1 = std::tuple{1,2,3,4,5};
  static_assert(get_size(std::array{1,2,3,4,5}) == 5);
  static_assert(get_size(tup1) == 5);
  static_assert(values::fixed<decltype(get_size(std::tuple{1,2,3,4,5}))>);
  static_assert(values::fixed<decltype(get_size(tup1))>);

  static constexpr auto arr1 = std::array{1,2,3,4};
  static_assert(get_size(std::array{1,2,3,4}) == 4);
  static_assert(get_size(arr1) == 4);
  static_assert(values::fixed<decltype(get_size(std::array{1,2,3,4}))>);
  static_assert(values::fixed<decltype(get_size(arr1))>);

  constexpr int a1[5] = {1, 2, 3, 4, 5};
  static_assert(get_size(a1) == 5);
  static_assert(values::fixed<decltype(get_size(a1))>);
  static_assert(values::fixed<decltype(get_size(std::declval<int(&)[5]>()))>);
  static_assert(values::fixed<decltype(get_size(std::declval<const int(&)[5]>()))>);

  auto v1 = std::vector{1,2,3};
  EXPECT_EQ(get_size(v1), 3);
#if __cplusplus >= 202002L
  static_assert(get_size(std::vector{1,2,3}) == 3);
#else
  EXPECT_EQ(get_size(std::vector{1,2,3}), 3);
#endif
  static_assert(not values::fixed<decltype(get_size(v1))>);
  static_assert(not values::fixed<decltype(get_size(std::vector{1,2,3}))>);

  constexpr auto all_arr1 = stdcompat::ranges::views::all(arr1);
  static_assert(stdcompat::ranges::size(stdcompat::ranges::views::all(arr1)) == 4);
  static_assert(stdcompat::ranges::size(all_arr1) == 4);
  static_assert(std::integral_constant<std::size_t, stdcompat::ranges::size(all_arr1)>::value == 4);

  static_assert(get_size(stdcompat::ranges::views::all(std::array{1,2,3,4})) == 4);
  static_assert(get_size(stdcompat::ranges::views::all(arr1)) == 4);

  static_assert(values::fixed<decltype(get_size(stdcompat::ranges::views::all(std::array{1,2,3,4})))>);
  static_assert(values::fixed<decltype(get_size(stdcompat::ranges::views::all(arr1)))>);
  static_assert(get_size(stdcompat::ranges::views::reverse(stdcompat::ranges::views::all(std::array{1,2,3,4}))) == 4);
  static_assert(get_size(stdcompat::ranges::views::reverse(stdcompat::ranges::views::all(arr1))) == 4);
  static_assert(values::fixed<decltype(get_size(stdcompat::ranges::views::reverse(stdcompat::ranges::views::all(std::array{1,2,3,4}))))>);
  static_assert(values::fixed<decltype(get_size(stdcompat::ranges::views::reverse(stdcompat::ranges::views::all(arr1))))>);

  static_assert(get_size(stdcompat::ranges::views::all(a1)) == 5);
  static_assert(get_size(stdcompat::ranges::views::reverse(stdcompat::ranges::views::all(a1))) == 5);
  static_assert(values::fixed<decltype(get_size(stdcompat::ranges::views::all(a1)))>);
  static_assert(values::fixed<decltype(get_size(stdcompat::ranges::views::reverse(stdcompat::ranges::views::all(a1))))>);

  EXPECT_EQ(get_size(stdcompat::ranges::views::concat(arr1, a1)), 9);
  EXPECT_EQ(get_size(stdcompat::ranges::views::concat(arr1, a1) | stdcompat::ranges::views::transform(std::negate{}) | stdcompat::ranges::views::reverse), 9);
  EXPECT_EQ(get_size(stdcompat::ranges::views::concat(arr1, v1, a1) | stdcompat::ranges::views::transform(std::negate{}) | stdcompat::ranges::views::reverse), 12);
}


#include "collections/traits/size_of.hpp"

TEST(collections, size_of)
{
  static_assert(size_of_v<std::tuple<int, double, long double>> == 3);
  static_assert(size_of_v<std::array<double, 5>> == 5);
  static_assert(size_of_v<std::vector<double>> == dynamic_size);
  static_assert(size_of_v<std::initializer_list<double>> == dynamic_size);
  static_assert(size_of_v<int(&)[5]> == 5);

  constexpr int a1[5] = {1, 2, 3, 4, 5};
  static_assert(size_of_v<std::tuple<>> == 0);
  static_assert(size_of_v<std::tuple<int, double>> == 2);
  static_assert(size_of_v<std::array<int, 7>> == 7);
  static_assert(size_of_v<decltype(a1)> == 5);
  static_assert(size_of_v<int(&)[5]> == 5);
  static_assert(size_of_v<const int(&)[6]> == 6);

  static_assert(size_of_v<stdcompat::ranges::views::all_t<std::array<double, 5>>> == 5);
  static_assert(size_of_v<stdcompat::ranges::views::all_t<int(&)[7]>> == 7);

  static_assert(size_of_v<stdcompat::ranges::empty_view<int>> == 0);
  static_assert(size_of_v<stdcompat::ranges::single_view<int>> == 1);

  static_assert(size_of_v<stdcompat::ranges::concat_view<stdcompat::ranges::views::all_t<std::array<double, 5>>, stdcompat::ranges::views::all_t<int(&)[4]>>> == 9);
}

