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
 * \brief Tests for \ref ranges::repeat_view and \ref ranges::views::repeat.
 */

#include "tests.hpp"
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/views/repeat.hpp"
#endif
#include "collections/views/repeat.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

#ifdef __cpp_lib_ranges
namespace rg = std::ranges;
#else
namespace rg = OpenKalman::ranges;
#endif

#ifdef __cpp_lib_ranges_repeat
namespace cv = std::ranges::views;
#else
namespace cv = ranges::views;
#endif

TEST(collections, repeat_tuple_view)
{
  static_assert(std::tuple_size_v<repeat_tuple_view<4, double>> == 4);
  static_assert(std::is_same_v<std::tuple_element_t<0, repeat_tuple_view<4, std::integral_constant<std::size_t, 2>>>, std::integral_constant<std::size_t, 2>>);
  static_assert(get(repeat_tuple_view<4, double>{7.0}, std::integral_constant<std::size_t, 0>{}) == 7.0);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(repeat_tuple_view<4, std::integral_constant<std::size_t, 2>>())>, std::integral_constant<std::size_t, 2>>);

  constexpr double d = 7.0;
  static_assert(std::tuple_size_v<decltype(repeat_tuple_view<0, double>(d))> == 0);
  static_assert(std::tuple_size_v<decltype(repeat_tuple_view<4, double>(d))> == 4);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(repeat_tuple_view<4, double>(d))>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<3, decltype(repeat_tuple_view<4, double>(d))>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<3, decltype(repeat_tuple_view<4, double>(std::declval<double&>()))>, double>);
  static_assert(get(repeat_tuple_view<4, double>(d), std::integral_constant<std::size_t, 0>{}) == 7.0);
  static_assert(get(repeat_tuple_view<4, double>(d), std::integral_constant<std::size_t, 3>{}) == 7.0);
  static_assert(std::is_same_v<decltype(get<0>(repeat_tuple_view<4, double>(d))), double>);
  static_assert(std::is_same_v<decltype(get<0>(repeat_tuple_view<4, double>(std::declval<double&>()))), double>);

  static_assert(std::tuple_size_v<decltype(repeat_tuple_view<0, double>(5.0))> == 0);
  static_assert(std::tuple_size_v<decltype(repeat_tuple_view<4, double>(5.0))> == 4);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(repeat_tuple_view<4, double>(5.0))>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<3, decltype(repeat_tuple_view<4, double>(5.0))>, double>);
  static_assert(get(repeat_tuple_view<4, double>(5.0), std::integral_constant<std::size_t, 0>{}) == 5.0);
  static_assert(get(repeat_tuple_view<4, double>(6.0), std::integral_constant<std::size_t, 3>{}) == 6.0);
}


TEST(collections, repeat_view)
{
  constexpr auto c0 = std::integral_constant<std::size_t, 0>{};
  auto i3 = views::repeat(7., 4u);
  static_assert(rg::view<decltype(i3)>);
  EXPECT_EQ(i3.size(), 4);
  static_assert(views::repeat(7., 4u).size() == 4);

  EXPECT_EQ(views::repeat(7., 4u).begin()[2], 7.);
  EXPECT_EQ(views::repeat(7., 4u).begin()[3], 7.);
  EXPECT_EQ(views::repeat(7.).begin()[100], 7.);

  static constexpr auto i1 = views::repeat(7., std::integral_constant<std::size_t, 1>{});
  auto it1 = i1.begin();
  EXPECT_EQ(*it1, 7.);
  static_assert(size_of_v<decltype(i1)> == 1);
  static_assert(i1[c0] == 7.);
  EXPECT_EQ(i1[0U], 7.);
  static_assert(get(i1, c0) == 7.);
  static_assert(get(i1, 0U) == 7.);

  constexpr auto i8 = views::repeat(7., std::integral_constant<std::size_t, 8>{});
  auto it8 = i8.begin();
  EXPECT_EQ(*it8, 7.);
  it8++;
  EXPECT_EQ(*it8, 7.);
  EXPECT_EQ(*(it8 + 1), 7.);
  EXPECT_EQ(it8[1], 7.);
  EXPECT_EQ(*(2 + it8), 7.);
  EXPECT_EQ(it8[3], 7.);
  ++it8;
  EXPECT_EQ(*it8, 7.);
  EXPECT_EQ(*(it8 - 1), 7.);
  EXPECT_EQ(it8[-1], 7.);
  EXPECT_EQ(it8[1], 7.);
  EXPECT_EQ(it8++[2], 7.);
  EXPECT_EQ(*it8, 7.);
  EXPECT_EQ(*(it8 - 2), 7.);
  EXPECT_EQ(it8[1], 7.);
  EXPECT_EQ(it8--[2], 7.);
  EXPECT_EQ(*it8, 7.);
  --it8;
  EXPECT_EQ(*it8, 7.);

  static_assert(size_of_v<decltype(i8)> == 8);
  static_assert(i8[c0] == 7.);
  static_assert(i8[std::integral_constant<std::size_t, 5>{}] == 7.);
  EXPECT_EQ(i8[0U], 7.);
  EXPECT_EQ(i8[5U], 7.);
  static_assert(get(i8, c0) == 7.);
  static_assert(get(i8, std::integral_constant<std::size_t, 3>{}) == 7.);
  EXPECT_EQ(get(i8, 2U), 7.);
  EXPECT_EQ(get(i8, 6U), 7.);
}
