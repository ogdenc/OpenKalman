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
 * \brief Tests for utils.hpp
 */

#include <tuple>
#include <array>
#include <gtest/gtest.h>
#include "basics/language-features.hpp"
#include "basics/utils.hpp"

using namespace OpenKalman;

TEST(basics, tuple_slice)
{
  std::tuple t {1, "c", 5.0, 6.0};
  static_assert(std::is_same_v<decltype(internal::tuple_slice<0, 0>(t)), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<0, 1>(t)), std::tuple<int>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 2>(t)), std::tuple<const char*>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<2, 2>(t)), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<2, 3>(t)), std::tuple<double>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<3, 4>(t)), std::tuple<double>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(t)), std::tuple<const char*, double>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<2, 4>(t)), std::tuple<double, double>>);

  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(std::tuple {1, "c", 5.0, 6.0})), std::tuple<const char*, double>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<2, 4>(std::tuple {1, "c", 5.0, 6.0})), std::tuple<double, double>>);

  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(std::forward_as_tuple(1, "c", 5.0, 6.0))), std::tuple<const char*, double>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<2, 4>(std::forward_as_tuple(1, "c", 5.0, 6.0))), std::tuple<double, double>>);

  std::array a {1, 2, 3, 4};
  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(a)), std::tuple<int, int>>);
  static_assert(std::is_same_v<decltype(internal::tuple_slice<1, 3>(std::array {1, 2, 3, 4})), std::tuple<int, int>>);
}


TEST(basics, forward_as_tuple_slice)
{
  std::tuple t {1, "c", 5.0, 6.0};
  static_assert(std::is_same_v<decltype(internal::forward_as_tuple_slice<0, 0>(t)), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::forward_as_tuple_slice<0, 1>(t)), std::tuple<int&>>);
  static_assert(std::is_same_v<decltype(internal::forward_as_tuple_slice<1, 2>(t)), std::tuple<const char*&>>);
  static_assert(std::is_same_v<decltype(internal::forward_as_tuple_slice<2, 2>(t)), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::forward_as_tuple_slice<2, 3>(t)), std::tuple<double&>>);
  static_assert(std::is_same_v<decltype(internal::forward_as_tuple_slice<3, 4>(t)), std::tuple<double&>>);
  static_assert(std::is_same_v<decltype(internal::forward_as_tuple_slice<1, 3>(t)), std::tuple<const char*&, double&>>);
  static_assert(std::is_same_v<decltype(internal::forward_as_tuple_slice<2, 4>(t)), std::tuple<double&, double&>>);

  static_assert(std::is_same_v<decltype(internal::forward_as_tuple_slice<1, 3>(std::tuple {1, "c", 5.0, 6.0})), std::tuple<const char*&&, double&&>>);
  static_assert(std::is_same_v<decltype(internal::forward_as_tuple_slice<2, 4>(std::tuple {1, "c", 5.0, 6.0})), std::tuple<double&&, double&&>>);

  std::array a {1, 2, 3, 4};
  static_assert(std::is_same_v<decltype(internal::forward_as_tuple_slice<1, 3>(a)), std::tuple<int&, int&>>);
  static_assert(std::is_same_v<decltype(internal::forward_as_tuple_slice<1, 3>(std::array {1, 2, 3, 4})), std::tuple<int&&, int&&>>);
}


TEST(basics, fill_tuple)
{
  double d = 7.0;
  static_assert(std::is_same_v<decltype(internal::fill_tuple<0>(d)), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::fill_tuple<1>(d)), std::tuple<double&>>);
  static_assert(std::is_same_v<decltype(internal::fill_tuple<4>(d)), std::tuple<double&, double&, double&, double&>>);

  static_assert(std::is_same_v<decltype(internal::fill_tuple<0>(5.0)), std::tuple<>>);
  static_assert(std::is_same_v<decltype(internal::fill_tuple<1>(5.0)), std::tuple<double>>);
  static_assert(std::is_same_v<decltype(internal::fill_tuple<4>(5.0)), std::tuple<double, double, double, double>>);
}


TEST(basics, tuple_reverse)
{
  constexpr std::tuple ta {1, 'c', 5.0, 6.0};
  constexpr auto rta = internal::tuple_reverse(ta);
  static_assert(std::get<0>(rta) == 6.0);
  static_assert(std::get<1>(rta) == 5.0);
  static_assert(std::get<2>(rta) == 'c');
  static_assert(std::get<3>(rta) == 1);
  static_assert(std::is_same_v<decltype(internal::tuple_reverse(ta)), std::tuple<double, double, char, int>>);
  static_assert(std::is_same_v<decltype(internal::tuple_reverse(std::tuple {1, 'c', 5.0, 6.0})), std::tuple<double, double, char, int>>);
  int n0 = 1;
  char n1 = 'c';
  double n2 = 5.0, n3 = 6.0;
  auto tb = std::forward_as_tuple(n0, n1, n2, n3);
  auto rtb = internal::tuple_reverse(tb);
  EXPECT_TRUE(std::get<0>(rtb) == n3);
  EXPECT_TRUE(std::get<1>(rtb) == n2);
  EXPECT_TRUE(std::get<2>(rtb) == n1);
  EXPECT_TRUE(std::get<3>(rtb) == n0);
  static_assert(std::is_same_v<decltype(internal::tuple_reverse(tb)), std::tuple<double, double, char, int>>);
}


TEST(basics, tuple_flatten)
{
  constexpr auto fta = internal::tuple_flatten(std::tuple {0, std::tuple{1, std::tuple{2, 3}}, 4, 5});
  static_assert(std::get<0>(fta) == 0);
  static_assert(std::get<1>(fta) == 1);
  static_assert(std::get<2>(fta) == 2);
  static_assert(std::get<3>(fta) == 3);
  static_assert(std::get<4>(fta) == 4);
  static_assert(std::get<5>(fta) == 5);

  static_assert(std::get<0>(internal::tuple_flatten(std::tuple{7})) == 7);

  constexpr auto ftb = internal::tuple_flatten(std::array{3, 4, 5});
  static_assert(std::get<2>(ftb) == 5);
  static_assert(std::is_same_v<const decltype(ftb), const std::array<int, 3>>);

  constexpr auto ftc = internal::tuple_flatten(std::array{std::tuple{0, 1}, std::tuple{2, 3}, std::tuple{4, 5}});
  static_assert(std::get<0>(ftc) == 0);
  static_assert(std::get<1>(ftc) == 1);
  static_assert(std::get<2>(ftc) == 2);
  static_assert(std::get<3>(ftc) == 3);
  static_assert(std::get<4>(ftc) == 4);
  static_assert(std::get<5>(ftc) == 5);
}


#include "basics/internal/iota_tuple.hpp"

TEST(basics, iota_tuple)
{
  using I3 = decltype(internal::iota_tuple<0, 3>());
  static_assert(std::tuple_size_v<I3> == 3);
  static_assert(std::tuple_element_t<0, I3>::value == 0);
  static_assert(std::tuple_element_t<1, I3>::value == 1);
  using internal::get;
  static_assert(get<3>(I3{})() == 3);

  using I27 = decltype(internal::iota_tuple<2, 7>());
  static_assert(std::tuple_size_v<I27> == 5);
  static_assert(std::tuple_element_t<0, I27>::value == 2);
  static_assert(std::tuple_element_t<1, I27>::value == 3);
  using internal::get;
  static_assert(get<3>(I27{})() == 5);
}


#include "basics/internal/iota_range.hpp"

TEST(basics, iota_range)
{
  auto i9 = internal::iota_range(0, 9);
  using std::size;
  EXPECT_EQ(size(i9), 9);
  auto it9 = i9.begin();
  EXPECT_EQ(*it9, 0);
  it9++;
  EXPECT_EQ(*it9, 1);
  EXPECT_EQ(*(it9 + 1), 2);
  EXPECT_EQ(it9[1], 2);
  EXPECT_EQ(*(2 + it9), 3);
  EXPECT_EQ(it9[3], 4);
  ++it9;
  EXPECT_EQ(*it9, 2);
  EXPECT_EQ(*(it9 - 1), 1);
  EXPECT_EQ(it9[-1], 1);
  EXPECT_EQ(it9[1], 3);
  EXPECT_EQ(it9++[2], 4);
  EXPECT_EQ(*it9, 3);
  EXPECT_EQ(*(it9 - 2), 1);
  EXPECT_EQ(it9[1], 4);
  EXPECT_EQ(it9--[2], 5);
  EXPECT_EQ(*it9, 2);
  --it9;
  EXPECT_EQ(*it9, 1);
}

