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
 * \brief Tests for \ref ranges::views::all
 */

#include "tests.hpp"
#include "basics/compatibility/ranges/range-concepts.hpp"
#include "basics/compatibility/views/all.hpp"

using namespace OpenKalman;

TEST(basics, all_view)
{
  static_assert(stdex::ranges::range<stdex::ranges::views::all_t<std::array<int, 7>>>);
  static_assert(stdex::ranges::range<stdex::ranges::views::all_t<std::vector<int>>>);
  static_assert(stdex::ranges::range<stdex::ranges::views::all_t<std::vector<int>&>>);
  static_assert(stdex::ranges::random_access_range<stdex::ranges::views::all_t<std::vector<int>>>);
  static_assert(stdex::ranges::random_access_range<stdex::ranges::views::all_t<std::vector<int>&>>);

  static_assert(stdex::ranges::viewable_range<std::array<int, 7>>);
  static_assert(stdex::ranges::viewable_range<std::vector<int>&>);

  static_assert(stdex::ranges::view<stdex::ranges::views::all_t<std::array<int, 7>>>);
  static_assert(stdex::ranges::view<stdex::ranges::views::all_t<std::vector<int>>>);
  static_assert(stdex::ranges::view<stdex::ranges::views::all_t<std::vector<int>&>>);
  static_assert(stdex::ranges::view<stdex::ranges::views::all_t<const std::vector<int>&>>);

  auto v1 = std::vector{4, 5, 6, 7, 8};
  EXPECT_EQ((stdex::ranges::views::all(v1)[0u]), 4);
  EXPECT_EQ((stdex::ranges::views::all(v1)[4u]), 8);
  EXPECT_EQ((stdex::ranges::views::all(std::vector{4, 5, 6, 7, 8})[3u]), 7);
  EXPECT_EQ((stdex::ranges::views::all(v1)[std::integral_constant<std::size_t, 4>{}]), 8);
  EXPECT_EQ(*stdex::ranges::begin(stdex::ranges::views::all(v1)), 4);
  EXPECT_EQ(*--stdex::ranges::end(stdex::ranges::views::all(v1)), 8);
  EXPECT_EQ(stdex::ranges::size(stdex::ranges::views::all(v1)), 5);

  EXPECT_EQ(stdex::ranges::views::all(v1).front(), 4);
  EXPECT_EQ(stdex::ranges::views::all(std::vector{4, 5, 6, 7, 8}).front(), 4);
  EXPECT_EQ(stdex::ranges::views::all(v1).back(), 8);
  EXPECT_EQ(stdex::ranges::views::all(std::vector{4, 5, 6, 7, 8}).back(), 8);
  EXPECT_EQ(*stdex::ranges::cbegin(stdex::ranges::views::all(v1)), 4);
  EXPECT_EQ(*--stdex::ranges::cend(stdex::ranges::views::all(v1)), 8);
  EXPECT_TRUE(stdex::ranges::views::all(std::vector<int>{}).empty());
  EXPECT_FALSE(stdex::ranges::views::all(std::vector{4, 5, 6, 7, 8}).empty());
  EXPECT_FALSE(stdex::ranges::views::all(std::vector<int>{}));
  EXPECT_TRUE(stdex::ranges::views::all(std::vector{4, 5, 6, 7, 8}));

  auto v2 = std::vector{9, 10, 11, 12, 13};
  auto id1_v2 = stdex::ranges::views::all(v2);
  id1_v2 = stdex::ranges::views::all(v2);
  EXPECT_EQ((id1_v2[3u]), 12);
  id1_v2 = stdex::ranges::views::all(v1);
  EXPECT_EQ((id1_v2[3u]), 7);

  auto id2_v1 = stdex::ranges::views::all(std::vector{4, 5, 6, 7, 8});
  EXPECT_EQ((id2_v1[2u]), 6);

  auto v1b = stdex::ranges::begin(v1);
  EXPECT_EQ(*v1b, 4);
  EXPECT_EQ(*++v1b, 5);
  EXPECT_EQ(v1b[3], 8);
  EXPECT_EQ(*--v1b, 4);
  EXPECT_EQ(*--stdex::ranges::end(v1), 8);

  constexpr int a1[5] = {1, 2, 3, 4, 5};
  static_assert(stdex::ranges::random_access_range<stdex::ranges::views::all_t<int(&)[5]>>);
  static_assert((stdex::ranges::views::all(a1)[0u]) == 1);
  static_assert((stdex::ranges::views::all(a1)[4u]) == 5);
  static_assert(*stdex::ranges::begin(stdex::ranges::views::all(a1)) == 1);
  static_assert(*(stdex::ranges::end(stdex::ranges::views::all(a1)) - 1) == 5);
  static_assert(*stdex::ranges::cbegin(stdex::ranges::views::all(a1)) == 1);
  static_assert(*(stdex::ranges::cend(stdex::ranges::views::all(a1)) - 1) == 5);
  static_assert(stdex::ranges::size(stdex::ranges::views::all(a1)) == 5);
  static_assert(stdex::ranges::begin(stdex::ranges::views::all(a1))[2] == 3);
  static_assert(stdex::ranges::views::all(a1).front() == 1);
  static_assert(stdex::ranges::views::all(a1).back() == 5);
  static_assert(not stdex::ranges::views::all(a1).empty());

  EXPECT_EQ((v1 | stdex::ranges::views::all)[2u], 6);
  EXPECT_EQ((v1 | stdex::ranges::views::all | stdex::ranges::views::all)[3u], 7);
  EXPECT_EQ((std::vector{4, 5, 6, 7, 8} | stdex::ranges::views::all)[2u], 6);
  EXPECT_EQ((a1 | stdex::ranges::views::all)[1u], 2);
}

