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
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/views/all.hpp"
#endif

using namespace OpenKalman;

#ifdef __cpp_lib_ranges
  namespace rg = std::ranges;
#else
  namespace rg = OpenKalman::ranges;
#endif
  namespace vw = rg::views;

TEST(basics, all_view)
{
  static_assert(rg::range<vw::all_t<std::array<int, 7>>>);
  static_assert(rg::range<vw::all_t<std::vector<int>>>);
  static_assert(rg::range<vw::all_t<std::vector<int>&>>);
  static_assert(rg::random_access_range<vw::all_t<std::vector<int>>>);
  static_assert(rg::random_access_range<vw::all_t<std::vector<int>&>>);

  static_assert(rg::viewable_range<std::array<int, 7>>);
  static_assert(rg::viewable_range<std::vector<int>&>);

  static_assert(rg::view<vw::all_t<std::array<int, 7>>>);
  static_assert(rg::view<vw::all_t<std::vector<int>>>);
  static_assert(rg::view<vw::all_t<std::vector<int>&>>);
  static_assert(rg::view<vw::all_t<const std::vector<int>&>>);

  auto v1 = std::vector{4, 5, 6, 7, 8};
  EXPECT_EQ((vw::all(v1)[0u]), 4);
  EXPECT_EQ((vw::all(v1)[4u]), 8);
  EXPECT_EQ((vw::all(std::vector{4, 5, 6, 7, 8})[3u]), 7);
  EXPECT_EQ((vw::all(v1)[std::integral_constant<std::size_t, 4>{}]), 8);
  EXPECT_EQ(*rg::begin(vw::all(v1)), 4);
  EXPECT_EQ(*--rg::end(vw::all(v1)), 8);
  EXPECT_EQ(rg::size(vw::all(v1)), 5);

  EXPECT_EQ(vw::all(v1).front(), 4);
  EXPECT_EQ(vw::all(std::vector{4, 5, 6, 7, 8}).front(), 4);
  EXPECT_EQ(vw::all(v1).back(), 8);
  EXPECT_EQ(vw::all(std::vector{4, 5, 6, 7, 8}).back(), 8);
  EXPECT_EQ(*rg::cbegin(vw::all(v1)), 4);
  EXPECT_EQ(*--rg::cend(vw::all(v1)), 8);
  EXPECT_TRUE(vw::all(std::vector<int>{}).empty());
  EXPECT_FALSE(vw::all(std::vector{4, 5, 6, 7, 8}).empty());
  EXPECT_FALSE(vw::all(std::vector<int>{}));
  EXPECT_TRUE(vw::all(std::vector{4, 5, 6, 7, 8}));

  auto v2 = std::vector{9, 10, 11, 12, 13};
  auto id1_v2 = vw::all(v2);
  id1_v2 = vw::all(v2);
  EXPECT_EQ((id1_v2[3u]), 12);
  id1_v2 = vw::all(v1);
  EXPECT_EQ((id1_v2[3u]), 7);

  auto id2_v1 = vw::all(std::vector{4, 5, 6, 7, 8});
  EXPECT_EQ((id2_v1[2u]), 6);

  auto v1b = rg::begin(v1);
  EXPECT_EQ(*v1b, 4);
  EXPECT_EQ(*++v1b, 5);
  EXPECT_EQ(v1b[3], 8);
  EXPECT_EQ(*--v1b, 4);
  EXPECT_EQ(*--rg::end(v1), 8);

  constexpr int a1[5] = {1, 2, 3, 4, 5};
  static_assert(rg::random_access_range<vw::all_t<int(&)[5]>>);
  static_assert((vw::all(a1)[0u]) == 1);
  static_assert((vw::all(a1)[4u]) == 5);
  static_assert(*rg::begin(vw::all(a1)) == 1);
  static_assert(*(rg::end(vw::all(a1)) - 1) == 5);
  static_assert(*rg::cbegin(vw::all(a1)) == 1);
  static_assert(*(rg::cend(vw::all(a1)) - 1) == 5);
  static_assert(rg::size(vw::all(a1)) == 5);
  static_assert(rg::begin(vw::all(a1))[2] == 3);
  static_assert(vw::all(a1).front() == 1);
  static_assert(vw::all(a1).back() == 5);
  static_assert(not vw::all(a1).empty());

  EXPECT_EQ((v1 | vw::all)[2u], 6);
  EXPECT_EQ((v1 | vw::all | vw::all)[3u], 7);
  EXPECT_EQ((std::vector{4, 5, 6, 7, 8} | vw::all)[2u], 6);
  EXPECT_EQ((a1 | vw::all)[1u], 2);
}

