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
 * \brief Tests for \ref ranges::single_view and \ref ranges::views::single
 */

#include "tests.hpp"
#include "basics/compatibility/views/all.hpp"
#include "basics/compatibility/views/single.hpp"

using namespace OpenKalman;

TEST(basics, single_view)
{
  static_assert(stdex::ranges::view<stdex::ranges::single_view<std::tuple<int, double>>>);
  static_assert(stdex::ranges::view<stdex::ranges::single_view<std::vector<int>>>);
  static_assert(stdex::ranges::viewable_range<stdex::ranges::single_view<std::tuple<int, double>>>);
  static_assert(stdex::ranges::viewable_range<stdex::ranges::single_view<std::vector<int>>>);

  static_assert((stdex::ranges::views::single(4)[0u]) == 4);

  static constexpr auto s1 = stdex::ranges::views::single(7);
  constexpr auto is1 = stdex::ranges::begin(s1);
  static_assert(*is1 == 7);
  auto e_s1 = stdex::ranges::end(s1);
  EXPECT_EQ(*--e_s1, 7);

  auto s2 = stdex::ranges::views::single(7);
  EXPECT_EQ(*stdex::ranges::begin(s2), 7);
  *s2.data() = 8;
  EXPECT_EQ(*stdex::ranges::begin(s2), 8);

  EXPECT_EQ((s1 | stdex::ranges::views::all)[0u], 7);
  EXPECT_EQ((stdex::ranges::views::single(7) | stdex::ranges::views::all)[0u], 7);
}


