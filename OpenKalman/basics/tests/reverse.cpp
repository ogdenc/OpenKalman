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
 * \brief Tests for \ref ranges::reverse_view and \ref ranges::stdcompat::ranges::views::reverse.
 */

#include <vector>
#include "tests.hpp"
#include "basics/compatibility/views/all.hpp"
#include "basics/compatibility/views/reverse.hpp"

using namespace OpenKalman;

TEST(basics, reverse_view)
{
  constexpr int a1[5] = {1, 2, 3, 4, 5};
  static_assert(stdcompat::ranges::range<stdcompat::ranges::reverse_view<stdcompat::ranges::views::all_t<int(&)[5]>>>);
  static_assert(stdcompat::ranges::range<stdcompat::ranges::reverse_view<stdcompat::ranges::views::all_t<const int(&)[5]>>>);

  static_assert((stdcompat::ranges::views::reverse(a1)[0_uz]) == 5);
  static_assert((stdcompat::ranges::views::reverse(a1)[4u]) == 1);
  static_assert(*stdcompat::ranges::begin(stdcompat::ranges::views::reverse(a1)) == 5);
  static_assert(*(stdcompat::ranges::end(stdcompat::ranges::views::reverse(a1)) - 1) == 1);
  static_assert(stdcompat::ranges::views::reverse(a1).size() == 5);
  static_assert(*stdcompat::ranges::cbegin(stdcompat::ranges::views::reverse(a1)) == 5);
  static_assert(*(stdcompat::ranges::cend(stdcompat::ranges::views::reverse(a1)) - 1) == 1);
  static_assert(stdcompat::ranges::size(stdcompat::ranges::views::reverse(a1)) == 5);
  static_assert(stdcompat::ranges::begin(stdcompat::ranges::views::reverse(a1))[3] == 2);
  static_assert(stdcompat::ranges::views::reverse(a1).front() == 5);
  static_assert(stdcompat::ranges::views::reverse(a1).back() == 1);
  static_assert(not stdcompat::ranges::views::reverse(a1).empty());

  auto a2 = std::array{3, 4, 5};
  EXPECT_EQ(*stdcompat::ranges::begin(stdcompat::ranges::views::reverse(a2)), 5);
  EXPECT_EQ(*--stdcompat::ranges::end(stdcompat::ranges::views::reverse(a2)), 3);
  EXPECT_EQ(stdcompat::ranges::reverse_view(a2).front(), 5);
  EXPECT_EQ(stdcompat::ranges::reverse_view {a2}.back(), 3);
  EXPECT_EQ((stdcompat::ranges::reverse_view {a2}[0u]), 5);
  EXPECT_EQ((stdcompat::ranges::reverse_view {a2}[1u]), 4);
  EXPECT_EQ((stdcompat::ranges::reverse_view {a2}[2u]), 3);

  auto v1 = std::vector{3, 4, 5};
  EXPECT_EQ((stdcompat::ranges::views::reverse(v1)[0u]), 5);
  EXPECT_EQ((stdcompat::ranges::views::reverse(v1)[1u]), 4);
  EXPECT_EQ((stdcompat::ranges::views::reverse(v1)[2u]), 3);

  EXPECT_EQ((v1 | stdcompat::ranges::views::reverse)[0], 5);
  EXPECT_EQ((a1 | stdcompat::ranges::views::reverse)[1], 4);

}

