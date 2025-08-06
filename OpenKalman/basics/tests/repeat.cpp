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
#include "basics/compatibility/views/repeat.hpp"

using namespace OpenKalman;

TEST(basics, repeat_view)
{
  static_assert(stdcompat::ranges::view<stdcompat::ranges::repeat_view<double, int>>);
  auto i3 = stdcompat::ranges::views::repeat(7., 4);
  EXPECT_EQ(i3.size(), 4);
  static_assert(stdcompat::ranges::views::repeat(7., 4).size() == 4);

  EXPECT_EQ(stdcompat::ranges::views::repeat(7., 4).begin()[2], 7.);
  EXPECT_EQ(stdcompat::ranges::views::repeat(7., 4u).begin()[3], 7.);
  EXPECT_EQ(stdcompat::ranges::views::repeat(7.).begin()[100], 7.);

  auto i8 = stdcompat::ranges::views::repeat(7., 8u);
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
}
