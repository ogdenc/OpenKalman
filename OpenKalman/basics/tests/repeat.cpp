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

using namespace OpenKalman;

#ifdef __cpp_lib_ranges
namespace rg = std::ranges;
#else
namespace rg = OpenKalman::ranges;
#endif
namespace vw = rg::views;

TEST(basics, repeat_view)
{
  static_assert(rg::view<rg::repeat_view<double, int>>);
  auto i3 = rg::views::repeat(7., 4);
  EXPECT_EQ(i3.size(), 4);
  static_assert(rg::views::repeat(7., 4).size() == 4);

  EXPECT_EQ(rg::views::repeat(7., 4).begin()[2], 7.);
  EXPECT_EQ(rg::views::repeat(7., 4u).begin()[3], 7.);
  EXPECT_EQ(rg::views::repeat(7.).begin()[100], 7.);

  auto i8 = rg::views::repeat(7., 8u);
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
