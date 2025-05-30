/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for \ref ranges::iota_view and \ref ranges::views::iota.
 */

#include "tests.hpp"
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/views/iota.hpp"
#endif

using namespace OpenKalman;

#ifdef __cpp_lib_ranges
namespace rg = std::ranges;
#else
namespace rg = OpenKalman::ranges;
#endif
namespace vw = rg::views;

TEST(basics, iota_view)
{
  auto i3 = rg::views::iota(0, 4);
  EXPECT_EQ(i3.size(), 4);
  static_assert(rg::views::iota(0, 4).size() == 4);

  EXPECT_EQ(rg::views::iota(0, 4).begin()[2], 2);
  EXPECT_EQ(rg::views::iota(0u, 4u).begin()[3], 3);
  EXPECT_EQ(rg::views::iota(1u).begin()[100], 101);
  static_assert(rg::views::iota(0, 4).size() == 4);

  auto i8 = rg::views::iota(0u, 9u);
  auto it8 = i8.begin();
  EXPECT_EQ(*it8, 0);
  it8++;
  EXPECT_EQ(*it8, 1);
  EXPECT_EQ(*(it8 + 1), 2);
  EXPECT_EQ(it8[1], 2);
  EXPECT_EQ(*(2 + it8), 3);
  EXPECT_EQ(it8[3], 4);
  ++it8;
  EXPECT_EQ(*it8, 2);
  EXPECT_EQ(*(it8 - 1), 1);
  EXPECT_EQ(it8[-1], 1);
  EXPECT_EQ(it8[1], 3);
  EXPECT_EQ(it8++[2], 4);
  EXPECT_EQ(*it8, 3);
  EXPECT_EQ(*(it8 - 2), 1);
  EXPECT_EQ(it8[1], 4);
  EXPECT_EQ(it8--[2], 5);
  EXPECT_EQ(*it8, 2);
  --it8;
  EXPECT_EQ(*it8, 1);

  auto i9 = rg::views::iota(0, 10);
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
