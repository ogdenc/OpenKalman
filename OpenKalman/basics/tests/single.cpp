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
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/views/all.hpp"
#include "basics/compatibility/views/single.hpp"
#endif

using namespace OpenKalman;

#ifdef __cpp_lib_ranges
  namespace rg = std::ranges;
#else
  namespace rg = OpenKalman::ranges;
#endif
  namespace vw = rg::views;

TEST(basics, single_view)
{
  static_assert(rg::view<rg::single_view<std::tuple<int, double>>>);
  static_assert(rg::view<rg::single_view<std::vector<int>>>);
  static_assert(rg::viewable_range<rg::single_view<std::tuple<int, double>>>);
  static_assert(rg::viewable_range<rg::single_view<std::vector<int>>>);

  static_assert((vw::single(4)[0u]) == 4);

  static constexpr auto s1 = vw::single(7);
  constexpr auto is1 = rg::begin(s1);
  static_assert(*is1 == 7);
  auto e_s1 = rg::end(s1);
  EXPECT_EQ(*--e_s1, 7);

  auto s2 = vw::single(7);
  EXPECT_EQ(*rg::begin(s2), 7);
  *s2.data() = 8;
  EXPECT_EQ(*rg::begin(s2), 8);

  EXPECT_EQ((s1 | vw::all)[0u], 7);
  EXPECT_EQ((vw::single(7) | vw::all)[0u], 7);
}


