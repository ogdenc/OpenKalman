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
 * \brief Tests for \ref ranges::reverse_view and \ref ranges::vw::reverse.
 */

#include <vector>
#include "tests.hpp"
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/views/all.hpp"
#include "basics/compatibility/views/reverse.hpp"
#endif

using namespace OpenKalman;

#ifdef __cpp_lib_ranges
namespace rg = std::ranges;
#else
namespace rg = OpenKalman::ranges;
#endif
namespace vw = rg::views;

TEST(basics, reverse_view)
{
  constexpr int a1[5] = {1, 2, 3, 4, 5};
  static_assert(rg::range<rg::reverse_view<vw::all_t<int(&)[5]>>>);
  static_assert(rg::range<rg::reverse_view<vw::all_t<const int(&)[5]>>>);

  static_assert((vw::reverse(a1)[0_uz]) == 5);
  static_assert((vw::reverse(a1)[4u]) == 1);
  static_assert(*rg::begin(vw::reverse(a1)) == 5);
  static_assert(*(rg::end(vw::reverse(a1)) - 1) == 1);
  static_assert(vw::reverse(a1).size() == 5);
  static_assert(*rg::cbegin(vw::reverse(a1)) == 5);
  static_assert(*(rg::cend(vw::reverse(a1)) - 1) == 1);
  static_assert(rg::size(vw::reverse(a1)) == 5);
  static_assert(rg::begin(vw::reverse(a1))[3] == 2);
  static_assert(vw::reverse(a1).front() == 5);
  static_assert(vw::reverse(a1).back() == 1);
  static_assert(not vw::reverse(a1).empty());

  auto a2 = std::array{3, 4, 5};
  EXPECT_EQ(*rg::begin(vw::reverse(a2)), 5);
  EXPECT_EQ(*--rg::end(vw::reverse(a2)), 3);
  EXPECT_EQ(rg::reverse_view(a2).front(), 5);
  EXPECT_EQ(rg::reverse_view {a2}.back(), 3);
  EXPECT_EQ((rg::reverse_view {a2}[0u]), 5);
  EXPECT_EQ((rg::reverse_view {a2}[1u]), 4);
  EXPECT_EQ((rg::reverse_view {a2}[2u]), 3);

  auto v1 = std::vector{3, 4, 5};
  EXPECT_EQ((vw::reverse(v1)[0u]), 5);
  EXPECT_EQ((vw::reverse(v1)[1u]), 4);
  EXPECT_EQ((vw::reverse(v1)[2u]), 3);

  EXPECT_EQ((v1 | vw::reverse)[0], 5);
  EXPECT_EQ((a1 | vw::reverse)[1], 4);

}

