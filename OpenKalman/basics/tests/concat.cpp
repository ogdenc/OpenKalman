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
 * \brief Tests for global views
 */

#include <vector>
#include "tests.hpp"
#if __cpp_lib_ranges_concat >= 202403L
#include <ranges>
#else
#include "basics/compatibility/views/concat.hpp"
#endif

using namespace OpenKalman;

#ifdef __cpp_lib_ranges
  namespace rg = std::ranges;
#else
  namespace rg = OpenKalman::ranges;
#endif
  namespace vw = rg::views;

#if __cpp_lib_ranges_concat >= 202403L
  using std::ranges::concat_view;
#else
  using ranges::concat_view;
  using ranges::views::concat;
  #endif

TEST(basics, concat_view)
{
  std::vector v1 {3, 4, 5};
  std::vector v2 {6.f, 7.f, 8.f};
  auto cat1 = concat(v1, v2);
  static_assert(rg::random_access_range<decltype(cat1)>);
  auto itv1 = rg::begin(cat1);
  EXPECT_EQ(*itv1, 3);
  EXPECT_EQ(*rg::begin(cat1), 3);
  EXPECT_EQ(*++itv1, 4);
  EXPECT_EQ(*++itv1, 5);
  EXPECT_EQ(*++itv1, 6);
  EXPECT_EQ(*++itv1, 7);
  EXPECT_EQ(*--itv1, 6);
  EXPECT_EQ(itv1[2], 8);
  EXPECT_EQ(cat1[0u], 3);
  EXPECT_EQ(cat1[1u], 4);
  EXPECT_EQ(cat1[2u], 5);
  EXPECT_EQ((cat1[std::integral_constant<std::size_t, 3>{}]), 6);
  EXPECT_EQ((cat1[std::integral_constant<std::size_t, 4>{}]), 7);
  EXPECT_EQ((cat1[std::integral_constant<std::size_t, 5>{}]), 8);

  auto cat2 = concat(std::vector {2, 3, 4}, std::vector {5., 6., 7.});
  static_assert(rg::random_access_range<decltype(cat2)>);
  static_assert(not std::is_const_v<std::remove_reference_t<decltype(cat2)>>);
  auto itv2 = rg::begin(cat2);
  EXPECT_EQ(*itv2, 2);
  EXPECT_EQ(*rg::begin(cat2), 2);
  EXPECT_EQ(*++itv2, 3);
  EXPECT_EQ(*++ ++ ++itv2, 6);
  EXPECT_EQ(cat2[0u], 2);
  EXPECT_EQ(cat2[2u], 4);
  EXPECT_EQ((cat2[std::integral_constant<std::size_t, 3>{}]), 5);
  EXPECT_EQ((cat2[std::integral_constant<std::size_t, 5>{}]), 7);

  constexpr double a1[3] = {5.3, 5.4, 5.5};
  auto cat5 = concat(v1, a1, v2);
  static_assert(rg::random_access_range<decltype(cat5)>);
  auto itv5 = rg::begin(cat5);
  EXPECT_EQ(*itv5, 3);
  EXPECT_EQ(*rg::begin(cat5), 3);
  EXPECT_EQ(*++itv5, 4);
  EXPECT_EQ(itv5[3], 5.4);
  EXPECT_EQ(*++itv5, 5);
  EXPECT_EQ(*++itv5, 5.3);
  EXPECT_EQ(itv5[4], 7);
  EXPECT_EQ(itv5[5], 8);

}

