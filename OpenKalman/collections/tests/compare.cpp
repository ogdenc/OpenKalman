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
 * \brief Tests for \ref collections::all_view and \ref collections::views::all involving \ref collections::views::from_tuple_like
 */

#include "tests.hpp"
#include "basics/basics.hpp"
#include "collections/concepts/settable.hpp"
#include "collections/views/all.hpp"
#include "collections/functions/lexicographical_compare_three_way.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;


TEST(collections, lexicographical_compare_views)
{
  static_assert(stdcompat::is_eq(lexicographical_compare_three_way(std::tuple{4, 5.} | views::all, std::tuple{4, 5.} | views::all)));
  static_assert(stdcompat::is_lt(lexicographical_compare_three_way(views::all(std::tuple{4, 5.}), std::tuple{4, 6.} | views::all)));
  static_assert(stdcompat::is_lt(lexicographical_compare_three_way(std::tuple{4, 5.} | views::all, views::all(std::tuple{4, 5., 1}))));
  static_assert(stdcompat::is_lt(lexicographical_compare_three_way(views::all(std::tuple{4, 5.}), views::all(std::tuple{4, 6.}))));
  static_assert(stdcompat::is_gt(lexicographical_compare_three_way(views::all(std::tuple{4, 6.}), std::tuple{4, 5.} | views::all)));
  static_assert(stdcompat::is_lteq(lexicographical_compare_three_way(views::all(std::tuple{4, 5.}), std::tuple{4, 6.} | views::all)));
  static_assert(stdcompat::is_gteq(lexicographical_compare_three_way(views::all(std::tuple{4, 6.}), std::tuple{4, 5.} | views::all)));
  static_assert(stdcompat::is_neq(lexicographical_compare_three_way(views::all(std::tuple{4, 5.}), std::tuple{4, 6.} | views::all)));

  auto v1 = std::vector{4, 5, 6, 7, 8};
  EXPECT_TRUE(stdcompat::is_eq(lexicographical_compare_three_way(views::all(std::vector{4, 5, 6}), std::vector{4, 5, 6})));
  EXPECT_TRUE(stdcompat::is_eq(lexicographical_compare_three_way(views::all(v1), v1)));
  EXPECT_TRUE(stdcompat::is_eq(lexicographical_compare_three_way(v1, views::all(v1))));
  EXPECT_TRUE(stdcompat::is_lt(lexicographical_compare_three_way(views::all(std::vector{4, 5, 6}), std::vector{4, 5, 7})));
  EXPECT_TRUE(stdcompat::is_lteq(lexicographical_compare_three_way(views::all(std::vector{4, 5.}), std::vector{4, 5.})));

  EXPECT_TRUE(stdcompat::is_eq(lexicographical_compare_three_way(std::array{4, 5, 6}, views::all(std::array{4, 5, 6}))));
  EXPECT_TRUE(stdcompat::is_lt(lexicographical_compare_three_way(std::array{4, 5, 6}, views::all(std::array{4, 5, 6, 1}))));

  EXPECT_TRUE(stdcompat::is_eq(lexicographical_compare_three_way(views::all(std::tuple{4, 5, 6} | views::all), views::all(std::vector{4, 5, 6}))));
  EXPECT_TRUE(stdcompat::is_eq(lexicographical_compare_three_way(views::all(std::tuple{4, 5, 6} | views::all), std::vector{4, 5, 6})));
  EXPECT_TRUE(stdcompat::is_eq(lexicographical_compare_three_way(views::all(std::vector{4, 5, 6}), views::all(std::tuple{4, 5, 6} | views::all))));
  EXPECT_TRUE(stdcompat::is_eq(lexicographical_compare_three_way(std::vector{4, 5, 6}, views::all(std::tuple{4, 5, 6}))));
  EXPECT_TRUE(stdcompat::is_lt(lexicographical_compare_three_way(views::all(std::vector{4, 5, 6}), views::all(std::tuple{4, 5, 7}))));
  EXPECT_TRUE(stdcompat::is_lt(lexicographical_compare_three_way(views::all(std::tuple{4, 5, 6}), views::all(std::vector{4, 5, 6, 1}))));
  EXPECT_TRUE(stdcompat::is_lt(lexicographical_compare_three_way(views::all(std::tuple{4, 5, 6}), std::vector{4, 5, 7})));
  EXPECT_TRUE(stdcompat::is_lt(lexicographical_compare_three_way(std::vector{4, 5, 6}, views::all(std::tuple{4, 5, 7}))));
  EXPECT_TRUE(stdcompat::is_gteq(lexicographical_compare_three_way(views::all(std::tuple{4, 5.}), std::vector{4, 5.})));
}
