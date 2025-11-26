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
#include "collections/functions/lexicographical_compare_three_way.hpp"
#include "collections/views/all.hpp"
#include "collections/views/iota.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

namespace
{
  constexpr auto n0 = std::integral_constant<std::size_t, 0>{};
  constexpr auto n1 = std::integral_constant<std::size_t, 1>{};
  constexpr auto n2 = std::integral_constant<std::size_t, 2>{};
  constexpr auto n3 = std::integral_constant<std::size_t, 3>{};
}


TEST(collections, lexicographical_compare_views)
{
  static_assert(stdex::is_eq(lexicographical_compare_three_way(std::tuple{4, 5.} | views::all, std::tuple{4, 5.} | views::all)));
  static_assert(stdex::is_lt(lexicographical_compare_three_way(views::all(std::tuple{4, 5.}), std::tuple{4, 6.} | views::all)));
  static_assert(stdex::is_lt(lexicographical_compare_three_way(std::tuple{4, 5.} | views::all, views::all(std::tuple{4, 5., 1}))));
  static_assert(stdex::is_lt(lexicographical_compare_three_way(views::all(std::tuple{4, 5.}), views::all(std::tuple{4, 6.}))));
  static_assert(stdex::is_gt(lexicographical_compare_three_way(views::all(std::tuple{4, 6.}), std::tuple{4, 5.} | views::all)));
  static_assert(stdex::is_lteq(lexicographical_compare_three_way(views::all(std::tuple{4, 5.}), std::tuple{4, 6.} | views::all)));
  static_assert(stdex::is_gteq(lexicographical_compare_three_way(views::all(std::tuple{4, 6.}), std::tuple{4, 5.} | views::all)));
  static_assert(stdex::is_neq(lexicographical_compare_three_way(views::all(std::tuple{4, 5.}), std::tuple{4, 6.} | views::all)));

  auto v1 = std::vector{4, 5, 6, 7, 8};
  EXPECT_TRUE(stdex::is_eq(lexicographical_compare_three_way(views::all(std::vector{4, 5, 6}), std::vector{4, 5, 6})));
  EXPECT_TRUE(stdex::is_eq(lexicographical_compare_three_way(views::all(v1), v1)));
  EXPECT_TRUE(stdex::is_eq(lexicographical_compare_three_way(v1, views::all(v1))));
  EXPECT_TRUE(stdex::is_lt(lexicographical_compare_three_way(views::all(std::vector{4, 5, 6}), std::vector{4, 5, 7})));
  EXPECT_TRUE(stdex::is_lteq(lexicographical_compare_three_way(views::all(std::vector{4, 5.}), std::vector{4, 5.})));

  EXPECT_TRUE(stdex::is_eq(lexicographical_compare_three_way(std::array{4, 5, 6}, views::all(std::array{4, 5, 6}))));
  EXPECT_TRUE(stdex::is_lt(lexicographical_compare_three_way(std::array{4, 5, 6}, views::all(std::array{4, 5, 6, 1}))));

  EXPECT_TRUE(stdex::is_eq(lexicographical_compare_three_way(views::all(std::tuple{4, 5, 6} | views::all), views::all(std::vector{4, 5, 6}))));
  EXPECT_TRUE(stdex::is_eq(lexicographical_compare_three_way(views::all(std::tuple{4, 5, 6} | views::all), std::vector{4, 5, 6})));
  EXPECT_TRUE(stdex::is_eq(lexicographical_compare_three_way(views::all(std::vector{4, 5, 6}), views::all(std::tuple{4, 5, 6} | views::all))));
  EXPECT_TRUE(stdex::is_eq(lexicographical_compare_three_way(std::vector{4, 5, 6}, views::all(std::tuple{4, 5, 6}))));
  EXPECT_TRUE(stdex::is_lt(lexicographical_compare_three_way(views::all(std::vector{4, 5, 6}), views::all(std::tuple{4, 5, 7}))));
  EXPECT_TRUE(stdex::is_lt(lexicographical_compare_three_way(views::all(std::tuple{4, 5, 6}), views::all(std::vector{4, 5, 6, 1}))));
  EXPECT_TRUE(stdex::is_lt(lexicographical_compare_three_way(views::all(std::tuple{4, 5, 6}), std::vector{4, 5, 7})));
  EXPECT_TRUE(stdex::is_lt(lexicographical_compare_three_way(std::vector{4, 5, 6}, views::all(std::tuple{4, 5, 7}))));
  EXPECT_TRUE(stdex::is_gteq(lexicographical_compare_three_way(views::all(std::tuple{4, 5.}), std::vector{4, 5.})));

  EXPECT_TRUE(stdex::is_lt(lexicographical_compare_three_way(views::all(std::vector{0U, 1U, 2U}), views::iota())));
  EXPECT_TRUE(stdex::is_lt(lexicographical_compare_three_way(views::all(std::vector{0U, 0U, 2U}), views::iota())));
  EXPECT_TRUE(stdex::is_gt(lexicographical_compare_three_way(views::all(std::vector{0U, 2U, 2U}), views::iota())));
  EXPECT_TRUE(stdex::is_gt(lexicographical_compare_three_way(views::iota(), views::all(std::vector{0U, 0U, 2U}))));
}

#include "collections/functions/compare_indices.hpp"

TEST(collections, compare_indices)
{
  static_assert(compare_indices(std::tuple{}, std::tuple{}));
  EXPECT_TRUE(compare_indices(std::vector<std::size_t>{}, std::tuple{}));
  EXPECT_TRUE(compare_indices(std::tuple{}, std::vector<std::size_t>{}));
  EXPECT_TRUE(compare_indices(std::vector<std::size_t>{}, std::vector<std::size_t>{}));

  static_assert(compare_indices(std::tuple{4U, 5U}, std::tuple{4U, 5U}));
  static_assert(compare_indices(std::array{4U, 5U}, std::array{4U, 5U}));
  EXPECT_TRUE(compare_indices(std::vector{4U, 5U}, std::vector{4U, 5U}));
  EXPECT_TRUE(compare_indices<&stdex::is_lt>(std::vector{4U, 5U}, std::vector{5U, 6U}));
  EXPECT_TRUE(compare_indices<&stdex::is_gt>(std::vector{4U, 5U}, std::vector{3U, 4U}));
  EXPECT_FALSE(compare_indices<&stdex::is_gt>(std::vector{4U, 5U}, std::vector{3U, 6U}));
  EXPECT_TRUE(compare_indices<&stdex::is_neq>(std::vector{4U, 5U}, std::vector{3U, 6U}));
  EXPECT_TRUE(compare_indices(std::vector{4U, 5U}, std::tuple{4U, 5U}));
  static_assert(compare_indices(std::tuple{4U, 5U} | views::all, std::tuple{4U, 5U} | views::all));

  static_assert(compare_indices(std::tuple{n2, n3}, std::tuple{n2, n3}));
  static_assert(compare_indices(std::tuple{n2, n3}, std::tuple{2U, 3U}));

  static_assert(compare_indices(std::tuple{}, std::tuple{4U, 5U}));
  static_assert(compare_indices(std::tuple{4U, 5U}, std::tuple{4U, 5U, 6U}));
  static_assert(compare_indices(std::tuple{4U, 5U, 6U, 6U}, std::tuple{4U, 5U}));
  EXPECT_TRUE(compare_indices(std::vector{4U, 5U}, std::tuple{4U, 5U, 6U}));
  EXPECT_TRUE(compare_indices(std::vector{4U, 5U, 6U, 6U}, std::tuple{4U, 5U}));
  EXPECT_TRUE(compare_indices(std::vector{4U, 5U}, std::vector{4U, 5U, 6U}));
  EXPECT_TRUE(compare_indices(std::vector{4U, 5U, 6U, 6U}, std::vector{4U, 5U}));
  EXPECT_TRUE(compare_indices(std::vector{4U, 5U}, std::vector{4U, 5U, 6U}));
  EXPECT_TRUE(compare_indices(std::vector{4U, 5U, 6U, 6U}, std::vector{4U, 5U}));
  EXPECT_TRUE(compare_indices<&stdex::is_lt>(std::vector{4U, 5U}, std::vector{5U, 6U, 0U}));
  EXPECT_TRUE(compare_indices<&stdex::is_lt>(std::vector{4U, 5U}, std::tuple{5U, 6U, 0U}));
  static_assert(compare_indices<&stdex::is_lt>(std::tuple{4U, 5U}, std::tuple{5U, 6U, 0U}));
  static_assert(compare_indices<&stdex::is_lteq>(std::tuple{4U, 5U}, std::tuple{4U, 5U, 0U}));
  static_assert(compare_indices<&stdex::is_neq>(std::tuple{4U, 5U}, std::tuple{3U, 6U, 0U}));
  static_assert(compare_indices<&stdex::is_gt>(std::tuple{5U, 6U, 0U}, std::tuple{4U, 5U}));

  static_assert(compare_indices(std::tuple{}, views::iota()));
  static_assert(compare_indices(std::tuple{0U, 1U, 2U}, views::iota(n0, n3)));
  static_assert(compare_indices(std::tuple{0U, 1U, 2U}, views::iota(0U, 3U)));
  static_assert(compare_indices(std::tuple{0U, 1U, 2U}, views::iota()));

  static_assert(get_element(views::iota(1U, 5U), n0) == 1U);

  static_assert(compare_indices(std::tuple{1U, 2U, 3U}, views::iota(1U)));
  static_assert(compare_indices(std::tuple{n1, n2, n3}, views::iota(n1)));
  static_assert(compare_indices<&stdex::is_lt>(views::iota(), std::tuple{1U, 2U, 3U}));
  static_assert(compare_indices<&stdex::is_gt>(std::tuple{1U, 2U, 3U}, views::iota()));
  static_assert(compare_indices<&stdex::is_lt>(std::tuple{0U, 1U, 2U}, views::iota(n1)));
  static_assert(compare_indices<&stdex::is_lt>(std::tuple{0U, n1, 2U}, views::iota(n1)));
  static_assert(compare_indices<&stdex::is_gt>(views::iota(n1), std::tuple{0U, 1U, n2}));

  EXPECT_TRUE(compare_indices(std::vector<std::size_t>{}, views::iota()));
  EXPECT_TRUE(compare_indices(std::vector{0U, 1U, 2U}, views::iota()));
  EXPECT_TRUE(compare_indices<&stdex::is_gt>(std::vector{1U, 2U, 3U}, views::iota()));
  EXPECT_TRUE(compare_indices<&stdex::is_lt>(std::vector{0U, 1U, 2U}, views::iota(n1)));
  EXPECT_TRUE(compare_indices<&stdex::is_gt>(std::vector{1U, 2U, 3U}, views::iota()));
  EXPECT_TRUE(compare_indices<&stdex::is_gt>(views::iota(n1), std::vector{0U, 1U, 2U}));
}