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
 * \brief Tests for \ref collections::concat_tuple_view and \ref collections::views::concat
 */

#include "tests.hpp"
#include "basics/basics.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/concepts/tuple_like.hpp"
#include "collections/views/all.hpp"
#include "collections/views/concat.hpp"
#include "collections/functions/internal/tuple_like_to_tuple.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

TEST(collections, concat_tuple_view)
{
  using collections::internal::tuple_like_to_tuple;
  static constexpr std::tuple a {1, 1.4f};
  static constexpr std::tuple b {2, 2.1};
  static constexpr std::tuple c {3.2l};
  static_assert(std::tuple_size_v<decltype(concat_tuple_view(a, b, c))> == 5);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(concat_tuple_view(a, b, c))>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<1, decltype(concat_tuple_view(a, b, c))>, float>);
  static_assert(std::is_same_v<std::tuple_element_t<2, decltype(concat_tuple_view(a, b, c))>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<3, decltype(concat_tuple_view(a, b, c))>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<4, decltype(concat_tuple_view(a, b, c))>, long double>);
  static_assert(get<0>(concat_tuple_view(a, b, c)) == 1);
  static_assert(get<1>(concat_tuple_view(a, b, c)) == 1.4f);
  static_assert(get<2>(concat_tuple_view(a, b, c)) == 2);
  static_assert(get<3>(concat_tuple_view(a, b, c)) == 2.1);
  static_assert(get<4>(concat_tuple_view(a, b, c)) == 3.2l);

  constexpr auto tup1 = concat_tuple_view(std::tuple {1, 2}, std::tuple {3}, std::tuple {4, 5, 6});
  static_assert(tuple_like_to_tuple(tup1) == std::tuple {1, 2, 3, 4, 5, 6});
  static_assert(tuple_like_to_tuple(concat_tuple_view(std::tuple {1, 2}, std::tuple {3}, std::tuple {4, 5, 6})) == std::tuple {1, 2, 3, 4, 5, 6});
}


TEST(collections, concat)
{
  static_assert(collection_view<views::all_t<concat_tuple_view<std::tuple<int, double>, std::tuple<int, double>>>>);

  constexpr auto t1 = std::tuple{1, 2., 3.f, 4., 5};
  constexpr int a1[5] = {1, 2, 3, 4, 5};
  auto v1 = std::vector{4, 5, 6, 7, 8};

  static_assert(size_of_v<decltype(stdex::ranges::views::concat(t1 | views::all, a1 | views::all) | stdex::ranges::views::transform(std::negate{}) | stdex::ranges::views::reverse)> == 10);
  static_assert(std::tuple_size_v<decltype(stdex::ranges::views::concat(t1 | views::all, t1 | views::all) | views::all)> == 10);
  static_assert(std::is_same_v<std::tuple_element_t<1, decltype(stdex::ranges::views::concat(t1 | views::all, t1 | views::all) | views::all)>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<2, decltype(stdex::ranges::views::concat(t1 | views::all, t1 | views::all) | views::all)>, double>);
  EXPECT_EQ(get<7>(stdex::ranges::views::concat(t1 | views::all, t1 | views::all) | views::all), 3);
  static_assert(tuple_like<decltype(stdex::ranges::views::concat(t1 | views::all, t1 | views::all) | views::all)>);

  static_assert(tuple_like<decltype(views::concat(t1, t1, t1))>);
  static_assert(stdex::ranges::random_access_range<decltype(views::concat(t1, t1, t1))>);
  static_assert(tuple_like<decltype(views::concat(a1, t1))>);
  static_assert(stdex::ranges::random_access_range<decltype(views::concat(a1, t1))>);
  static_assert(size_of_v<decltype(views::concat(a1, v1))> == stdex::dynamic_extent);
  static_assert(not tuple_like<decltype(views::concat(a1, v1))>);
  static_assert(stdex::ranges::random_access_range<decltype(views::concat(a1, v1))>);

  static_assert(std::is_same_v<std::tuple_element_t<5, decltype(views::concat(t1, t1) | views::all)>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<6, decltype(views::concat(t1, t1) | views::all)>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<7, decltype(views::concat(t1, t1) | views::all)>, float>);

  static_assert(views::concat(t1, t1)[0U] == 1);
  static_assert(views::concat(t1, t1)[5U] == 1);
  static_assert(views::concat(t1, t1)[7U] == 3.f);
  EXPECT_TRUE(views::concat(t1, a1)[5U] == 1);
  EXPECT_TRUE(views::concat(t1, a1)[7U] == 3);
  EXPECT_TRUE(views::concat(t1, v1)[2U] == 3);
  EXPECT_TRUE(views::concat(t1, v1)[5U] == 4);
  EXPECT_TRUE(views::concat(t1, v1)[7U] == 6);
}

