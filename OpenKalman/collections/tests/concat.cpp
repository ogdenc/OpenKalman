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
#include "basics/classes/equal_to.hpp"
#include "basics/classes/not_equal_to.hpp"
#include "basics/classes/less.hpp"
#include "basics/classes/greater.hpp"
#include "basics/classes/less_equal.hpp"
#include "basics/classes/greater_equal.hpp"
#include "basics/compatibility/views.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "collections/views/all.hpp"
#include "collections/views/concat.hpp"
#include "collections/functions/internal/tuple_like_to_tuple.hpp"
#include "collections/functions/compare.hpp"


using namespace OpenKalman;
using namespace OpenKalman::collections;

#ifdef __cpp_lib_ranges
#include<ranges>
  namespace rg = std::ranges;
#else
#include "basics/compatibility/views.hpp"
  namespace rg = OpenKalman::ranges;
#endif

#if __cpp_lib_ranges_concat >= 202403L
  namespace cv = std::ranges::views;
#else
#include "basics/compatibility/views.hpp"
  namespace cv = ranges::views;
#endif

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
  static_assert(get(concat_tuple_view(a, b, c), std::integral_constant<std::size_t, 0>{}) == 1);
  static_assert(get(concat_tuple_view(a, b, c), std::integral_constant<std::size_t, 1>{}) == 1.4f);
  static_assert(get(concat_tuple_view(a, b, c), std::integral_constant<std::size_t, 2>{}) == 2);
  static_assert(get(concat_tuple_view(a, b, c), std::integral_constant<std::size_t, 3>{}) == 2.1);
  static_assert(get(concat_tuple_view(a, b, c), std::integral_constant<std::size_t, 4>{}) == 3.2l);

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

  static_assert(size_of_v<decltype(cv::concat(t1 | views::all, a1 | views::all) | rg::views::transform(std::negate{}) | rg::views::reverse)> == 10);
  static_assert(std::tuple_size_v<decltype(cv::concat(t1 | views::all, t1 | views::all) | views::all)> == 10);
  static_assert(std::is_same_v<std::tuple_element_t<1, decltype(cv::concat(t1 | views::all, t1 | views::all) | views::all)>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<2, decltype(cv::concat(t1 | views::all, t1 | views::all) | views::all)>, double>);
  EXPECT_EQ(OpenKalman::internal::generalized_std_get<7>(cv::concat(t1 | views::all, t1 | views::all) | views::all), 3);
  static_assert(tuple_like<decltype(cv::concat(t1 | views::all, t1 | views::all) | views::all)>);

  static_assert(tuple_like<decltype(views::concat(t1, t1, t1))>);
  static_assert(rg::random_access_range<decltype(views::concat(t1, t1, t1))>);
  static_assert(tuple_like<decltype(views::concat(a1, t1))>);
  static_assert(rg::random_access_range<decltype(views::concat(a1, t1))>);
  static_assert(size_of_v<decltype(views::concat(a1, v1))> == dynamic_size);
  static_assert(not tuple_like<decltype(views::concat(a1, v1))>);
  static_assert(rg::random_access_range<decltype(views::concat(a1, v1))>);

  static_assert(std::is_same_v<std::tuple_element_t<5, decltype(views::concat(t1, t1) | views::all)>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<6, decltype(views::concat(t1, t1) | views::all)>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<7, decltype(views::concat(t1, t1) | views::all)>, float>);

  static_assert(equal_to{}(views::concat(t1, t1), std::tuple{1, 2., 3.f, 4., 5, 1, 2., 3.f, 4., 5} | views::all));
  EXPECT_TRUE(equal_to{}(views::concat(t1, a1), std::tuple{1, 2., 3.f, 4., 5, 1, 2, 3, 4, 5} | views::all));
  EXPECT_TRUE(equal_to{}(views::concat(t1, v1), std::tuple{1, 2., 3.f, 4., 5, 4, 5, 6, 7, 8} | views::all));

  static_assert(less{}(views::concat(t1, t1), std::tuple{1, 2., 3.f, 4., 5, 1, 2., 3.f, 4., 5, 6} | views::all));
  EXPECT_TRUE(less{}(views::concat(a1, t1), std::tuple{1, 2, 3, 4, 5, 1, 2., 3.f, 4., 5, 6} | views::all));
  EXPECT_TRUE(less{}(views::concat(v1, t1), std::tuple{4, 5, 6, 7, 8, 1, 2., 3.f, 4., 5, 6} | views::all));
}

