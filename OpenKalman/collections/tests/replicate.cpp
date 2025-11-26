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
 * \brief Tests for \ref collections::replicate_view and \ref collections::views::replicate
 */

#include <tuple>
#include "tests.hpp"
#include "collections/concepts/sized.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/views/replicate.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

TEST(collections, replicate_view)
{
  static_assert(collection_view<replicate_view<views::all_t<std::tuple<int, double>>, std::integral_constant<std::size_t, 3>>>);
  static_assert(std::tuple_size_v<replicate_view<views::all_t<std::tuple<int, double>>, std::integral_constant<std::size_t, 3>>> == 6);
  static_assert(std::tuple_size_v<replicate_view<views::all_t<std::tuple<>>, std::integral_constant<std::size_t, 3>>> == 0);
  static_assert(std::is_same_v<std::tuple_element_t<0, replicate_view<views::all_t<std::tuple<double, int&, float&>>, std::integral_constant<std::size_t, 2>>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, replicate_view<views::all_t<std::tuple<double, int&, float&>>, std::integral_constant<std::size_t, 2>>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<2, replicate_view<views::all_t<std::tuple<double, int&, float&>>, std::integral_constant<std::size_t, 2>>>, float&>);
  static_assert(std::is_same_v<std::tuple_element_t<3, replicate_view<views::all_t<std::tuple<double, int&, float&>&>, std::integral_constant<std::size_t, 2>>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<4, replicate_view<views::all_t<std::tuple<double, int&, float&>&>, std::integral_constant<std::size_t, 2>>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<5, replicate_view<views::all_t<std::tuple<double, int&, float&>&>, std::integral_constant<std::size_t, 2>>>, float&>);

  using T1 = replicate_view<views::all_t<std::tuple<int, double>>, unsigned>;
  static_assert(std::is_same_v<decltype(*std::declval<stdex::ranges::iterator_t<T1>>()), stdex::iter_reference_t<stdex::ranges::iterator_t<T1>>>);
  static_assert(std::is_same_v<decltype(std::declval<stdex::ranges::iterator_t<T1>>() + std::declval<stdex::iter_difference_t<stdex::ranges::iterator_t<T1>>>()), stdex::ranges::iterator_t<T1>>);
  static_assert(std::is_same_v<decltype(std::declval<stdex::iter_difference_t<stdex::ranges::iterator_t<T1>>>() + std::declval<stdex::ranges::iterator_t<T1>>()), stdex::ranges::iterator_t<T1>>);
  static_assert(std::is_same_v<decltype(std::declval<stdex::ranges::iterator_t<T1>>() - std::declval<stdex::iter_difference_t<stdex::ranges::iterator_t<T1>>>()), stdex::ranges::iterator_t<T1>>);

  constexpr auto t1 = std::tuple{4, 5.};
  static_assert(get<0>(replicate_view {views::all(t1), std::integral_constant<std::size_t, 2>{}}) == 4);
  static_assert(get<1>(views::replicate(t1, std::integral_constant<std::size_t, 2>{})) == 5.);
  static_assert(get<2>(replicate_view {views::all(std::tuple{4, 5.}), std::integral_constant<std::size_t, 2>{}}) == 4);
  static_assert(get<3>(replicate_view {views::all(std::tuple{4, 5.}), std::integral_constant<std::size_t, 2>{}}) == 5.);
  static_assert(get<3>(views::replicate(std::tuple{4, 5.}, std::integral_constant<std::size_t, 2>{})) == 5.);

  static_assert(sized<replicate_view<views::all_t<std::tuple<int, double>>, unsigned>>);
  EXPECT_EQ(get<3>(replicate_view {views::all(std::tuple{4, 5.}), 2u}), 5.);
  EXPECT_EQ(get<3>(views::replicate(t1, 2u)), 5.);

  static_assert(*stdex::ranges::begin(views::replicate(t1, std::integral_constant<std::size_t, 3>{})) == 4.);
  static_assert(views::replicate(t1, std::integral_constant<std::size_t, 3>{})[3_uz] == 5.);
  static_assert(views::replicate(views::all(std::tuple{1, 2, 3, 4, 5}), std::integral_constant<std::size_t, 3>{})[8_uz] == 4.);
  static_assert(*++stdex::ranges::begin(views::replicate(t1, 3u)) == 5.);
  static_assert(*++ ++stdex::ranges::begin(views::replicate(t1, 3u)) == 4.);
  static_assert(views::replicate(t1, 3u).front() == 4.);
  static_assert(views::replicate(views::all(std::tuple{1, 2, 3, 4, 5}), 3u).front() == 1);
  static_assert(views::replicate(t1, std::integral_constant<std::size_t, 3>{}).back() == 5.);
  static_assert(views::replicate(views::all(std::tuple{1, 2, 3, 4, 5}), 3u).back() == 5.);

  auto at1 = views::replicate(t1, 4u);
  auto it1 = stdex::ranges::begin(at1);
  EXPECT_EQ(it1[2], 4.);
  EXPECT_EQ(stdex::ranges::begin(at1)[2], 4.);
  EXPECT_EQ(at1.front(), 4.);
  EXPECT_EQ(views::replicate(t1, 3u).front(), 4.);
  EXPECT_EQ(at1.back(), 5.);
  EXPECT_EQ(views::replicate(t1, 3u).back(), 5.);

  auto v1 = std::vector{3, 4, 5};
  static_assert(sized<replicate_view<views::all_t<std::vector<int>>, unsigned>>);
  EXPECT_EQ((views::replicate(v1, 3u)[3u]), 3);
  EXPECT_EQ((views::replicate(3u)(v1)[7u]), 4);
  EXPECT_EQ((views::replicate(std::integral_constant<std::size_t, 3>{})(v1)[2u]), 5);
  EXPECT_EQ((views::replicate(v1, 3u)[std::integral_constant<std::size_t, 3>{}]), 3);
  EXPECT_EQ((views::replicate(3u)(v1)[std::integral_constant<std::size_t, 7>{}]), 4);
  EXPECT_EQ((views::replicate(std::integral_constant<std::size_t, 3>{})(v1)[std::integral_constant<std::size_t, 2>{}]), 5);

  EXPECT_EQ((v1 | views::replicate(2u))[0u], 3);
  EXPECT_EQ((t1 | views::replicate(std::integral_constant<std::size_t, 3>{}))[1u], 5.);
  EXPECT_EQ((t1 | views::replicate(std::integral_constant<std::size_t, 3>{}))[2u], 4);
  static_assert(get<4>(std::tuple{7., 8.f, 9} | views::replicate(std::integral_constant<std::size_t, 3>{})) == 8.f);

  static_assert(views::replicate(std::tuple{4, 5., 6.f}, std::integral_constant<std::size_t, 2>{})[0U] == 4);
  static_assert(views::replicate(std::tuple{4, 5., 6.f}, std::integral_constant<std::size_t, 2>{})[3U] == 4);
  static_assert(views::replicate(std::tuple{4, 5., 6.f}, std::integral_constant<std::size_t, 2>{})[4U] == 5.);
  static_assert(views::replicate(std::tuple{4, 5., 6.f}, std::integral_constant<std::size_t, 2>{})[8U] == 6.f);
  EXPECT_TRUE(views::replicate(std::tuple{4, 5, 6}, std::integral_constant<std::size_t, 2>{})[0U] == 4);
  EXPECT_TRUE(views::replicate(std::tuple{4, 5, 6}, std::integral_constant<std::size_t, 2>{})[3U] == 4);
  EXPECT_TRUE(views::replicate(std::tuple{4, 5, 6}, std::integral_constant<std::size_t, 2>{})[4U] == 5);
  EXPECT_TRUE(views::replicate(std::tuple{4, 5, 6}, std::integral_constant<std::size_t, 2>{})[8U] == 6);
}

