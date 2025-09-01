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
 * \brief Tests for \ref collections::all_view and \ref collections::views::all
 */

#include "tests.hpp"
#include "basics/basics.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "collections/concepts/settable.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

#include "collections/views/internal/movable_wrapper.hpp"

TEST(collections, movable_wrapper)
{
  using collections::internal::movable_wrapper;
  // owning
  constexpr movable_wrapper mr {6};
  static_assert(mr.get() == 6);
  static_assert(movable_wrapper{5}.get() == 5);
  static_assert(static_cast<int>(mr) == 6);
  static_assert(static_cast<int>(std::as_const(mr)) == 6);
  static_assert(movable_wrapper{5} == 5);
  static_assert(movable_wrapper{5} < 6);
  static_assert(movable_wrapper{5} > 4);
  static_assert(5 == movable_wrapper{5});
  static_assert(6 > movable_wrapper{5});
  static_assert(4 < movable_wrapper{5});

  movable_wrapper ms {7};
  ms = mr;
  EXPECT_EQ(ms, mr);

  // non-owning
  static constexpr auto i = 7;
  constexpr movable_wrapper ml {i};
  static_assert(ml.get() == 7);
  static_assert(movable_wrapper{i}.get() == 7);
  static_assert(movable_wrapper{std::as_const(i)}.get() == 7);
  static_assert(static_cast<int>(ml) == 7);
  static_assert(static_cast<int>(std::as_const(ml)) == 7);
  static_assert(static_cast<const int&>(ml) == 7);
  static_assert(movable_wrapper{i} == 7);
  static_assert(movable_wrapper{i} < 8);
  static_assert(movable_wrapper{i} > 6);
  static_assert(7 == movable_wrapper{i});
  static_assert(6 < movable_wrapper{i});
  static_assert(8 > movable_wrapper{i});
}


#include "collections/views/all.hpp"

TEST(collections, all_view_range)
{
  static_assert(collection<std::vector<double>>);

  static_assert(stdcompat::ranges::range<views::all_t<std::vector<int>&>>);
  static_assert(stdcompat::ranges::range<views::all_t<std::vector<int>>>);
  static_assert(stdcompat::ranges::view<views::all_t<std::vector<double>>>);
  static_assert(stdcompat::ranges::view<views::all_t<std::vector<int>>>);
  static_assert(stdcompat::ranges::view<views::all_t<std::vector<int>&>>);
  static_assert(stdcompat::ranges::view<views::all_t<const std::vector<int>&>>);

  static_assert(sized_random_access_range<views::all_t<std::vector<double>>>);
  static_assert(sized_random_access_range<views::all_t<std::vector<int>>>);
  static_assert(sized_random_access_range<views::all_t<std::vector<int>&>>);
  static_assert(viewable_collection<std::vector<double>>);
  static_assert(viewable_collection<views::all_t<std::vector<double>>>);
  static_assert(collection_view<views::all_t<std::vector<double>>>);
  static_assert(not collection_view<std::vector<double>>);

  auto v1 = std::vector{4, 5, 6, 7, 8};
  static_assert(stdcompat::ranges::random_access_range<decltype(views::all(v1))>);
  static_assert(gettable<0, views::all_t<decltype(v1)>>);
  static_assert(gettable<1, views::all_t<decltype(v1)>>);
  EXPECT_EQ((views::all(v1)[0u]), 4);
  EXPECT_EQ((views::all(v1)[4u]), 8);
  EXPECT_EQ((views::all(std::vector{4, 5, 6, 7, 8})[3u]), 7);
  EXPECT_EQ((views::all(v1)[std::integral_constant<std::size_t, 4>{}]), 8);
  EXPECT_EQ(*stdcompat::ranges::begin(views::all(v1)), 4);
  EXPECT_EQ(*--stdcompat::ranges::end(views::all(v1)), 8);
  EXPECT_EQ(stdcompat::ranges::size(views::all(v1)), 5);

  EXPECT_EQ(views::all(v1).front(), 4);
  EXPECT_EQ(views::all(std::vector{4, 5, 6, 7, 8}).front(), 4);
  EXPECT_EQ(views::all(v1).back(), 8);
  EXPECT_EQ(views::all(std::vector{4, 5, 6, 7, 8}).back(), 8);
  EXPECT_EQ(*stdcompat::ranges::cbegin(views::all(v1)), 4);
  EXPECT_EQ(*--stdcompat::ranges::cend(views::all(v1)), 8);
  EXPECT_TRUE(views::all(std::vector<int>{}).empty());
  EXPECT_FALSE(views::all(std::vector{4, 5, 6, 7, 8}).empty());
  EXPECT_FALSE(views::all(std::vector<int>{}));
  EXPECT_TRUE(views::all(std::vector{4, 5, 6, 7, 8}));

  auto v2 = std::vector{9, 10, 11, 12, 13};
  auto id1_v2 = views::all(v2);
  id1_v2 = views::all(v2);
  EXPECT_EQ((id1_v2[3u]), 12);
  id1_v2 = views::all(v1);
  EXPECT_EQ((id1_v2[3u]), 7);

  auto id2_v1 = views::all(std::vector{4, 5, 6, 7, 8});
  EXPECT_EQ((id2_v1[2u]), 6);

  auto v1b = stdcompat::ranges::begin(v1);
  EXPECT_EQ(*v1b, 4);
  EXPECT_EQ(*++v1b, 5);
  EXPECT_EQ(v1b[3], 8);
  EXPECT_EQ(*--v1b, 4);
  EXPECT_EQ(*--stdcompat::ranges::end(v1), 8);

  EXPECT_EQ((v1 | views::all | stdcompat::ranges::views::transform([](auto a){return a*a;}))[1u], 25);
  EXPECT_EQ((v1 | views::all | stdcompat::ranges::views::reverse)[1u], 7);
  EXPECT_EQ((v1 | views::all | stdcompat::ranges::views::reverse | stdcompat::ranges::views::transform([](auto a){return a*a;}))[1u], 49);
}


TEST(collections, all_view_cpp_array)
{
  constexpr int a1[5] = {1, 2, 3, 4, 5};
  static_assert(std::is_same_v<views::all_t<int(&)[5]>, from_range<int (&)[5]>>);
  static_assert(std::is_same_v<views::all_t<const int(&)[5]>, from_range<const int(&)[5]>>);
  static_assert(std::is_same_v<views::all_t<decltype((a1))>, from_range<const int(&)[5]>>);
  static_assert(stdcompat::ranges::range<views::all_t<int(&)[5]>>);
  static_assert(std::tuple_size_v<views::all_t<int(&)[5]>> == 5);
  static_assert(std::tuple_size_v<stdcompat::ranges::views::all_t<views::all_t<int(&)[5]>>> == 5);
  static_assert(std::tuple_size_v<views::all_t<const int(&)[5]>> == 5);
  static_assert(std::tuple_size_v<stdcompat::ranges::views::all_t<views::all_t<const int(&)[5]>>> == 5);
  static_assert(std::tuple_size_v<views::all_t<decltype(stdcompat::ranges::views::reverse(views::all(a1)))>> == 5);
  static_assert(std::is_same_v<std::tuple_element_t<1, views::all_t<int(&)[5]>>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<1, stdcompat::ranges::views::all_t<views::all_t<int(&)[5]>>>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<2, views::all_t<const int(&)[5]>>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<2, stdcompat::ranges::views::all_t<views::all_t<const int(&)[5]>>>, int>);
  static_assert(views::all(a1).template get<0>() == 1);
  static_assert(views::all(a1).template get<4>() == 5);
  static_assert(OpenKalman::internal::generalized_std_get<0>(views::all(a1)) == 1);
  static_assert(OpenKalman::internal::generalized_std_get<4>(views::all(a1)) == 5);
  static_assert(gettable<0, views::all_t<decltype((a1))>>);
  static_assert(gettable<1, views::all_t<decltype((a1))>>);
  static_assert(uniformly_gettable<views::all_t<int(&)[5]>>);
  static_assert(uniformly_gettable<views::all_t<const int(&)[5]>>);

  static_assert((views::all(a1)[0u]) == 1);
  static_assert((views::all(a1)[4u]) == 5);
  static_assert(*stdcompat::ranges::begin(views::all(a1)) == 1);
  static_assert(*(stdcompat::ranges::end(views::all(a1)) - 1) == 5);
  static_assert(*stdcompat::ranges::cbegin(views::all(a1)) == 1);
  static_assert(*(stdcompat::ranges::cend(views::all(a1)) - 1) == 5);
  static_assert(stdcompat::ranges::size(views::all(a1)) == 5);
  static_assert(stdcompat::ranges::begin(views::all(a1))[2] == 3);
  static_assert(views::all(a1).front() == 1);
  static_assert(views::all(a1).back() == 5);
  static_assert(not views::all(a1).empty());
}


TEST(collections, all_view_tuple_like_range)
{
  static_assert(collection<std::array<double, 4>>);

  static_assert(stdcompat::ranges::view<views::all_t<std::array<double, 4>>>);
  static_assert(stdcompat::ranges::view<views::all_t<std::array<int, 7>>>);

  static_assert(sized_random_access_range<views::all_t<std::array<double, 4>>>);
  static_assert(sized_random_access_range<views::all_t<std::array<int, 7>>>);
  static_assert(sized_random_access_range<views::all_t<const std::array<int, 7>&>>);
  static_assert(viewable_collection<std::array<double, 4>>);
  static_assert(viewable_collection<views::all_t<std::array<double, 4>>>);
  static_assert(collection_view<views::all_t<std::array<double, 4>>>);
  static_assert(not collection_view<std::array<double, 4>>);
}
