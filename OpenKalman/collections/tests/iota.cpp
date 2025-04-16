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
 * \brief Tests for \ref collections::iota_view and \ref collections::views::iota.
 */

#include "tests.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

#include "collections/functions/get.hpp"
#include "collections/functions/get_collection_size.hpp"
#include "collections/views/iota.hpp"

TEST(collections, iota_view)
{
  auto c0 = std::integral_constant<std::size_t, 0>{};
  auto c3 = std::integral_constant<std::size_t, 3>{};

  static_assert(tuple_like<iota_view<decltype(c0), decltype(c3)>>);
  static_assert(not tuple_like<iota_view<decltype(c0), std::size_t>>);
  static_assert(not tuple_like<iota_view<std::size_t, std::size_t>>);
  static_assert(sized_random_access_range<decltype(iota_view {0u, 5u})>);
  static_assert(sized_random_access_range<iota_view<decltype(c0), decltype(c3)>>);
  static_assert(sized_random_access_range<iota_view<std::size_t, decltype(c3)>>);
  static_assert(sized_random_access_range<iota_view<decltype(c0), std::size_t>>);
  static_assert(sized_random_access_range<iota_view<std::size_t, std::size_t>>);

  auto i3 = views::iota(c0, c3);
  EXPECT_EQ(i3.template get<2>()(), 2);
  EXPECT_EQ(i3.size(), 3);
  static_assert(i3.size() == 3);

  using I3 = iota_view<decltype(c0), decltype(c3)>;
  static_assert(std::tuple_size_v<I3> == 3);
  static_assert(std::tuple_element_t<0, I3>::value == 0);
  static_assert(std::tuple_element_t<1, I3>::value == 1);
  static_assert(get(I3{}, 2u) == 2);

  auto c2 = std::integral_constant<std::size_t, 2>{};
  auto c7 = std::integral_constant<std::size_t, 7>{};
  auto i27 = views::iota(c2, c7);
  EXPECT_EQ(get(i27, 3u), 5u);

  using I27 = iota_view<decltype(c2), decltype(c7)>;
  static_assert(std::tuple_size_v<I27> == 7);
  static_assert(std::tuple_element_t<0, I27>::value == 2);
  static_assert(std::tuple_element_t<6, I27>::value == 8);
  static_assert(get(I27{}, 3u) == 5);

  EXPECT_EQ(views::iota(0u, 3u).begin()[2], 2);
  EXPECT_EQ(views::iota(c0, 3u).begin()[2], 2);
  EXPECT_EQ(views::iota(0u, c3).begin()[2], 2);
  static_assert(views::iota(0u, c3).size() == 3);

  auto i9 = views::iota(0u, 9u);
  EXPECT_EQ(get_collection_size(i9), 9);
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


  constexpr auto iota_range1 = views::iota(3u, 8u);
  static_assert(not tuple_like<decltype(iota_range1)>);
  static_assert(sized_random_access_range<decltype(iota_range1)>);
  static_assert(get(iota_range1, std::integral_constant<std::size_t, 0>{}) == 3);
  static_assert(get(iota_range1, 0u) == 3);
  static_assert(get(iota_range1, 3u) == 6);
  static_assert(get(iota_range1, 4u) == 7);

  auto c8 = std::integral_constant<std::size_t, 8>{};
  constexpr auto iota_tup1 = views::iota(c3, c8);
  static_assert(tuple_like<decltype(iota_tup1)>);
  static_assert(sized_random_access_range<decltype(iota_tup1)>);
  static_assert(get(iota_tup1, std::integral_constant<std::size_t, 0>{}) == 3);
  static_assert(get(iota_tup1, std::integral_constant<std::size_t, 3>{}) == 6);

  static_assert(get(iota_tup1, 3u) == 6);
  static_assert(get(iota_tup1, 4u) == 7);

}
