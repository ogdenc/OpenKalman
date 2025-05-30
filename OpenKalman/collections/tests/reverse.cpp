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

#include <tuple>
#include "tests.hpp"
#include "basics/classes/equal_to.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/views/all.hpp"
#include "collections/views/reverse.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

#ifdef __cpp_lib_ranges
  namespace rg = std::ranges;
#else
  namespace rg = OpenKalman::ranges;
#endif

TEST(collections, reverse_view)
{
  static_assert(collection_view<reverse_view<views::all_t<std::tuple<int, double>>>>);
  static_assert(collection_view<decltype(views::reverse(std::declval<std::tuple<int, double>>()))>);
  static_assert(std::tuple_size_v<reverse_view<views::all_t<std::tuple<int, double>>>> == 2);
  static_assert(std::tuple_size_v<reverse_view<views::all_t<std::tuple<>>>> == 0);
  static_assert(std::is_same_v<std::tuple_element_t<0, reverse_view<views::all_t<std::tuple<float, int, double>>>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<0, reverse_view<views::all_t<std::tuple<float, int, double&>>>>, double&>);
  static_assert(std::is_same_v<std::tuple_element_t<0, reverse_view<views::all_t<std::tuple<float, int, double&&>>>>, double&&>);
  static_assert(std::is_same_v<std::tuple_element_t<1, reverse_view<views::all_t<std::tuple<float, int, double>>>>&, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<1, reverse_view<views::all_t<std::tuple<float, int&&, double>>>>&, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<2, reverse_view<views::all_t<std::tuple<float, int, double>>>>, float>);
  static_assert(std::is_same_v<std::tuple_element_t<2, reverse_view<views::all_t<std::tuple<float&, int, double>>>>&&, float&>);

  static_assert(get(views::reverse(std::tuple{4, 5.}), std::integral_constant<std::size_t, 0>{}) == 5.);
  static_assert(get(views::reverse(std::tuple{4, 5.}), std::integral_constant<std::size_t, 1>{}) == 4);
  static_assert(get(views::reverse(std::array{4, 5}), std::integral_constant<std::size_t, 0>{}) == 5);

  constexpr int a1[5] = {1, 2, 3, 4, 5};
  static_assert(rg::range<reverse_view<views::all_t<int(&)[5]>>>);
  static_assert(rg::range<reverse_view<views::all_t<const int(&)[5]>>>);
  static_assert(std::tuple_size_v<reverse_view<views::all_t<int(&)[5]>>> == 5);
  static_assert(std::tuple_size_v<reverse_view<views::all_t<const int(&)[5]>>> == 5);
  static_assert(std::is_same_v<std::tuple_element_t<1U, reverse_view<views::all_t<int(&)[5]>>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<2U, reverse_view<views::all_t<const int(&)[5]>>>, const int&>);
  static_assert(tuple_like<reverse_view<views::all_t<int(&)[5]>>>);
  static_assert(tuple_like<reverse_view<views::all_t<const int(&)[5]>>>);

  static_assert((views::reverse(a1)[0_uz]) == 5);
  static_assert((views::reverse(a1)[4u]) == 1);
  static_assert(*rg::begin(views::reverse(a1)) == 5);
  static_assert(*(rg::end(views::reverse(a1)) - 1) == 1);
  static_assert(*rg::cbegin(views::reverse(a1)) == 5);
  static_assert(*(rg::cend(views::reverse(a1)) - 1) == 1);
  static_assert(rg::size(views::reverse(a1)) == 5);
  static_assert(rg::begin(views::reverse(a1))[3] == 2);
  static_assert(views::reverse(a1).front() == 5);
  static_assert(views::reverse(a1).back() == 1);
  static_assert(not views::reverse(a1).empty());

  auto a2 = std::array{3, 4, 5};
  EXPECT_EQ(*rg::begin(views::reverse(a2)), 5);
  EXPECT_EQ(*--rg::end(views::reverse(a2)), 3);
  EXPECT_EQ(reverse_view {a2}.front(), 5);
  EXPECT_EQ(reverse_view {a2}.back(), 3);
  EXPECT_EQ((reverse_view {a2}[0u]), 5);
  EXPECT_EQ((reverse_view {a2}[1u]), 4);
  EXPECT_EQ((reverse_view {a2}[2u]), 3);
  EXPECT_EQ(std::addressof(views::reverse {a2}), std::addressof(a2));

  auto v1 = std::vector{3, 4, 5};
  EXPECT_EQ((views::reverse(v1)[0u]), 5);
  EXPECT_EQ((views::reverse(v1)[1u]), 4);
  EXPECT_EQ((views::reverse(v1)[2u]), 3);

  constexpr auto t1 = std::tuple{1, 2, 3, 4, 5};
  static_assert(*rg::begin(views::reverse(t1)) == 5);
  static_assert(views::reverse(t1)[1_uz] == 4);
  static_assert(views::reverse(std::tuple{1, 2, 3, 4, 5})[2_uz] == 3);
  static_assert(*++rg::begin(views::reverse(t1)) == 4);
  static_assert(views::reverse(t1).front() == 5);
  static_assert(views::reverse(std::tuple{1, 2, 3, 4, 5}).front() == 5);
  static_assert(views::reverse(t1).back() == 1);
  static_assert(views::reverse(std::tuple{1, 2, 3, 4, 5}).back() == 1);

  auto at1 = views::reverse(t1);
  auto it1 = rg::begin(at1);
  EXPECT_EQ(it1[1], 4);
  EXPECT_EQ(rg::begin(at1)[1], 4);
  EXPECT_EQ(at1.front(), 5);
  EXPECT_EQ(views::reverse(t1).front(), 5);
  EXPECT_EQ(at1.back(), 1);
  EXPECT_EQ(views::reverse(t1).back(), 1);

  static_assert(equal_to{}(reverse_view {std::tuple{4, 5., 6.f}}, std::tuple{6.f, 5., 4}));
  EXPECT_TRUE(equal_to{}(std::vector{4, 5, 6}, views::reverse(std::tuple{6, 5, 4})));
}

