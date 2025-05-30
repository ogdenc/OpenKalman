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
 * \brief Tests for \ref collections::single_view and \ref collections::views::single
 */

#include <tuple>
#include "tests.hpp"
#include "basics/classes/equal_to.hpp"
#include "basics/classes/not_equal_to.hpp"
#include "basics/classes/less.hpp"
#include "basics/classes/greater.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "collections/views/single.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

#ifdef __cpp_lib_ranges
  namespace rg = std::ranges;
#else
  namespace rg = OpenKalman::ranges;
#endif


TEST(collections, single_view)
{
  static_assert(collection_view<single_view<std::tuple<int, double>>>);
  static_assert(collection_view<single_view<std::vector<int>>>);
  static_assert(viewable_collection<single_view<std::tuple<int, double>>>);
  static_assert(viewable_collection<single_view<std::vector<int>>>);

  static_assert(std::tuple_size_v<single_view<int>> == 1);
  static_assert(std::tuple_size_v<single_view<std::tuple<int, double>>> == 1);
  static_assert(std::tuple_size_v<single_view<std::tuple<>>> == 1);
  static_assert(std::is_same_v<std::tuple_element_t<0, single_view<int>>, int>);

  static_assert(get(single_view {4}, std::integral_constant<std::size_t, 0>{}) == 4);
  static_assert(get(single_view {std::tuple{}}, std::integral_constant<std::size_t, 0>{}) == std::tuple{});
  static_assert(get(single_view<std::tuple<>> {std::tuple{}}, std::integral_constant<std::size_t, 0>{}) == std::tuple{});
  auto t1 = std::tuple{};
  static_assert(get(single_view {t1}, std::integral_constant<std::size_t, 0>{}) == std::tuple{});

  static_assert((views::single(4)[0u]) == 4);

  static constexpr auto s1 = views::single(7);
  constexpr auto is1 = rg::begin(s1);
  static_assert(*is1 == 7);
  static_assert(*--rg::end(s1) == 7);

  auto s2 = views::single(7);
  EXPECT_EQ(get(s2, 0U), 7);
  EXPECT_EQ(*rg::begin(s2), 7);
  *s2.data() = 8;
  EXPECT_EQ(get(s2, 0U), 8);
  EXPECT_EQ(*rg::begin(s2), 8);

  EXPECT_TRUE(equal_to{}(single_view {4}, std::vector {4}));
  static_assert(equal_to{}(single_view {4}, single_view {4}));
  static_assert(equal_to{}(single_view {4}, 4));
  static_assert(less{}(std::array {4.}, views::single(5.)));
  static_assert(greater{}(6., views::single(5.)));
  static_assert(not_equal_to{}(views::single(5), views::single(6)));
}


