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
 * \brief Tests for \ref collections::slice_view and \ref collections::views::slice
 */

#include "tests.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/views/slice.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

TEST(collections, slice_view)
{
  using N0 = std::integral_constant<std::size_t, 0>;
  using N1 = std::integral_constant<std::size_t, 1>;
  using N3 = std::integral_constant<std::size_t, 3>;

  using T1 = slice_view<views::all_t<std::tuple<int, double, long double&, float&, unsigned>>, N1, N3>;
  static_assert(collection_view<T1>);
  static_assert(std::tuple_size_v<T1> == 3);
  static_assert(std::tuple_size_v<slice_view<std::tuple<>, N0, N0>> == 0);
  static_assert(std::is_same_v<std::tuple_element_t<0, T1>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, T1>, long double&>);
  static_assert(std::is_same_v<std::tuple_element_t<2, T1>, float&>);

  constexpr auto t1 = std::tuple{4, 5., 6.f, 8.l, 7u};
  static_assert(get(views::slice(t1, N1{}, N3{}), std::integral_constant<std::size_t, 0>{}) == 5.);
  static_assert(get(views::slice(t1, N1{}, N3{}), std::integral_constant<std::size_t, 1>{}) == 6.f);
  static_assert(get(views::slice(t1, N1{}, N3{}), std::integral_constant<std::size_t, 2>{}) == 8.l);
  static_assert(get(views::slice(t1, N1{}, N3{}), std::integral_constant<std::size_t, 2>{}) == 8.l);
  static_assert(get(views::slice(std::move(t1), N1{}, N3{}), std::integral_constant<std::size_t, 2>{}) == 8.l);

  auto v1 = std::vector {3, 4, 5, 6, 7};
  EXPECT_EQ((slice_view(v1, N1{}, N3{}).size()), 3);
  EXPECT_EQ((slice_view(v1, N1{}, 3u).size()), 3);

  EXPECT_EQ((slice_view(v1, N1{}, N3{})[0u]), 4);
  EXPECT_EQ((views::slice(v1, N1{}, 3u)[1u]), 5);
  EXPECT_EQ((views::slice(v1, 1u, N3{})[2u]), 6);
  EXPECT_EQ((views::slice(v1, 1u, 3u)[N0{}]), 4);

#if __cpp_lib_ranges >= 202202L
  EXPECT_EQ((v1 | views::slice(2u, 2u))[0u], 5);
  EXPECT_EQ(((v1 | views::slice(2u, 2u))[std::integral_constant<std::size_t, 1>{}]), 6);
  EXPECT_EQ(((t1 | views::slice(std::integral_constant<std::size_t, 1>{}, 2u))[1u]), 6.f);
  EXPECT_EQ(((t1 | views::slice(1u, std::integral_constant<std::size_t, 3>{}))[2u]), 8.l);
  static_assert(get(std::tuple{7., 8.f, 9} | views::slice(std::integral_constant<std::size_t, 1>{}, std::integral_constant<std::size_t, 2>{}), std::integral_constant<std::size_t, 1>{}) == 9.f);
#endif
}

