/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for internal utilities
 */

#include <tuple>
#include <array>
#include "tests.hpp"
#include "collections/functions/get.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

#include "collections/functions/internal/tuple_like_to_tuple.hpp"

#include "collections/functions/internal/tuple_slice.hpp"

TEST(collections, tuple_slice)
{
  using OpenKalman::collections::internal::tuple_like_to_tuple;
  std::tuple t {1, "c", 5.0, 6.0};
  static_assert(std::tuple_size_v<decltype(tuple_slice<0, 0>(t))> == 0);
  static_assert(std::tuple_size_v<decltype(tuple_slice<0, 2>(t))> == 2);
  static_assert(std::tuple_size_v<decltype(tuple_slice<1, 3>(t))> == 2);
  static_assert(std::tuple_size_v<decltype(tuple_slice<3, 3>(t))> == 0);
  static_assert(std::tuple_size_v<decltype(tuple_slice<2, 4>(t))> == 2);

  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(tuple_slice<0, 1>(t))>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(tuple_slice<1, 2>(t))>, const char*>);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(tuple_slice<2, 3>(t))>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(tuple_slice<3, 4>(t))>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(tuple_slice<1, 3>(t))>, const char*>);
  static_assert(std::is_same_v<std::tuple_element_t<1, decltype(tuple_slice<1, 3>(t))>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(tuple_slice<2, 4>(t))>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, decltype(tuple_slice<2, 4>(t))>, double>);

  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(tuple_slice<1, 3>(std::tuple {1, "c", 5.0, 6.0}))>, const char*>);
  static_assert(std::is_same_v<std::tuple_element_t<1, decltype(tuple_slice<1, 3>(std::tuple {1, "c", 5.0, 6.0}))>, double>);

  static_assert(std::is_same_v<decltype(get(tuple_slice<1, 3>(t), std::integral_constant<std::size_t, 0>{})), const char*&>);
  static_assert(std::is_same_v<decltype(get(tuple_slice<1, 3>(t), std::integral_constant<std::size_t, 1>{})), double&>);

  constexpr std::array a {1, 2, 3, 4, 5};
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(tuple_slice<0, 2>(a))>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<1, decltype(tuple_slice<1, 3>(a))>, int>);
  static_assert(get(tuple_slice<1, 3>(a), std::integral_constant<std::size_t, 0>{}) == 2);

  static_assert(tuple_like_to_tuple(tuple_slice<1, 4>(a)) == std::tuple {2, 3, 4});
  static_assert(tuple_like_to_tuple(tuple_slice<1, 4>(std::array{1, 2, 3, 4, 5})) == std::tuple {2, 3, 4});
}


#include "collections/functions/internal/tuple_reverse.hpp"

TEST(collections, tuple_reverse)
{
  using OpenKalman::collections::internal::tuple_like_to_tuple;
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(tuple_reverse<std::tuple<std::integral_constant<std::size_t, 2>>>())>, std::integral_constant<std::size_t, 2>>);

  static constexpr std::tuple ta {1, 'c', 5.0, 6.0};
  constexpr auto rta = tuple_reverse(ta);
  static_assert(get(rta, std::integral_constant<std::size_t, 0>{}) == 6.0);
  static_assert(get(rta, std::integral_constant<std::size_t, 1>{}) == 5.0);
  static_assert(get(rta, std::integral_constant<std::size_t, 2>{}) == 'c');
  static_assert(get(rta, std::integral_constant<std::size_t, 3>{}) == 1);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(tuple_reverse(ta))>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, decltype(tuple_reverse(ta))>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<2, decltype(tuple_reverse(ta))>, char>);
  static_assert(std::is_same_v<std::tuple_element_t<3, decltype(tuple_reverse(ta))>, int>);
  int n0 = 1;
  char n1 = 'c';
  double n2 = 5.0, n3 = 6.0;
  auto tb = std::forward_as_tuple(n0, n1, n2, n3);
  auto rtb = tuple_reverse(tb);
  EXPECT_TRUE(get(rtb, std::integral_constant<std::size_t, 0>{}) == n3);
  EXPECT_TRUE(get(rtb, std::integral_constant<std::size_t, 1>{}) == n2);
  EXPECT_TRUE(get(rtb, std::integral_constant<std::size_t, 2>{}) == n1);
  EXPECT_TRUE(get(rtb, std::integral_constant<std::size_t, 3>{}) == n0);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(tuple_reverse(tb))>, double&>);
  static_assert(std::is_same_v<std::tuple_element_t<1, decltype(tuple_reverse(tb))>, double&>);
  static_assert(std::is_same_v<std::tuple_element_t<2, decltype(tuple_reverse(tb))>, char&>);
  static_assert(std::is_same_v<std::tuple_element_t<3, decltype(tuple_reverse(tb))>, int&>);

  static_assert(tuple_like_to_tuple(tuple_reverse(ta)) == std::tuple {6.0, 5.0, 'c', 1});
  static_assert(tuple_like_to_tuple(tuple_reverse(std::tuple{1, 'c', 5.0, 6.0})) == std::tuple {6.0, 5.0, 'c', 1});
}


#include "collections/functions/internal/tuple_flatten.hpp"

TEST(collections, tuple_flatten)
{
  constexpr auto fta = tuple_flatten(std::tuple {0, std::tuple{1, std::tuple{2, 3}}, 4, 5});
  static_assert(get(fta, std::integral_constant<std::size_t, 0>{}) == 0);
  static_assert(get(fta, std::integral_constant<std::size_t, 1>{}) == 1);
  static_assert(get(fta, std::integral_constant<std::size_t, 2>{}) == 2);
  static_assert(get(fta, std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get(fta, std::integral_constant<std::size_t, 4>{}) == 4);
  static_assert(get(fta, std::integral_constant<std::size_t, 5>{}) == 5);

  static_assert(get(tuple_flatten(std::tuple{7}), std::integral_constant<std::size_t, 0>{}) == 7);

  constexpr auto ftb = tuple_flatten(std::array{3, 4, 5});
  static_assert(get(ftb, std::integral_constant<std::size_t, 2>{}) == 5);
  static_assert(std::is_same_v<const decltype(ftb), const std::array<int, 3>>);

  constexpr auto ftc = tuple_flatten(std::array{std::tuple{0, 1}, std::tuple{2, 3}, std::tuple{4, 5}});
  static_assert(get(ftc, std::integral_constant<std::size_t, 0>{}) == 0);
  static_assert(get(ftc, std::integral_constant<std::size_t, 1>{}) == 1);
  static_assert(get(ftc, std::integral_constant<std::size_t, 2>{}) == 2);
  static_assert(get(ftc, std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get(ftc, std::integral_constant<std::size_t, 4>{}) == 4);
  static_assert(get(ftc, std::integral_constant<std::size_t, 5>{}) == 5);
}

