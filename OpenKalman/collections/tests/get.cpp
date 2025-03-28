/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for \ref collections::get_collection_element.
 */

#include <type_traits>
#include "collections/tests/tests.hpp"
#include "collections/functions/get.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

namespace C1
{
  struct C1
  {
    int n1 = 4;
  };

  template<std::size_t i>
  constexpr auto get(const C1& c1) { return c1.n1 + i; }
}

namespace std
{
  template<> struct tuple_size<C1::C1> : std::integral_constant<std::size_t, 7> {};
  template<size_t i> struct tuple_element<i, C1::C1> { using type = int; };
}


struct C2
{
  template<std::size_t i>
  constexpr auto get() const { return 7_uz + i; }
};

namespace std
{
  template<> struct tuple_size<C2> : std::integral_constant<std::size_t, 7> {};
  template<size_t i> struct tuple_element<i, C2> { using type = std::size_t; };
}


TEST(collections, get)
{
  // array
  constexpr int a1[5] = {0, 1, 2, 3, 4};
  static_assert(get(a1, std::integral_constant<std::size_t, 2>{}) == 2);
  static_assert(get(a1, std::integral_constant<std::size_t, 3>{}) == 3);

  // member
  constexpr C2 c2;
  static_assert(get(c2, std::integral_constant<std::size_t, 0>{}) == 7);
  static_assert(get(c2, std::integral_constant<std::size_t, 1>{}) == 8);

  // ADL
  constexpr C1::C1 c1;
  static_assert(get(c1, std::integral_constant<std::size_t, 0>{}) == 4);
  static_assert(get(c1, std::integral_constant<std::size_t, 1>{}) == 5);

  // std
  constexpr auto tup1 = std::tuple {0, 1.f, 2u, std::integral_constant<std::size_t, 3>{}};
  static_assert(get(tup1, std::integral_constant<std::size_t, 0>{}) == 0);
  static_assert(get(tup1, std::integral_constant<std::size_t, 1>{}) == 1.f);
  static_assert(get(tup1, std::integral_constant<std::size_t, 2>{}) == 2u);
  static_assert(get(tup1, std::integral_constant<std::size_t, 3>{}) == 3);

  static_assert(get(std::array{1, 2, 3, 4}, std::integral_constant<std::size_t, 2>{}) == 3);
  static_assert(get(std::array{1, 2, 3, 4}, std::integral_constant<std::size_t, 3>{}) == 4);

  std::vector v1 = {1, 2, 3, 4};
  EXPECT_EQ(get(v1, std::integral_constant<std::size_t, 2>{}), 3);
  EXPECT_EQ(get(v1, std::integral_constant<std::size_t, 3>{}), 4);

  std::initializer_list l1 = {1, 2, 3, 4};
  EXPECT_EQ(get(l1, std::integral_constant<std::size_t, 2>{}), 3);
  EXPECT_EQ(get(l1, std::integral_constant<std::size_t, 3>{}), 4);


  static_assert(tuple_like<decltype(tup1)>);
  static_assert(not sized_random_access_range<decltype(tup1)>);
  static_assert(get(tup1, std::integral_constant<std::size_t, 0>{}) == 0);
  static_assert(get(tup1, std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get<std::size_t>(tup1, 1u) == 1);
  static_assert(get<std::size_t>(tup1, 2u) == 2);

  auto range1 = std::vector<std::size_t> {3, 4, 5, 6, 7, 8};
  static_assert(not tuple_like<decltype(range1)>);
  static_assert(sized_random_access_range<decltype(range1)>);
  EXPECT_EQ(get(range1, std::integral_constant<std::size_t, 0>{}), 3);
  EXPECT_EQ(get(range1, 0u), 3);
  EXPECT_EQ(get(range1, 3u), 6);
  EXPECT_EQ(get(range1, 4u), 7);
}
