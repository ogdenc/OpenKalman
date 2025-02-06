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
 * \brief Tests for scalar types and constexpr math functions.
 */

#include <linear-algebra/values/functions/internal/get_collection_element.hpp>

#include "linear-algebra/values/tests/tests.hpp"
#include "linear-algebra/values/concepts/fixed.hpp"
#include "linear-algebra/values/concepts/dynamic.hpp"
#include "linear-algebra/values/traits/number_type_of_t.hpp"
#include "linear-algebra/values/traits/real_type_of_t.hpp"
#include "linear-algebra/values/concepts/integral.hpp"
#include "linear-algebra/values/concepts/index.hpp"

using namespace OpenKalman;

TEST(values, integral)
{
  static_assert(value::index<std::integral_constant<int, 2>>);
  static_assert(value::fixed<std::integral_constant<int, 2>>);
  static_assert(not value::dynamic<std::integral_constant<int, 2>>);
  static_assert(std::is_same_v<value::number_type_of_t<std::integral_constant<int, 2>>, int>);
  static_assert(std::is_same_v<value::number_type_of_t<std::integral_constant<std::size_t, 2>>, std::size_t>);
  static_assert(std::is_same_v<value::real_type_of_t<std::integral_constant<std::size_t, 2>>, std::integral_constant<std::size_t, 2>>);

  static_assert(value::index<std::size_t>);
  static_assert(not value::fixed<std::size_t>);
  static_assert(value::dynamic<std::size_t>);
  static_assert(std::is_same_v<value::number_type_of_t<std::size_t>, std::size_t>);
  static_assert(std::is_same_v<value::real_type_of_t<int>, double>);
  static_assert(std::is_same_v<value::real_type_of_t<std::size_t>, double>);

  static_assert(not value::index<int>);
  static_assert(value::integral<int>);
  static_assert(not value::fixed<int>);
  static_assert(value::dynamic<int>);
}


#include "linear-algebra/values/concepts/index_tuple.hpp"
#include "linear-algebra/values/concepts/index_collection.hpp"

TEST(values, index_collection)
{
  using Tup1 = std::tuple<std::size_t, unsigned, unsigned long>;
  static_assert(value::index_tuple<Tup1>);
  static_assert(value::index_collection<Tup1>);
  static_assert(not value::index_tuple<std::tuple<std::size_t, unsigned, int>>);

  using Range1 = std::vector<std::size_t>;
  static_assert(value::index_collection<Range1>);
  static_assert(not value::index_tuple<Range1>);
  using Range2 = std::array<std::size_t, 5>;
  static_assert(value::index_collection<Range2>);
  static_assert(value::index_tuple<Range2>);
}


#include "basics/internal/iota_tuple.hpp"
#include "basics/internal/iota_range.hpp"
#include "linear-algebra/values/functions/internal/transform_collection.hpp"

TEST(values, transform_collection)
{
  using value::internal::transform_collection;
  constexpr auto t_identity = transform_collection(internal::iota_tuple<0, 5>(), [](auto i){ return i; });
  static_assert(std::tuple_size_v<decltype(t_identity)> == 5);
  using value::internal::get;
  static_assert(get<3>(t_identity) == 3);

  auto r_identity = transform_collection(internal::iota_range(0, 5), [](auto i){ return i; });
  EXPECT_EQ(r_identity.size(), 5_uz);
  std::size_t j = 0;
  for (auto i : r_identity) EXPECT_EQ(i, j++);

  auto r_reverse = transform_collection(internal::iota_range(0, 9), [](auto i){ return 10_uz - i; });
  EXPECT_EQ(r_reverse.size(), 9_uz);
  j = 10;
  for (auto i : r_reverse) EXPECT_EQ(i, j--);

  auto ita = r_reverse.begin();
  EXPECT_EQ(*ita, 10);
  EXPECT_EQ(*(ita + 1), 9);
  EXPECT_EQ(ita[1], 9);
  EXPECT_EQ(*(2 + ita), 8);
  EXPECT_EQ(ita[3], 7);
  ++ita;
  EXPECT_EQ(*ita, 9);
  EXPECT_EQ(*(ita - 1), 10);
  EXPECT_EQ(ita[-1], 10);
  EXPECT_EQ(ita[1], 8);
  EXPECT_EQ(ita++[2], 7);
  EXPECT_EQ(*ita, 8);
  EXPECT_EQ(*(ita - 2), 10);
  EXPECT_EQ(ita[1], 7);
  EXPECT_EQ(ita--[2], 6);
  EXPECT_EQ(*ita, 9);
  --ita;
  EXPECT_EQ(*ita, 10);
}


#include "linear-algebra/values/functions/internal/get_collection_element.hpp"
TEST(values, get_collection_element)
{
  using value::internal::get_collection_element;
  auto iota_range1 = internal::iota_range(3, 8);
  EXPECT_EQ(get_collection_element(iota_range1, std::integral_constant<std::size_t, 0>{}), 3);
  EXPECT_EQ(get_collection_element(iota_range1, 3u), 6);
  EXPECT_EQ(get_collection_element(iota_range1, 4u), 7);

  auto iota_tup1 = internal::iota_tuple<3, 8>();
  static_assert(get_collection_element(iota_tup1, std::integral_constant<std::size_t, 0>{}) == 3);
  static_assert(get_collection_element(iota_tup1, std::integral_constant<std::size_t, 3>{}) == 6);

  static_assert(get_collection_element<std::size_t>(iota_tup1, 3u) == 6);
  static_assert(get_collection_element<std::size_t>(iota_tup1, 4u) == 7);
}