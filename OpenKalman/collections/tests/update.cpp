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
 * \brief Tests for \ref collections::update_view and \ref collections::views::update.
 */

#include "values/tests/tests.hpp"
#include "values/concepts/fixed.hpp"
#include "values/concepts/dynamic.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/concepts/integral.hpp"
#include "values/concepts/index.hpp"
#include "collections/functions/get.hpp"
#include "collections/views/iota.hpp"
#include "collections/views/update.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

#ifdef __cpp_lib_ranges
namespace rg = std::ranges;
#else
namespace rg = OpenKalman::ranges;
#endif

namespace
{
  //auto c0 = std::integral_constant<std::size_t, 0>{};
  //auto c5 = std::integral_constant<std::size_t, 5>{};
}

TEST(collections, update_view)
{
  auto g1 = [](const auto& v, auto i)
  {
    return get(v, i) + 10;
  };

  auto s1 = [](auto& v, auto i, auto x) -> auto&
  {
    get(v, i) = get(v, i) + x;
    return v;
  };

  int a1[5] = {1, 2, 3, 4, 5};
  static_assert(std::is_invocable_r_v<int, decltype((g1)), views::all_t<int(&)[5]>, std::size_t>);
  static_assert(std::is_invocable_v<decltype((s1)), views::all_t<int(&)[5]>&, std::size_t, int>);
  EXPECT_EQ(g1(views::all(a1), 0_uz), 11);
  EXPECT_EQ(g1(views::all(a1), 3_uz), 14);

  auto a1up = update_view {a1 | views::all, g1, s1};
  static_assert(std::tuple_size_v<decltype(a1up)> == 5);
  static_assert(std::is_same_v<std::tuple_element_t<2, decltype(a1up)>, int>);
  EXPECT_EQ(a1up[0], 11);
  EXPECT_EQ(a1up[4], 15);
  EXPECT_EQ(a1up.get<0>(), 11);
  EXPECT_EQ(a1up.get<2>(), 13);
  a1up[4] = 2;
  EXPECT_EQ(a1up[4], 17);
  a1up.get<4>() = -3;
  EXPECT_EQ(a1up[4], 14);

  auto a1it = a1up.begin();
  EXPECT_EQ(*a1it, 11);
  *a1it = 4;
  EXPECT_EQ(*a1it++, 15);
  EXPECT_EQ(a1it[1], 13);

  auto v1 = std::vector{4, 5, 6, 7, 8};
  static_assert(std::is_invocable_r_v<double, decltype((g1)), views::all_t<decltype(v1)>, std::size_t>);
  static_assert(std::is_invocable_v<decltype((s1)), views::all_t<decltype(v1)>&, std::size_t, double>);
  EXPECT_EQ(g1(views::all(v1), 0_uz), 11.);
  EXPECT_EQ(g1(views::all(v1), 3_uz), 14.);

  auto v1up = update_view {v1 | views::all, g1, s1};
  static_assert(size_of_v<decltype(v1up)> == dynamic_size);
  static_assert(std::is_same_v<std::tuple_element_t<2, decltype(v1up)>, int>);
  EXPECT_EQ(v1up[0], 11);
  EXPECT_EQ(v1up[4], 15);
  EXPECT_EQ(v1up.get<0>(), 11);
  EXPECT_EQ(v1up.get<2>(), 13);
  v1up[4] = 2;
  EXPECT_EQ(v1up[4], 17);
  v1up.get<4>() = -3;
  EXPECT_EQ(v1up[4], 14);

  auto v1it = v1up.begin();
  EXPECT_EQ(*v1it, 11);
  *v1it = 4;
  EXPECT_EQ(*v1it++, 15);
  EXPECT_EQ(v1it[1], 13);

  auto t1 = std::tuple{1, 2., 3.f, 4., 5};
  static_assert(std::is_invocable_r_v<double, decltype((g1)), views::all_t<decltype(t1)>, std::size_t>);
  static_assert(std::is_invocable_r_v<double, decltype((g1)), views::all_t<decltype(t1)>, std::integral_constant<std::size_t, 4>>);
  static_assert(std::is_invocable_v<decltype((s1)), views::all_t<decltype(t1)>&, std::integral_constant<std::size_t, 0>, double>);
  EXPECT_EQ(g1(views::all(t1), 0_uz), 11.);
  EXPECT_EQ(g1(views::all(t1), 3_uz), 14.);

  auto t1up = update_view {t1 | views::all, g1, s1};
  static_assert(std::tuple_size_v<decltype(t1up)> == 5);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(t1up)>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<1, decltype(t1up)>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<2, decltype(t1up)>, float>);
  static_assert(std::is_same_v<std::tuple_element_t<3, decltype(t1up)>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<4, decltype(t1up)>, int>);
  EXPECT_EQ(t1up[0], 11);
  EXPECT_EQ(t1up[4], 15);
  EXPECT_EQ(t1up.get<0>(), 11);
  EXPECT_EQ(t1up.get<2>(), 13);
  t1up[4] = 2;
  EXPECT_EQ(t1up[4], 17);
  t1up.get<4>() = -3;
  EXPECT_EQ(t1up[4], 14);

  auto t1it = t1up.begin();
  EXPECT_EQ(*t1it, 11);
  *t1it = 4;
  EXPECT_EQ(*t1it++, 15);
  EXPECT_EQ(t1it[1], 13);

}
