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
 * \brief Tests for \ref collections::all_view and \ref collections::views::all involving \ref collections::views::from_tuple_like
 */

#include "tests.hpp"
#include "basics/basics.hpp"
#include "collections/concepts/sized.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "collections/concepts/settable.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

#include "collections/views/internal/tuple_wrapper.hpp"

TEST(collections, tuple_wrapper)
{
  using collections::internal::tuple_wrapper;

  static_assert(viewable_tuple_like<std::tuple<int, double>>);
  static_assert(viewable_tuple_like<std::tuple<int&, double>>);
  static_assert(not viewable_tuple_like<std::tuple<int&&, double>>);
  static_assert(viewable_tuple_like<std::tuple<std::tuple<int>&, double>>);
  static_assert(not viewable_tuple_like<std::tuple<std::tuple<int&>, double>>);

  static constexpr auto mrt = std::tuple {4, 5., 6.f};
  constexpr tuple_wrapper mr {mrt};
  static_assert(mr.get<0>() == 4);
  static_assert(tuple_wrapper {mrt}.get<1>() == 5.);
  static_assert(stdex::same_as<std::decay_t<decltype(mr.get<2>())>, float>);

  constexpr tuple_wrapper ms {std::tuple {4, 5., 6.f}};
  static_assert(ms.get<0>() == 4);
  static_assert(tuple_wrapper {std::tuple{4, 5., 6.f}}.get<1>() == 5.);
  static_assert(stdex::same_as<std::decay_t<decltype(ms.get<2>())>, float>);

  static constexpr double x = 5.;
  constexpr tuple_wrapper mt {std::tuple {4, x, 6.f}};
  static_assert(mt.get<0>() == 4);
  static_assert(tuple_wrapper {std::tuple{4, x, 6.f}}.get<1>() == 5.);
  static_assert(stdex::same_as<std::decay_t<decltype(mt.get<2>())>, float>);

  static_assert(gettable<0, tuple_wrapper<std::tuple<int, double>>>);
  static_assert(gettable<1, tuple_wrapper<std::tuple<int, double>>>);
  static_assert(viewable_collection<tuple_wrapper<std::tuple<int, double>>>);
  static_assert(viewable_collection<tuple_wrapper<std::tuple<int&, double>>>);
  static_assert(viewable_collection<tuple_wrapper<std::tuple<std::tuple<int>&, double>>>);

  static_assert(stdex::copyable<tuple_wrapper<std::tuple<int, double, const float>>>);
  static_assert(stdex::copyable<tuple_wrapper<std::tuple<int, double, const float>&>>);
  static_assert(stdex::copyable<tuple_wrapper<std::tuple<int&, double, const float>>>);
  static_assert(stdex::copyable<tuple_wrapper<std::tuple<int&, double, const float>&>>);
  static_assert(stdex::copyable<tuple_wrapper<std::tuple<int, std::tuple<double>, const float>>>);
  static_assert(stdex::copyable<tuple_wrapper<std::tuple<int, std::tuple<double>&, const float>>>);
}


#include "collections/views/all.hpp"

TEST(collections, all_view_tuple_like)
{
  static_assert(uniformly_gettable<std::tuple<>>);
  static_assert(size_of_v<std::tuple<>> == 0);
  static_assert(collection<std::tuple<int, double>>);
  static_assert(collection<std::tuple<>>);

  static_assert(stdex::ranges::view<views::all_t<std::tuple<>>>);
  static_assert(stdex::ranges::view<views::all_t<std::tuple<int, double>>>);

  static_assert(sized<views::all_t<std::tuple<int, double>>>);
  static_assert(sized<views::all_t<std::tuple<>>>);
  static_assert(viewable_collection<std::tuple<double, int>>);
  static_assert(viewable_collection<std::tuple<>>);
  static_assert(viewable_collection<views::all_t<std::tuple<double, int>>>);
  static_assert(viewable_collection<views::all_t<std::tuple<>>>);
  static_assert(collection_view<views::all_t<std::tuple<int, double>>>);
  static_assert(collection_view<views::all_t<std::tuple<>>>);
  static_assert(collection_view<views::all_t<std::tuple<int&>>>);
  static_assert(collection_view<views::all_t<std::tuple<int&>&>>);
  static_assert(not collection_view<std::tuple<int, double>>);
  static_assert(not collection_view<std::tuple<>>);

  static_assert(std::is_move_constructible_v<from_range<stdex::ranges::reverse_view<from_tuple_like<std::tuple<int, double, const float>>>>>);
  static_assert(std::is_move_constructible_v<from_range<stdex::ranges::reverse_view<from_tuple_like<std::tuple<int&, double, const float>>>>>);
  static_assert(std::is_move_constructible_v<from_range<stdex::ranges::reverse_view<from_tuple_like<std::tuple<int&, double, const float>&>>>>);
  static_assert(std::is_move_constructible_v<from_range<stdex::ranges::reverse_view<from_tuple_like<std::tuple<int&, double, const float>>>&>>);
  static_assert(gettable<0, views::all_t<std::tuple<int, double>>>);
  static_assert(gettable<1, views::all_t<std::tuple<int, double>>>);
  static_assert(settable<0, views::all_t<std::tuple<int, double>>, int>);
  static_assert(settable<1, views::all_t<std::tuple<int, double>>, double>);
  static_assert(stdex::ranges::range<views::all_t<std::tuple<int, double>>>);
  static_assert(uniformly_gettable<views::all_t<std::tuple<int, double>>>);
  static_assert(std::tuple_size_v<views::all_t<std::tuple<>>> == 0);
  static_assert(std::tuple_size_v<stdex::ranges::views::all_t<views::all_t<std::tuple<>>>> == 0);
  static_assert(std::tuple_size_v<views::all_t<std::tuple<int, double>>> == 2);
  static_assert(std::tuple_size_v<stdex::ranges::views::all_t<views::all_t<std::tuple<int, double>>>> == 2);
  static_assert(stdex::same_as<std::tuple_element_t<0, views::all_t<std::tuple<int&, double>>>, int&>);
  static_assert(stdex::same_as<std::tuple_element_t<0, views::all_t<std::tuple<int&, double>&>>, int&>);
  static_assert(stdex::same_as<std::tuple_element_t<0, views::all_t<std::tuple<int&, double>&&>>, int&>);
  static_assert(stdex::same_as<std::tuple_element_t<1, views::all_t<std::tuple<int&, double>>>, double>);
  static_assert(stdex::same_as<std::tuple_element_t<1, views::all_t<std::tuple<int&, double>&>>, double>);
  static_assert(stdex::same_as<std::tuple_element_t<1, views::all_t<std::tuple<int&, double>&&>>, double>);
  static_assert(stdex::same_as<std::tuple_element_t<0, views::all_t<stdex::ranges::reverse_view<views::all_t<std::tuple<int&, double, const float>>>>>, const float>);
  static_assert(stdex::same_as<std::tuple_element_t<1, views::all_t<stdex::ranges::reverse_view<views::all_t<std::tuple<int&, double, const float>>>>>, double>);
  static_assert(stdex::same_as<std::tuple_element_t<2, views::all_t<stdex::ranges::reverse_view<views::all_t<std::tuple<int&, double, const float>>>>>, int&>);
  static_assert(stdex::same_as<std::tuple_element_t<0, views::all_t<stdex::ranges::reverse_view<views::all_t<std::tuple<int&, double, const float>&>>>>, const float>);
  static_assert(stdex::same_as<std::tuple_element_t<1, views::all_t<stdex::ranges::reverse_view<views::all_t<std::tuple<int&, double, const float>&>>>>, double>);
  static_assert(stdex::same_as<std::tuple_element_t<2, views::all_t<stdex::ranges::reverse_view<views::all_t<std::tuple<int&, double, const float>&>>>>, int&>);
  static_assert(stdex::same_as<std::tuple_element_t<0, views::all_t<stdex::ranges::reverse_view<views::all_t<std::tuple<int&, double, const float>>>&>>, const float>);
  static_assert(stdex::same_as<std::tuple_element_t<1, views::all_t<stdex::ranges::reverse_view<views::all_t<std::tuple<int&, double, const float>>>&>>, double>);
  static_assert(stdex::same_as<std::tuple_element_t<2, views::all_t<stdex::ranges::reverse_view<views::all_t<std::tuple<int&, double, const float>>>&>>, int&>);

  constexpr auto t0 = std::tuple{4, 5.};
  static_assert(stdex::ranges::size(stdex::ranges::views::reverse(views::all(t0))) == 2);
  static_assert(size_of_v<decltype(stdex::ranges::views::reverse(views::all(t0)))> == 2);
  static_assert(std::tuple_size_v<views::all_t<decltype(stdex::ranges::views::reverse(views::all(t0)))>> == 2);
  static_assert(get<0>(views::all(t0)) == 4);
  static_assert(get<1>(views::all(t0)) == 5.);
  static_assert(get<0>(views::all(std::tuple{4, 5.})) == 4);
  static_assert(get<1>(views::all(std::tuple{4, 5.})) == 5.);

  constexpr auto t1 = std::tuple{1, 2, 3, 4, 5};
  static_assert(*stdex::ranges::begin(views::all(t1)) == 1);
  static_assert(views::all(t1)[1_uz] == 2);
  static_assert(views::all(std::tuple{1, 2, 3, 4, 5})[2_uz] == 3);
  static_assert(*++stdex::ranges::begin(views::all(t1)) == 2);
  static_assert(views::all(t1).front() == 1);
  static_assert(views::all(std::tuple{1, 2, 3, 4, 5}).front() == 1);
  static_assert(views::all(t1).back() == 5);
  static_assert(views::all(std::tuple{1, 2, 3, 4, 5}).back() == 5);

  auto at1 = views::all(t1);
  auto it1 = stdex::ranges::begin(at1);
  EXPECT_EQ(it1[2], 3);
  EXPECT_EQ(stdex::ranges::begin(at1)[2], 3);
  EXPECT_EQ(at1.front(), 1);
  EXPECT_EQ(views::all(t1).front(), 1);
  EXPECT_EQ(at1.back(), 5);
  EXPECT_EQ(views::all(t1).back(), 5);

  EXPECT_EQ((t1 | views::all | stdex::ranges::views::reverse | stdex::ranges::views::transform([](auto a){return a*a;}))[1u], 16);
  EXPECT_EQ((t1 | views::all | stdex::ranges::views::transform(std::negate{}) | stdex::ranges::views::reverse)[1u], -4);
#if __cplusplus >= 202002L
  EXPECT_EQ((t1 | views::all | stdex::ranges::views::transform([](auto a){return a*a + 1;}) | stdex::ranges::views::reverse)[1u], 17);
  static_assert(size_of_v<decltype(t1 | views::all | stdex::ranges::views::transform([](auto a){return a*a + 1;}) | stdex::ranges::views::reverse)> == 5);
#endif
  static_assert(size_of_v<decltype(t1 | views::all)> == 5);
  static_assert(size_of_v<decltype(t1 | views::all | stdex::ranges::views::reverse)> == 5);
  static_assert(size_of_v<decltype(t1 | views::all | stdex::ranges::views::reverse | stdex::ranges::views::transform(std::negate{}))> == 5);
  static_assert(size_of_v<decltype(t1 | views::all | stdex::ranges::views::transform(std::negate{}))> == 5);
  static_assert(size_of_v<decltype(t1 | views::all | stdex::ranges::views::transform(std::negate{}) | stdex::ranges::views::reverse)> == 5);

  static_assert(std::tuple_size_v<decltype(t1 | views::all | stdex::ranges::views::transform(std::negate{}) | views::all)> == 5);
  static_assert(stdex::same_as<std::tuple_element_t<1, decltype(t1 | views::all | stdex::ranges::views::transform(std::negate{}) | views::all)>, int>);
  static_assert(get<1>(t1 | views::all | stdex::ranges::views::transform(std::negate{}) | views::all) == -2);
  static_assert(uniformly_gettable<decltype(t1 | views::all | stdex::ranges::views::transform(std::negate{}) | views::all)>);

  static_assert(size_of_v<decltype(stdex::ranges::views::empty<int> | views::all)> == 0);
  static_assert(size_of_v<decltype(stdex::ranges::views::single(5.) | views::all)> == 1);
  static_assert(size_of_v<decltype(stdex::ranges::views::iota(0, 4) | views::all)> == stdex::dynamic_extent);
}
