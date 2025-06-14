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
#include "basics/classes/equal_to.hpp"
#include "basics/classes/not_equal_to.hpp"
#include "basics/classes/less.hpp"
#include "basics/classes/greater.hpp"
#include "basics/classes/less_equal.hpp"
#include "basics/classes/greater_equal.hpp"
#include "basics/compatibility/views.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "collections/concepts/settable.hpp"
#include "collections/views/internal/movable_wrapper.hpp"
#include "collections/views/all.hpp"
#include "collections/functions/compare.hpp"


using namespace OpenKalman;
using namespace OpenKalman::collections;

#ifdef __cpp_lib_ranges
#include<ranges>
  namespace rg = std::ranges;
#else
#include "basics/compatibility/views.hpp"
  namespace rg = OpenKalman::ranges;
#endif

TEST(collections, movable_wrapper)
{
  using OpenKalman::collections::internal::movable_wrapper;
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

  // tuple
  static constexpr double x = 5.;
  constexpr movable_wrapper mt {std::tuple {4, x, 6.f}};
  static_assert(mt.get() == std::tuple{4, 5., 6.f});
  static_assert(movable_wrapper {std::tuple{4, x, 6.f}}.get() == std::tuple{4, 5., 6.f});
  static_assert(gettable<0, movable_wrapper<std::tuple<int, double>>>);
  static_assert(gettable<1, movable_wrapper<std::tuple<int, double>>>);
  static_assert(tuple_like<movable_wrapper<std::tuple<int, double>>>);

  using namespace std;
  static_assert(copyable<movable_wrapper<std::tuple<int, double, const float>>>);
  static_assert(movable<movable_wrapper<std::tuple<int&, double, const float>>>);
  static_assert(movable<movable_wrapper<std::tuple<int&, double, const float>&>>);
}


TEST(collections, all_view)
{
  static_assert(collection<std::tuple<int, double>>);
  static_assert(rg::view<views::all_t<std::tuple<int, double>>>);
  static_assert(sized_random_access_range<views::all_t<std::tuple<int, double>>>);
  static_assert(viewable_collection<std::tuple<double, int>>);
  static_assert(viewable_collection<views::all_t<std::tuple<double, int>>>);
  static_assert(collection_view<views::all_t<std::tuple<int, double>>>);
  static_assert(not collection_view<std::tuple<int, double>>);

  static_assert(std::is_move_constructible_v<to_tuple<rg::reverse_view<from_tuple<std::tuple<int, double, const float>>>>>);
  static_assert(std::is_move_constructible_v<to_tuple<rg::reverse_view<from_tuple<std::tuple<int&, double, const float>>>>>);
  static_assert(std::is_move_constructible_v<to_tuple<rg::reverse_view<from_tuple<std::tuple<int&, double, const float>&>>>>);
  static_assert(std::is_move_constructible_v<to_tuple<rg::reverse_view<from_tuple<std::tuple<int&, double, const float>>>&>>);

  static_assert(gettable<0, views::all_t<std::tuple<int, double>>>);
  static_assert(gettable<1, views::all_t<std::tuple<int, double>>>);
  static_assert(settable<0, views::all_t<std::tuple<int, double>>, int>);
  static_assert(settable<1, views::all_t<std::tuple<int, double>>, double>);
  static_assert(rg::range<views::all_t<std::tuple<int, double>>>);
  static_assert(tuple_like<views::all_t<std::tuple<int, double>>>);
  static_assert(std::tuple_size_v<views::all_t<std::tuple<>>> == 0);
  static_assert(std::tuple_size_v<rg::views::all_t<views::all_t<std::tuple<>>>> == 0);
  static_assert(std::tuple_size_v<views::all_t<std::tuple<int, double>>> == 2);
  static_assert(std::tuple_size_v<rg::views::all_t<views::all_t<std::tuple<int, double>>>> == 2);
  static_assert(std::is_same_v<std::tuple_element_t<0, views::all_t<std::tuple<int&, double>>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<0, views::all_t<std::tuple<int&, double>&>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<0, views::all_t<std::tuple<int&, double>&&>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<1, views::all_t<std::tuple<int&, double>>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, views::all_t<std::tuple<int&, double>&>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, views::all_t<std::tuple<int&, double>&&>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<0, views::all_t<rg::reverse_view<views::all_t<std::tuple<int&, double, const float>>>>>, const float>);
  static_assert(std::is_same_v<std::tuple_element_t<1, views::all_t<rg::reverse_view<views::all_t<std::tuple<int&, double, const float>>>>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<2, views::all_t<rg::reverse_view<views::all_t<std::tuple<int&, double, const float>>>>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<0, views::all_t<rg::reverse_view<views::all_t<std::tuple<int&, double, const float>&>>>>, const float>);
  static_assert(std::is_same_v<std::tuple_element_t<1, views::all_t<rg::reverse_view<views::all_t<std::tuple<int&, double, const float>&>>>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<2, views::all_t<rg::reverse_view<views::all_t<std::tuple<int&, double, const float>&>>>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<0, views::all_t<rg::reverse_view<views::all_t<std::tuple<int&, double, const float>>>&>>, const float>);
  static_assert(std::is_same_v<std::tuple_element_t<1, views::all_t<rg::reverse_view<views::all_t<std::tuple<int&, double, const float>>>&>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<2, views::all_t<rg::reverse_view<views::all_t<std::tuple<int&, double, const float>>>&>>, int&>);

  constexpr auto t0 = std::tuple{4, 5.};
  static_assert(rg::size(rg::views::reverse(views::all(t0))) == 2);
  static_assert(size_of_v<decltype(rg::views::reverse(views::all(t0)))> == 2);
  static_assert(std::tuple_size_v<views::all_t<decltype(rg::views::reverse(views::all(t0)))>> == 2);
  static_assert(get(views::all(t0), std::integral_constant<std::size_t, 0>{}) == 4);
  static_assert(get(views::all(t0), std::integral_constant<std::size_t, 1>{}) == 5.);
  static_assert(get(views::all(std::tuple{4, 5.}), std::integral_constant<std::size_t, 0>{}) == 4);
  static_assert(get(views::all(std::tuple{4, 5.}), std::integral_constant<std::size_t, 1>{}) == 5.);

  static_assert(sized_random_access_range<views::all_t<std::vector<int>>>);
  static_assert(sized_random_access_range<views::all_t<std::vector<int>&>>);
  static_assert(sized_random_access_range<views::all_t<std::array<int, 7>>>);
  static_assert(sized_random_access_range<views::all_t<const std::array<int, 7>&>>);

  static_assert(rg::view<views::all_t<std::array<int, 7>>>);
  static_assert(rg::view<views::all_t<std::vector<int>>>);
  static_assert(rg::view<views::all_t<std::vector<int>&>>);
  static_assert(rg::view<views::all_t<const std::vector<int>&>>);
  static_assert(rg::range<views::all_t<std::vector<int>&>>);
  static_assert(rg::range<views::all_t<std::vector<int>>>);

  auto v1 = std::vector{4, 5, 6, 7, 8};
  static_assert(rg::random_access_range<decltype(views::all(v1))>);
  static_assert(gettable<0, views::all_t<decltype(v1)>>);
  static_assert(gettable<1, views::all_t<decltype(v1)>>);
  EXPECT_EQ((views::all(v1)[0u]), 4);
  EXPECT_EQ((views::all(v1)[4u]), 8);
  EXPECT_EQ((views::all(std::vector{4, 5, 6, 7, 8})[3u]), 7);
  EXPECT_EQ((views::all(v1)[std::integral_constant<std::size_t, 4>{}]), 8);
  EXPECT_EQ(*rg::begin(views::all(v1)), 4);
  EXPECT_EQ(*--rg::end(views::all(v1)), 8);
  EXPECT_EQ(rg::size(views::all(v1)), 5);

  EXPECT_EQ(views::all(v1).front(), 4);
  EXPECT_EQ(views::all(std::vector{4, 5, 6, 7, 8}).front(), 4);
  EXPECT_EQ(views::all(v1).back(), 8);
  EXPECT_EQ(views::all(std::vector{4, 5, 6, 7, 8}).back(), 8);
  EXPECT_EQ(*rg::cbegin(views::all(v1)), 4);
  EXPECT_EQ(*--rg::cend(views::all(v1)), 8);
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

  auto v1b = rg::begin(v1);
  EXPECT_EQ(*v1b, 4);
  EXPECT_EQ(*++v1b, 5);
  EXPECT_EQ(v1b[3], 8);
  EXPECT_EQ(*--v1b, 4);
  EXPECT_EQ(*--rg::end(v1), 8);

  constexpr int a1[5] = {1, 2, 3, 4, 5};
  static_assert(std::is_same_v<views::all_t<int(&)[5]>, to_tuple<int (&)[5]>>);
  static_assert(std::is_same_v<views::all_t<const int(&)[5]>, to_tuple<const int(&)[5]>>);
  static_assert(std::is_same_v<views::all_t<decltype((a1))>, to_tuple<const int(&)[5]>>);
  static_assert(rg::range<views::all_t<int(&)[5]>>);
  static_assert(std::tuple_size_v<views::all_t<int(&)[5]>> == 5);
  static_assert(std::tuple_size_v<rg::views::all_t<views::all_t<int(&)[5]>>> == 5);
  static_assert(std::tuple_size_v<views::all_t<const int(&)[5]>> == 5);
  static_assert(std::tuple_size_v<rg::views::all_t<views::all_t<const int(&)[5]>>> == 5);
  static_assert(std::tuple_size_v<views::all_t<decltype(rg::views::reverse(views::all(a1)))>> == 5);
  static_assert(std::is_same_v<std::tuple_element_t<1, views::all_t<int(&)[5]>>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<1, rg::views::all_t<views::all_t<int(&)[5]>>>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<2, views::all_t<const int(&)[5]>>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<2, rg::views::all_t<views::all_t<const int(&)[5]>>>, int>);
  static_assert(views::all(a1).template get<0>() == 1);
  static_assert(views::all(a1).template get<4>() == 5);
  static_assert(OpenKalman::internal::generalized_std_get<0>(views::all(a1)) == 1);
  static_assert(OpenKalman::internal::generalized_std_get<4>(views::all(a1)) == 5);
  static_assert(gettable<0, views::all_t<decltype((a1))>>);
  static_assert(gettable<1, views::all_t<decltype((a1))>>);
  static_assert(tuple_like<views::all_t<int(&)[5]>>);
  static_assert(tuple_like<views::all_t<const int(&)[5]>>);

  static_assert((views::all(a1)[0u]) == 1);
  static_assert((views::all(a1)[4u]) == 5);
  static_assert(*rg::begin(views::all(a1)) == 1);
  static_assert(*(rg::end(views::all(a1)) - 1) == 5);
  static_assert(*rg::cbegin(views::all(a1)) == 1);
  static_assert(*(rg::cend(views::all(a1)) - 1) == 5);
  static_assert(rg::size(views::all(a1)) == 5);
  static_assert(rg::begin(views::all(a1))[2] == 3);
  static_assert(views::all(a1).front() == 1);
  static_assert(views::all(a1).back() == 5);
  static_assert(not views::all(a1).empty());

  constexpr auto t1 = std::tuple{1, 2, 3, 4, 5};
  static_assert(*rg::begin(views::all(t1)) == 1);
  static_assert(views::all(t1)[1_uz] == 2);
  static_assert(views::all(std::tuple{1, 2, 3, 4, 5})[2_uz] == 3);
  static_assert(*++rg::begin(views::all(t1)) == 2);
  static_assert(views::all(t1).front() == 1);
  static_assert(views::all(std::tuple{1, 2, 3, 4, 5}).front() == 1);
  static_assert(views::all(t1).back() == 5);
  static_assert(views::all(std::tuple{1, 2, 3, 4, 5}).back() == 5);

  auto at1 = views::all(t1);
  auto it1 = rg::begin(at1);
  EXPECT_EQ(it1[2], 3);
  EXPECT_EQ(rg::begin(at1)[2], 3);
  EXPECT_EQ(at1.front(), 1);
  EXPECT_EQ(views::all(t1).front(), 1);
  EXPECT_EQ(at1.back(), 5);
  EXPECT_EQ(views::all(t1).back(), 5);

  EXPECT_EQ((v1 | views::all | rg::views::reverse | rg::views::transform([](auto a){return a*a;}))[1u], 49);
  EXPECT_EQ((t1 | views::all | rg::views::transform(std::negate{}) | rg::views::reverse)[1u], -4);
#if __cplusplus >= 202002L
  EXPECT_EQ((t1 | views::all | rg::views::transform([](auto a){return a*a + 1;}) | rg::views::reverse)[1u], 17);
  static_assert(size_of_v<decltype(t1 | views::all | rg::views::transform([](auto a){return a*a + 1;}) | rg::views::reverse)> == 5);
#endif
  static_assert(size_of_v<decltype(t1 | views::all)> == 5);
  static_assert(size_of_v<decltype(t1 | views::all | rg::views::reverse)> == 5);
  static_assert(size_of_v<decltype(t1 | views::all | rg::views::reverse | rg::views::transform(std::negate{}))> == 5);
  static_assert(size_of_v<decltype(t1 | views::all | rg::views::transform(std::negate{}))> == 5);
  static_assert(size_of_v<decltype(t1 | views::all | rg::views::transform(std::negate{}) | rg::views::reverse)> == 5);

  static_assert(std::tuple_size_v<decltype(t1 | views::all | rg::views::transform(std::negate{}) | views::all)> == 5);
  static_assert(std::is_same_v<std::tuple_element_t<1, decltype(t1 | views::all | rg::views::transform(std::negate{}) | views::all)>, int>);
  static_assert(OpenKalman::internal::generalized_std_get<1>(t1 | views::all | rg::views::transform(std::negate{}) | views::all) == -2);
  static_assert(tuple_like<decltype(t1 | views::all | rg::views::transform(std::negate{}) | views::all)>);

  static_assert(size_of_v<decltype(rg::views::empty<int> | views::all)> == 0);
  static_assert(size_of_v<decltype(rg::views::single(5.) | views::all)> == 1);
  static_assert(size_of_v<decltype(rg::views::iota(0, 4) | views::all)> == dynamic_size);

  static_assert(equal_to{}(std::tuple{4, 5.} | views::all, std::tuple{4, 5.} | views::all));
  EXPECT_TRUE(equal_to{}(views::all(std::vector{4, 5, 6}), std::vector{4, 5, 6}));
  EXPECT_TRUE(equal_to{}(std::array{4, 5, 6}, views::all(std::array{4, 5, 6})));
  EXPECT_TRUE(equal_to{}(views::all(v1), v1));
  EXPECT_TRUE(equal_to{}(v1, views::all(v1)));
  EXPECT_TRUE(equal_to{}(views::all(std::vector{4, 5, 6}), views::all(std::tuple{4, 5, 6} | views::all)));
  EXPECT_TRUE(equal_to{}(views::all(std::tuple{4, 5, 6} | views::all), views::all(std::vector{4, 5, 6})));
  EXPECT_TRUE(equal_to{}(views::all(std::tuple{4, 5, 6} | views::all), std::vector{4, 5, 6}));
  EXPECT_TRUE(equal_to{}(std::vector{4, 5, 6}, views::all(std::tuple{4, 5, 6})));

  static_assert(less{}(views::all(std::tuple{4, 5.}), std::tuple{4, 6.} | views::all));
  static_assert(less{}(std::tuple{4, 5.} | views::all, views::all(std::tuple{4, 5., 1})));
  static_assert(less{}(views::all(std::tuple{4, 5.}), views::all(std::tuple{4, 6.})));
  EXPECT_TRUE(less{}(views::all(std::vector{4, 5, 6}), std::vector{4, 5, 7}));
  EXPECT_TRUE(less{}(std::array{4, 5, 6}, views::all(std::array{4, 5, 6, 1})));
  EXPECT_TRUE(less{}(views::all(std::vector{4, 5, 6}), views::all(std::tuple{4, 5, 7})));
  EXPECT_TRUE(less{}(views::all(std::tuple{4, 5, 6}), views::all(std::vector{4, 5, 6, 1})));
  EXPECT_TRUE(less{}(views::all(std::tuple{4, 5, 6}), std::vector{4, 5, 7}));
  EXPECT_TRUE(less{}(std::vector{4, 5, 6}, views::all(std::tuple{4, 5, 7})));

  static_assert(greater{}(views::all(std::tuple{4, 6.}), std::tuple{4, 5.} | views::all));
  static_assert(less_equal{}(views::all(std::tuple{4, 5.}), std::tuple{4, 6.} | views::all));
  EXPECT_TRUE(less_equal{}(views::all(std::vector{4, 5.}), std::vector{4, 5.}));
  static_assert(greater_equal{}(views::all(std::tuple{4, 6.}), std::tuple{4, 5.} | views::all));
  EXPECT_TRUE(greater_equal{}(views::all(std::tuple{4, 5.}), std::vector{4, 5.}));
  static_assert(not_equal_to<>{}(views::all(std::tuple{4, 5.}), std::tuple{4, 6.} | views::all));
}

