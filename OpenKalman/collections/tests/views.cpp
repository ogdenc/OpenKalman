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
#include "basics/classes/not_equal_to.hpp"
#include "basics/classes/less.hpp"
#include "basics/classes/greater.hpp"
#include "basics/classes/less_equal.hpp"
#include "basics/classes/greater_equal.hpp"

using namespace OpenKalman;
using namespace OpenKalman::collections;

#include "collections/concepts/viewable_collection.hpp"
#include "collections/views/all.hpp"

#ifdef __cpp_lib_ranges
  namespace rg = std::ranges;
#else
  namespace rg = OpenKalman::ranges;
#endif

TEST(collections, all_view)
{
  static_assert(tuple_like<all_view<std::tuple<int, double>>>);
  static_assert(std::tuple_size_v<all_view<std::tuple<int, double>>> == 2);
  static_assert(std::tuple_size_v<all_view<std::tuple<>>> == 0);
  static_assert(std::is_same_v<std::tuple_element_t<0, all_view<std::tuple<int&, double>>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<0, all_view<std::tuple<int&, double>&>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<0, all_view<std::tuple<int&, double>&&>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<1, all_view<std::tuple<int&, double>>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, all_view<std::tuple<int&, double>&>>, double&>);
  static_assert(std::is_same_v<std::tuple_element_t<1, all_view<std::tuple<int&, double>&&>>, double&&>);

  constexpr auto t0 = std::tuple{4, 5.};
  static_assert(get(all_view {t0}, std::integral_constant<std::size_t, 0>{}) == 4);
  static_assert(get(all_view {t0}, std::integral_constant<std::size_t, 1>{}) == 5.);
  static_assert(get(all_view {std::tuple{4, 5.}}, std::integral_constant<std::size_t, 0>{}) == 4);
  static_assert(get(all_view {std::tuple{4, 5.}}, std::integral_constant<std::size_t, 1>{}) == 5.);

  static_assert(sized_random_access_range<all_view<std::vector<int>>>);
  static_assert(sized_random_access_range<all_view<std::array<int, 7>>>);

  static_assert(rg::view<all_view<std::array<int, 7>>>);
  static_assert(rg::view<all_view<std::vector<int>>>);
  static_assert(rg::view<all_view<std::vector<int>&>>);
  static_assert(rg::view<all_view<const std::vector<int>&>>);
  static_assert(rg::range<all_view<std::vector<int>&>>);
  static_assert(rg::range<all_view<std::vector<int>>>);

  auto v1 = std::vector{4, 5, 6, 7, 8};
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

  all_view<std::vector<int>&> id1_v1{v1};
  EXPECT_EQ((id1_v1[2u]), 6);
  id1_v1 = v1;
  EXPECT_EQ((id1_v1[3u]), 7);

  auto v2 = std::vector{9, 10, 11, 12, 13};
  all_view<std::vector<int>&> id1_v2 {v1};
  id1_v2 = views::all(v2);
  EXPECT_EQ((id1_v2[3u]), 12);
  id1_v2 = views::all(v1);
  EXPECT_EQ((id1_v2[3u]), 7);

  all_view<std::vector<int>> id2_v1 {std::vector{4, 5, 6, 7, 8}};
  EXPECT_EQ((id2_v1[2u]), 6);

  auto v1b = rg::begin(v1);
  EXPECT_EQ(*v1b, 4);
  EXPECT_EQ(*++v1b, 5);
  EXPECT_EQ(v1b[3], 8);
  EXPECT_EQ(*--v1b, 4);
  EXPECT_EQ(*--rg::end(v1), 8);

  constexpr int a1[5] = {1, 2, 3, 4, 5};
  static_assert(rg::range<all_view<int[5]>>);
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

  static_assert(rg::range<all_view<std::tuple<int, double>>>);
  static_assert(rg::view<all_view<std::tuple<int, double>>>);
  static_assert(sized_random_access_range<all_view<std::tuple<int, double>>>);
  static_assert(viewable_collection<all_view<std::tuple<double, int>>>);

  static_assert(equal_to{}(all_view {std::tuple{4, 5.}}, std::tuple{4, 5.}));
  static_assert(equal_to{}(std::tuple{4, 5.}, views::all(std::tuple{4, 5.})));
  static_assert(equal_to{}(views::all(std::tuple{4, 5.}), views::all(std::tuple{4, 5.})));
  EXPECT_TRUE(equal_to{}(views::all(std::vector{4, 5, 6}), std::vector{4, 5, 6}));
  EXPECT_TRUE(equal_to{}(std::array{4, 5, 6}, all_view {std::array{4, 5, 6}}));
  EXPECT_TRUE(equal_to{}(views::all(v1), v1));
  EXPECT_TRUE(equal_to{}(v1, views::all(v1)));
  EXPECT_TRUE(equal_to{}(views::all(std::vector{4, 5, 6}), views::all(std::tuple{4, 5, 6})));
  EXPECT_TRUE(equal_to{}(views::all(std::tuple{4, 5, 6}), views::all(std::vector{4, 5, 6})));
  EXPECT_TRUE(equal_to{}(views::all(std::tuple{4, 5, 6}), std::vector{4, 5, 6}));
  EXPECT_TRUE(equal_to{}(std::vector{4, 5, 6}, views::all(std::tuple{4, 5, 6})));

  static_assert(less{}(all_view {std::tuple{4, 5.}}, std::tuple{4, 6.}));
  static_assert(less{}(std::tuple{4, 5.}, views::all(std::tuple{4, 5., 1})));
  static_assert(less{}(views::all(std::tuple{4, 5.}), views::all(std::tuple{4, 6.})));
  EXPECT_TRUE(less{}(views::all(std::vector{4, 5, 6}), std::vector{4, 5, 7}));
  EXPECT_TRUE(less{}(std::array{4, 5, 6}, all_view {std::array{4, 5, 6, 1}}));
  EXPECT_TRUE(less{}(views::all(std::vector{4, 5, 6}), views::all(std::tuple{4, 5, 7})));
  EXPECT_TRUE(less{}(views::all(std::tuple{4, 5, 6}), views::all(std::vector{4, 5, 6, 1})));
  EXPECT_TRUE(less{}(views::all(std::tuple{4, 5, 6}), std::vector{4, 5, 7}));
  EXPECT_TRUE(less{}(std::vector{4, 5, 6}, views::all(std::tuple{4, 5, 7})));

  static_assert(greater{}(all_view {std::tuple{4, 6.}}, std::tuple{4, 5.}));
  static_assert(less_equal{}(all_view {std::tuple{4, 5.}}, std::tuple{4, 6.}));
  EXPECT_TRUE(less_equal{}(all_view {std::vector{4, 5.}}, std::vector{4, 5.}));
  static_assert(greater_equal{}(all_view {std::tuple{4, 6.}}, std::tuple{4, 5.}));
  EXPECT_TRUE(greater_equal{}(all_view {std::tuple{4, 5.}}, std::vector{4, 5.}));
  static_assert(not_equal_to<>{}(all_view {std::tuple{4, 5.}}, std::tuple{4, 6.}));
}


#include "collections/views/single.hpp"

TEST(collections, single_view)
{
  static_assert(std::tuple_size_v<single_view<int>> == 1);
  static_assert(std::tuple_size_v<single_view<std::tuple<int, double>>> == 1);
  static_assert(std::tuple_size_v<single_view<std::tuple<>>> == 1);
  static_assert(std::is_same_v<std::tuple_element_t<0, single_view<int>>, int>);
  static_assert(std::is_same_v<std::tuple_element_t<0, single_view<int&>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<0, single_view<int&&>>, int&&>);

  static_assert(get(single_view {4}, std::integral_constant<std::size_t, 0>{}) == 4);
  static_assert(get(single_view {std::tuple{}}, std::integral_constant<std::size_t, 0>{}) == std::tuple{});
  static_assert(get(single_view<std::tuple<>&&> {std::tuple{}}, std::integral_constant<std::size_t, 0>{}) == std::tuple{});
  auto t1 = std::tuple{};
  static_assert(get(single_view {t1}, std::integral_constant<std::size_t, 0>{}) == std::tuple{});

  static_assert((views::single(4)[0u]) == 4);

  static constexpr auto s1 = views::single(7);
  constexpr auto is1 = rg::begin(s1);
  static_assert(*is1 == 7);
  static_assert(*--rg::end(s1) == 7);

  EXPECT_TRUE(equal_to{}(single_view {4}, std::vector {4}));
  static_assert(equal_to{}(single_view {4}, 4));
  static_assert(less{}(std::array {4.}, views::single(5.)));
  static_assert(greater{}(6., views::single(5.)));
  static_assert(not_equal_to{}(views::single(5), views::single(6)));
}


#include "collections/views/reverse.hpp"

TEST(collections, reverse_view)
{
  static_assert(std::tuple_size_v<reverse_view<std::tuple<int, double>>> == 2);
  static_assert(std::tuple_size_v<reverse_view<std::tuple<>>> == 0);
  static_assert(std::is_same_v<std::tuple_element_t<0, reverse_view<std::tuple<float, int, double>>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<0, reverse_view<std::tuple<float, int, const double&>>>, const double&>);
  static_assert(std::is_same_v<std::tuple_element_t<0, reverse_view<std::tuple<float, int, double&&>>>, double&&>);
  static_assert(std::is_same_v<std::tuple_element_t<1, reverse_view<std::tuple<float, int, double>>>&, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<1, reverse_view<std::tuple<float, int&&, double>>>&, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<2, reverse_view<std::tuple<float, int, double>>>, float>);
  static_assert(std::is_same_v<std::tuple_element_t<2, reverse_view<std::tuple<float&, int, double>>>&&, float&>);

  static_assert(get(reverse_view {std::tuple{4, 5.}}, std::integral_constant<std::size_t, 0>{}) == 5.);
  static_assert(get(reverse_view {std::tuple{4, 5.}}, std::integral_constant<std::size_t, 1>{}) == 4);
  static_assert(get(reverse_view {std::array{4, 5}}, std::integral_constant<std::size_t, 0>{}) == 5);

  constexpr int a1[5] = {1, 2, 3, 4, 5};
  static_assert(rg::range<reverse_view<int[5]>>);
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


#include "collections/views/replicate.hpp"

TEST(collections, replicate_view)
{
  static_assert(std::tuple_size_v<replicate_view<std::tuple<int, double>, std::integral_constant<std::size_t, 3>>> == 6);
  static_assert(std::tuple_size_v<replicate_view<std::tuple<>, std::integral_constant<std::size_t, 3>>> == 0);
  static_assert(std::is_same_v<std::tuple_element_t<0, replicate_view<std::tuple<double, int&, float&&>, std::integral_constant<std::size_t, 2>>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, replicate_view<std::tuple<double, int&, float&&>, std::integral_constant<std::size_t, 2>>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<2, replicate_view<std::tuple<double, int&, float&&>, std::integral_constant<std::size_t, 2>>>, float&&>);
  static_assert(std::is_same_v<std::tuple_element_t<3, replicate_view<std::tuple<double, int&, float&&>&, std::integral_constant<std::size_t, 2>>>, double&>);
  static_assert(std::is_same_v<std::tuple_element_t<4, replicate_view<std::tuple<double, int&, float&&>&, std::integral_constant<std::size_t, 2>>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<5, replicate_view<std::tuple<double, int&, float&&>&, std::integral_constant<std::size_t, 2>>>, float&>);

  using T1 = replicate_view<std::tuple<int, double>, unsigned>;
#ifdef __cpp_lib_ranges
  using std::iter_reference_t;
  using std::iter_difference_t;
#endif
  static_assert(std::is_same_v<decltype(*std::declval<ranges::iterator_t<T1>>()), iter_reference_t<ranges::iterator_t<T1>>>);
  static_assert(std::is_same_v<decltype(std::declval<ranges::iterator_t<T1>>() + std::declval<iter_difference_t<ranges::iterator_t<T1>>>()), ranges::iterator_t<T1>>);
  static_assert(std::is_same_v<decltype(std::declval<iter_difference_t<ranges::iterator_t<T1>>>() + std::declval<ranges::iterator_t<T1>>()), ranges::iterator_t<T1>>);
  static_assert(std::is_same_v<decltype(std::declval<ranges::iterator_t<T1>>() - std::declval<iter_difference_t<ranges::iterator_t<T1>>>()), ranges::iterator_t<T1>>);

  constexpr auto t1 = std::tuple{4, 5.};
  static_assert(get(replicate_view {t1, std::integral_constant<std::size_t, 2>{}}, std::integral_constant<std::size_t, 0>{}) == 4);
  static_assert(get(replicate_view {t1, std::integral_constant<std::size_t, 2>{}}, std::integral_constant<std::size_t, 1>{}) == 5.);
  static_assert(get(replicate_view {std::tuple{4, 5.}, std::integral_constant<std::size_t, 2>{}}, std::integral_constant<std::size_t, 2>{}) == 4);
  static_assert(get(replicate_view {std::tuple{4, 5.}, std::integral_constant<std::size_t, 2>{}}, std::integral_constant<std::size_t, 3>{}) == 5.);
  static_assert(get(views::replicate(std::tuple{4, 5.}, std::integral_constant<std::size_t, 2>{}), std::integral_constant<std::size_t, 3>{}) == 5.);

  static_assert(sized_random_access_range<replicate_view<std::tuple<int, double>, unsigned>>);
  EXPECT_EQ(get(replicate_view {t1, 2u}, std::integral_constant<std::size_t, 3>{}), 5.);
  EXPECT_EQ(get(replicate_view {std::tuple{4, 5.}, 2u}, std::integral_constant<std::size_t, 3>{}), 5.);

  static_assert(*rg::begin(views::replicate(t1, std::integral_constant<std::size_t, 3>{})) == 4.);
  static_assert(views::replicate(t1, std::integral_constant<std::size_t, 3>{})[3_uz] == 5.);
  static_assert(views::replicate(std::tuple{1, 2, 3, 4, 5}, std::integral_constant<std::size_t, 3>{})[8_uz] == 4.);
  static_assert(*++rg::begin(views::replicate(t1, 3u)) == 5.);
  static_assert(*++ ++rg::begin(views::replicate(t1, 3u)) == 4.);
  static_assert(views::replicate(t1, 3u).front() == 4.);
  static_assert(views::replicate(std::tuple{1, 2, 3, 4, 5}, 3u).front() == 1);
  static_assert(views::replicate(t1, std::integral_constant<std::size_t, 3>{}).back() == 5.);
  static_assert(views::replicate(std::tuple{1, 2, 3, 4, 5}, 3u).back() == 5.);

  auto at1 = views::replicate(t1, 4u);
  auto it1 = rg::begin(at1);
  EXPECT_EQ(it1[2], 4.);
  EXPECT_EQ(rg::begin(at1)[2], 4.);
  EXPECT_EQ(at1.front(), 4.);
  EXPECT_EQ(views::replicate(t1, 3u).front(), 4.);
  EXPECT_EQ(at1.back(), 5.);
  EXPECT_EQ(views::replicate(t1, 3u).back(), 5.);

  auto v1 = std::vector{3, 4, 5};
  static_assert(sized_random_access_range<replicate_view<std::vector<int>, unsigned>>);
  EXPECT_EQ((views::replicate(v1, 3u)[3u]), 3);
  EXPECT_EQ((views::replicate(3u)(v1)[7u]), 4);
  EXPECT_EQ((views::replicate(std::integral_constant<std::size_t, 3>{})(v1)[2u]), 5);
  EXPECT_EQ((views::replicate(v1, 3u)[std::integral_constant<std::size_t, 3>{}]), 3);
  EXPECT_EQ((views::replicate(3u)(v1)[std::integral_constant<std::size_t, 7>{}]), 4);
  EXPECT_EQ((views::replicate(std::integral_constant<std::size_t, 3>{})(v1)[std::integral_constant<std::size_t, 2>{}]), 5);

#if __cpp_lib_ranges >= 202202L
  EXPECT_EQ((v1 | views::replicate(2u))[0u], 3);
  EXPECT_EQ((t1 | views::replicate(std::integral_constant<std::size_t, 3>{}))[1u], 5.);
  EXPECT_EQ((t1 | views::replicate(std::integral_constant<std::size_t, 3>{}))[2u], 4);
  static_assert(get(std::tuple{7., 8.f, 9} | views::replicate(std::integral_constant<std::size_t, 3>{}), std::integral_constant<std::size_t, 4>{}) == 8.f);
#endif

  static_assert(equal_to{}(replicate_view {std::tuple{4, 5., 6.f}, std::integral_constant<std::size_t, 2>{}}, std::tuple{4, 5., 6.f, 4, 5., 6.f}));
  EXPECT_TRUE(equal_to{}(std::vector{4, 5, 6, 4, 5, 6}, replicate_view(std::tuple{4, 5, 6}, std::integral_constant<std::size_t, 2>{})));
}


#include "collections/views/concat.hpp"

TEST(collections, concat_view)
{
  static_assert(std::tuple_size_v<concat_view<std::tuple<int, double>, std::tuple<float, unsigned>>> == 4);
  static_assert(std::tuple_size_v<concat_view<std::tuple<>, std::tuple<>>> == 0);
  static_assert(std::tuple_size_v<concat_view<std::tuple<>, std::tuple<int>>> == 1);
  static_assert(std::tuple_size_v<concat_view<std::tuple<double>, std::tuple<>>> == 1);
  static_assert(std::tuple_size_v<concat_view<std::tuple<double, int>, std::tuple<unsigned, long double, float>>> == 5);
  static_assert(std::is_same_v<std::tuple_element_t<0, concat_view<std::tuple<double, int&>, std::tuple<unsigned&&, long double, float>>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, concat_view<std::tuple<double, int&>, std::tuple<unsigned&&, long double, float>>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<2, concat_view<std::tuple<double, int&>, std::tuple<unsigned&&, long double, float>>>, unsigned&&>);
  static_assert(std::is_same_v<std::tuple_element_t<3, concat_view<std::tuple<double, int&>, std::tuple<unsigned&&, long double, float>>>, long double>);
  static_assert(std::is_same_v<std::tuple_element_t<4, concat_view<std::tuple<double, int&>, std::tuple<unsigned&&, long double, float>>>, float>);

  static_assert(get(concat_view {std::tuple{4, 5.}, std::tuple{6.f, 7u}}, std::integral_constant<std::size_t, 0>{}) == 4);
  static_assert(get(concat_view {std::tuple{4, 5.}, std::tuple{6.f, 7u}}, std::integral_constant<std::size_t, 1>{}) == 5.);
  static_assert(get(concat_view {std::tuple{4, 5.}, std::tuple{6.f, 7u}}, std::integral_constant<std::size_t, 2>{}) == 6.f);
  static_assert(get(views::concat(std::tuple{4, 5.}, std::tuple{6.f, 7u}), std::integral_constant<std::size_t, 3>{}) == 7u);

  constexpr auto t1 = std::tuple{9, 10, 11};
  constexpr auto t2 = std::tuple{12, 13, 14};
  static_assert(get(concat_view {t1, t2}, std::integral_constant<std::size_t, 5>{}) == 14);
  static_assert(get(concat_view {std::tuple{9, 10, 11}, t2}, std::integral_constant<std::size_t, 2>{}) == 11);
  static_assert(get(concat_view {t1, std::tuple{12, 13, 14}}, std::integral_constant<std::size_t, 4>{}) == 13);

  std::vector v1 {3, 4, 5};
  std::vector v2 {6, 7, 8};
  auto cat1 = views::concat(v1, v2);
  static_assert(sized_random_access_range<decltype(cat1)>);
  auto itv1 = ranges::begin(cat1);
  EXPECT_EQ(*itv1, 3);
  EXPECT_EQ(*ranges::begin(cat1), 3);
  EXPECT_EQ(*++itv1, 4);
  EXPECT_EQ(*++itv1, 5);
  EXPECT_EQ(*++itv1, 6);
  EXPECT_EQ(*++itv1, 7);
  EXPECT_EQ(*--itv1, 6);
  EXPECT_EQ(itv1[2], 8);
  EXPECT_EQ(cat1[0u], 3);
  EXPECT_EQ(cat1[1u], 4);
  EXPECT_EQ(cat1[2u], 5);
  EXPECT_EQ((cat1[std::integral_constant<std::size_t, 3>{}]), 6);
  EXPECT_EQ((cat1[std::integral_constant<std::size_t, 4>{}]), 7);
  EXPECT_EQ((cat1[std::integral_constant<std::size_t, 5>{}]), 8);

  auto cat2 = views::concat(std::vector {2, 3, 4}, std::vector {5, 6, 7});
  static_assert(sized_random_access_range<decltype(cat2)>);
  auto itv2 = ranges::begin(cat2);
  EXPECT_EQ(*itv2, 2);
  EXPECT_EQ(*ranges::begin(cat2), 2);
  EXPECT_EQ(*++itv2, 3);
  EXPECT_EQ(*++ ++ ++itv2, 6);
  EXPECT_EQ(cat2[0u], 2);
  EXPECT_EQ(cat2[2u], 4);
  EXPECT_EQ((cat2[std::integral_constant<std::size_t, 3>{}]), 5);
  EXPECT_EQ((cat2[std::integral_constant<std::size_t, 5>{}]), 7);

  auto cat3 = views::concat(t1, t2);
  static_assert(sized_random_access_range<decltype(cat3)>);
  auto itv3 = ranges::begin(cat3);
  EXPECT_EQ(*itv3, 9);
  EXPECT_EQ(*ranges::begin(cat3), 9);
  EXPECT_EQ(*++itv3, 10);
  EXPECT_EQ(itv3[3], 13);

  auto cat4 = views::concat(v1, t2);
  static_assert(sized_random_access_range<decltype(cat4)>);
  auto itv4 = ranges::begin(cat4);
  EXPECT_EQ(*itv4, 3);
  EXPECT_EQ(*ranges::begin(cat4), 3);
  EXPECT_EQ(*++itv4, 4);
  EXPECT_EQ(itv4[3], 13);

  constexpr int a1[3] = {3, 4, 5};
  auto cat5 = views::concat(v2, t2, a1, t1);
  static_assert(sized_random_access_range<decltype(cat5)>);
  auto itv5 = ranges::begin(cat5);
  EXPECT_EQ(*itv5, 6);
  EXPECT_EQ(*ranges::begin(cat5), 6);
  EXPECT_EQ(*++itv5, 7);
  EXPECT_EQ(itv5[3], 13);
  EXPECT_EQ(*++itv5, 8);
  EXPECT_EQ(*++itv5, 12);
  EXPECT_EQ(itv5[4], 4);
  EXPECT_EQ(itv5[7], 10);
}


#include "collections/views/slice.hpp"

TEST(collections, slice_view)
{
  using N0 = std::integral_constant<std::size_t, 0>;
  using N1 = std::integral_constant<std::size_t, 1>;
  using N3 = std::integral_constant<std::size_t, 3>;

  using T1 = slice_view<std::tuple<int, double, std::tuple<>&, float&&, unsigned>, N1, N3>;
  static_assert(std::tuple_size_v<T1> == 3);
  static_assert(std::tuple_size_v<slice_view<std::tuple<>, N0, N0>> == 0);
  static_assert(std::is_same_v<std::tuple_element_t<0, T1>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, T1>, std::tuple<>&>);
  static_assert(std::is_same_v<std::tuple_element_t<2, T1>, float&&>);

  constexpr auto t1 = std::tuple{4, 5., 6.f, 8.l, 7u};
  static_assert(get(slice_view {t1, N1{}, N3{}}, std::integral_constant<std::size_t, 0>{}) == 5.);
  static_assert(get(slice_view {t1, N1{}, N3{}}, std::integral_constant<std::size_t, 1>{}) == 6.f);
  static_assert(get(slice_view {t1, N1{}, N3{}}, std::integral_constant<std::size_t, 2>{}) == 8.l);
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

