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

using namespace OpenKalman;
using namespace OpenKalman::collections;

#include "collections/views/identity.hpp"

#ifdef __cpp_lib_ranges
  namespace ranges = std::ranges;
#endif

TEST(collections, identity_view)
{
  static_assert(tuple_like<identity_view<std::tuple<int, double>>>);
  static_assert(not sized_random_access_range<identity_view<std::tuple<int, double>>>);

#ifdef __cpp_lib_ranges
  static_assert(not std::ranges::view<identity_view<std::tuple<int, double>>>);
#endif
  static_assert(std::tuple_size_v<identity_view<std::tuple<int, double>>> == 2);
  static_assert(std::tuple_size_v<identity_view<std::tuple<>>> == 0);
  static_assert(std::is_same_v<std::tuple_element_t<0, identity_view<std::tuple<int&, double>>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<0, identity_view<std::tuple<int&, double>&>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<0, identity_view<std::tuple<int&, double>&&>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<1, identity_view<std::tuple<int&, double>>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, identity_view<std::tuple<int&, double>&>>, double&>);
  static_assert(std::is_same_v<std::tuple_element_t<1, identity_view<std::tuple<int&, double>&&>>, double&&>);

  static_assert(get(identity_view {std::tuple{4, 5.}}, std::integral_constant<std::size_t, 0>{}) == 4);
  static_assert(get(identity_view {std::tuple{4, 5.}}, std::integral_constant<std::size_t, 1>{}) == 5.);
  static_assert(get(identity_view {std::tuple{4, std::tuple{}}}, std::integral_constant<std::size_t, 1>{}) == std::tuple{});
  static_assert(equal_to{}(identity_view {std::tuple{4, 5.}}, std::tuple{4, 5.}));
  static_assert(equal_to{}(std::tuple{4, 5.}, views::identity(std::tuple{4, 5.})));
  static_assert(equal_to{}(views::identity(std::tuple{4, 5.}), views::identity(std::tuple{4, 5.})));
  EXPECT_TRUE(equal_to{}(views::identity(std::vector{4, 5, 6}), std::vector{4, 5, 6}));
  EXPECT_TRUE(equal_to{}(std::array{4, 5, 6}, identity_view {std::array{4, 5, 6}}));
  static_assert(equal_to{}(identity_view {std::tuple{4, std::tuple{}}}, std::tuple{4, std::tuple{}}));

  //static_assert(sized_random_access_range<identity_view<std::vector<int>>>);
  //static_assert(sized_random_access_range<identity_view<std::array<int, 7>>>);
  //static_assert(collections::internal::begin_is_defined<identity_view<std::vector<int>>>::value);

#ifdef __cpp_lib_ranges
  static_assert(std::ranges::view<identity_view<std::array<int, 7>>>);
  static_assert(std::ranges::view<identity_view<std::vector<int>>>);
  static_assert(std::ranges::view<identity_view<std::vector<int>&>>);
  static_assert(std::ranges::view<identity_view<const std::vector<int>&>>);
#endif
  auto v1 = std::vector{4, 5, 6, 7, 8};
  static_assert(ranges::range<identity_view<std::vector<int>&>>);
  static_assert(ranges::range<identity_view<std::vector<int>>>);
  EXPECT_EQ((views::identity(v1)[0u]), 4);
  EXPECT_EQ((views::identity(v1)[4u]), 8);
  EXPECT_EQ((views::identity(v1)[std::integral_constant<std::size_t, 4>{}]), 8);
  EXPECT_EQ(*ranges::begin(views::identity(v1)), 4);
  EXPECT_EQ(*--ranges::end(views::identity(v1)), 8);
  EXPECT_EQ(ranges::size(views::identity(v1)), 5);

  EXPECT_EQ(views::identity(v1).front(), 4);
  EXPECT_EQ(views::identity(std::vector{4, 5, 6, 7, 8}).front(), 4);
  EXPECT_EQ(views::identity(v1).back(), 8);
  EXPECT_EQ(views::identity(std::vector{4, 5, 6, 7, 8}).back(), 8);
  EXPECT_EQ(*ranges::cbegin(views::identity(v1)), 4);
  EXPECT_EQ(*--ranges::cend(views::identity(v1)), 8);
  EXPECT_TRUE(views::identity(std::vector<int>{}).empty());
  EXPECT_FALSE(views::identity(std::vector{4, 5, 6, 7, 8}).empty());
  EXPECT_FALSE(views::identity(std::vector<int>{}));
  EXPECT_TRUE(views::identity(std::vector{4, 5, 6, 7, 8}));

  identity_view<std::vector<int>> id1_v1;
  id1_v1 = v1;
  EXPECT_EQ((id1_v1[3u]), 7);

  auto v2 = std::vector{9, 10, 11, 12, 13};
  identity_view<std::vector<int>&> id1_v2 {v1};
  id1_v2 = views::identity(v2);
  EXPECT_EQ((id1_v2[3u]), 12);
  id1_v2 = views::identity(v1);
  EXPECT_EQ((id1_v2[3u]), 7);

  identity_view<std::vector<int>> id2_v1 {std::vector{4, 5, 6, 7, 8}};
  EXPECT_EQ((id2_v1[2u]), 6);

  EXPECT_TRUE(equal_to{}(views::identity(v1), v1));
  EXPECT_TRUE(equal_to{}(v1, views::identity(v1)));
  auto v1b = ranges::begin(v1);
  EXPECT_EQ(*v1b, 4);
  EXPECT_EQ(*++v1b, 5);
  EXPECT_EQ(v1b[3], 8);
  EXPECT_EQ(*--v1b, 4);
  EXPECT_EQ(*--ranges::end(v1), 8);

  constexpr int a1[5] = {1, 2, 3, 4, 5};
  static_assert(ranges::range<identity_view<int[5]>>);
  static_assert((views::identity(a1)[0u]) == 1);
  static_assert((views::identity(a1)[4u]) == 5);
  static_assert(*ranges::begin(views::identity(a1)) == 1);
  static_assert(*(ranges::end(views::identity(a1)) - 1) == 5);
  static_assert(*ranges::cbegin(views::identity(a1)) == 1);
  static_assert(*(ranges::cend(views::identity(a1)) - 1) == 5);
  static_assert(ranges::size(views::identity(a1)) == 5);
  static_assert(ranges::begin(views::identity(a1))[2] == 3);
  static_assert(views::identity(a1).front() == 1);
  static_assert(views::identity(a1).back() == 5);
  static_assert(not views::identity(a1).empty());
}


/*#include "collections/views/single.hpp"

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

  constexpr auto s1 = views::single(7);
  constexpr auto is1 = s1.begin();
  static_assert(*is1 == 7);
  static_assert(*--ranges::end(s1) == 7);
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
  static_assert(get(reverse_view {std::tuple{4, std::tuple{}}}, std::integral_constant<std::size_t, 0>{}) == std::tuple{});
  static_assert(get(reverse_view {std::array{4, 5}}, std::integral_constant<std::size_t, 0>{}) == 5);

  auto a1 = std::array{4, 5};
  EXPECT_EQ((reverse_view {a1}[0u]), 5);
  EXPECT_EQ((reverse_view {a1}[1u]), 4);

  auto v1 = std::vector{3, 4, 5};
  EXPECT_EQ((views::reverse(v1)[0u]), 5);
  EXPECT_EQ((views::reverse(v1)[1u]), 4);
  EXPECT_EQ((views::reverse(v1)[2u]), 3);
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

  static_assert(get(replicate_view {std::tuple{4, 5.}, std::integral_constant<std::size_t, 2>{}}, std::integral_constant<std::size_t, 0>{}) == 4.);
  static_assert(get(replicate_view {std::tuple{4, 5.}, std::integral_constant<std::size_t, 2>{}}, std::integral_constant<std::size_t, 1>{}) == 5);
  static_assert(get(replicate_view {std::tuple{4, 5.}, std::integral_constant<std::size_t, 2>{}}, std::integral_constant<std::size_t, 2>{}) == 4.);
  static_assert(get(replicate_view {std::tuple{4, 5.}, std::integral_constant<std::size_t, 2>{}}, std::integral_constant<std::size_t, 3>{}) == 5);
  static_assert(get(views::replicate(std::tuple{4, std::tuple{}}, std::integral_constant<std::size_t, 2>{}), std::integral_constant<std::size_t, 1>{}) == std::tuple{});

  auto v1 = std::vector{3, 4, 5};
  EXPECT_EQ((views::replicate(v1, 2u)[0u]), 3);
  EXPECT_EQ((views::replicate(v1, 2u)[1u]), 4);
  EXPECT_EQ((views::replicate(v1, 2u)[2u]), 5);
  EXPECT_EQ((views::replicate(v1, 2u)[std::integral_constant<std::size_t, 3>{}]), 3);
  EXPECT_EQ((views::replicate(v1, 2u)[std::integral_constant<std::size_t, 4>{}]), 4);
  EXPECT_EQ((views::replicate(v1, 2u)[std::integral_constant<std::size_t, 5>{}]), 5);
}


#include "collections/views/concat.hpp"

TEST(collections, concat_view)
{
  static_assert(std::tuple_size_v<concat_view<std::tuple<int, double>, std::tuple<float, std::tuple<>>>> == 4);
  static_assert(std::tuple_size_v<concat_view<std::tuple<>, std::tuple<>>> == 0);
  static_assert(std::tuple_size_v<concat_view<std::tuple<>, std::tuple<int>>> == 1);
  static_assert(std::tuple_size_v<concat_view<std::tuple<double>, std::tuple<>>> == 1);
  static_assert(std::tuple_size_v<concat_view<std::tuple<double, int>, std::tuple<unsigned, std::tuple<>, float>>> == 5);
  static_assert(std::is_same_v<std::tuple_element_t<0, concat_view<std::tuple<double, int&>, std::tuple<unsigned&&, std::tuple<>, float>>>, double&&>);
  static_assert(std::is_same_v<std::tuple_element_t<1, concat_view<std::tuple<double, int&>, std::tuple<unsigned&&, std::tuple<>, float>>>, int&>);
  static_assert(std::is_same_v<std::tuple_element_t<2, concat_view<std::tuple<double, int&>, std::tuple<unsigned&&, std::tuple<>, float>>>, unsigned&&>);
  static_assert(std::is_same_v<std::tuple_element_t<3, concat_view<std::tuple<double, int&>, std::tuple<unsigned&&, std::tuple<>, float>>>, std::tuple<>&&>);
  static_assert(std::is_same_v<std::tuple_element_t<4, concat_view<std::tuple<double, int&>, std::tuple<unsigned&&, std::tuple<>, float>>>, float&&>);

  static_assert(get(concat_view {std::tuple{4, 5.}, std::tuple{6.f, std::tuple{}}}, std::integral_constant<std::size_t, 0>{}) == 4);
  static_assert(get(concat_view {std::tuple{4, 5.}, std::tuple{6.f, std::tuple{}}}, std::integral_constant<std::size_t, 1>{}) == 5.);
  static_assert(get(concat_view {std::tuple{4, 5.}, std::tuple{6.f, std::tuple{}}}, std::integral_constant<std::size_t, 2>{}) == 6.f);
  static_assert(get(views::concat(std::tuple{4, 5.}, std::tuple{6.f, std::tuple{}}), std::integral_constant<std::size_t, 3>{}) == std::tuple{});

  std::vector v1 {3, 4, 5};
  std::vector v2 {6, 7, 8};
  auto cat1 = views::concat(v1, v2);
  EXPECT_EQ(cat1[0u], 3);
  EXPECT_EQ(cat1[1u], 4);
  EXPECT_EQ(cat1[2u], 5);
  EXPECT_EQ((cat1[std::integral_constant<std::size_t, 3>{}]), 6);
  EXPECT_EQ((cat1[std::integral_constant<std::size_t, 4>{}]), 7);
  EXPECT_EQ((cat1[std::integral_constant<std::size_t, 5>{}]), 8);
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

  constexpr auto t2 = std::tuple{4, 5., 6.f, std::tuple{}, 7u};
  static_assert(get(slice_view {t2, N1{}, N3{}}, std::integral_constant<std::size_t, 0>{}) == 5.);
  static_assert(get(slice_view {t2, N1{}, N3{}}, std::integral_constant<std::size_t, 1>{}) == 6.f);
  static_assert(get(slice_view {t2, N1{}, N3{}}, std::integral_constant<std::size_t, 2>{}) == std::tuple{});
  static_assert(get(views::slice(t2, N1{}, N3{}), std::integral_constant<std::size_t, 2>{}) == std::tuple{});
  static_assert(get(views::slice(std::move(t2), N1{}, N3{}), std::integral_constant<std::size_t, 2>{}) == std::tuple{});

  auto v1 = std::vector {3, 4, 5, 6, 7};
  EXPECT_EQ((slice_view(v1, N1{}, N3{}).size()), 3);
  EXPECT_EQ((slice_view(v1, N1{}, 3u).size()), 3);

  EXPECT_EQ((slice_view(v1, N1{}, N3{})[0u]), 4);
  EXPECT_EQ((views::slice(v1, N1{}, 3u)[1u]), 5);
  EXPECT_EQ((views::slice(v1, 1u, N3{})[2u]), 6);
  EXPECT_EQ((views::slice(v1, 1u, 3u)[N0{}]), 4);
}

*/