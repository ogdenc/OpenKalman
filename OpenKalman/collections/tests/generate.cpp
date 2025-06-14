/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for \ref collections::transform_view and \ref collections::views::transform.
 */

#include "values/tests/tests.hpp"
#include "values/concepts/fixed.hpp"
#include "values/concepts/dynamic.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/concepts/integral.hpp"
#include "values/concepts/index.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/functions/get.hpp"

inline namespace {
  using namespace OpenKalman;
  using namespace OpenKalman::collections;
  using c0 = std::integral_constant<std::size_t, 0U>;
  using c1 = std::integral_constant<std::size_t, 1U>;
  using c2 = std::integral_constant<std::size_t, 2U>;
  using c3 = std::integral_constant<std::size_t, 3U>;
  using c4 = std::integral_constant<std::size_t, 4U>;
  using c5 = std::integral_constant<std::size_t, 5U>;

  struct fi
  {
    template<typename I>
    constexpr auto
    operator() (I i) const { return i; }
  };

}

#ifdef __cpp_lib_ranges
  namespace rg = std::ranges;
#else
  namespace rg = OpenKalman::ranges;
#endif
  namespace vw = rg::views;

#include "collections/views/generate.hpp"

TEST(collections, generate_view)
{
  constexpr auto i0 = generate_view(fi{}, c3{});
  using I0 = std::decay_t<decltype(i0)>;
  static_assert(rg::view<I0>);
  static_assert(collection_view<I0>);
  static_assert(std::tuple_size_v<std::decay_t<I0>> == 3U);
  static_assert(values::fixed_number_of_v<std::tuple_element_t<0, std::decay_t<I0>>> == 0U);
  static_assert(values::fixed_number_of_v<std::tuple_element_t<1, std::decay_t<I0>>> == 1U);
  static_assert(values::fixed_number_of_v<std::tuple_element_t<2, std::decay_t<I0>>> == 2U);
  static_assert(get(i0, c0{}) == 0U);
  static_assert(get(i0, c1{}) == 1U);
  static_assert(get(i0, c2{}) == 2U);
  static_assert(get(i0, 0U) == 0U);
  static_assert(get(i0, 1U) == 1U);
  static_assert(get(i0, 2U) == 2U);
  static_assert(i0.size() == 3U);
  static_assert(get(generate_view(fi{}, c3{}), c2{}) == 2U);

  auto iti0 = i0.begin();
  EXPECT_EQ(*iti0, 0U);
  EXPECT_EQ(*++iti0, 1U);
  EXPECT_EQ(*(iti0 - 1), 0U);
  EXPECT_EQ(iti0--[1], 2U);
  EXPECT_EQ(*iti0, 0U);
  EXPECT_EQ(*(iti0 + 2), 2U);
  static_assert(*i0.begin() == 0U);
  static_assert(*++i0.begin() == 1U);
  static_assert(i0.begin()[5] == 5U);
  static_assert(*generate_view(fi{}, c5{}).begin() == 0U);
  static_assert(*++generate_view(fi{}, c5{}).begin() == 1U);
  static_assert(generate_view(fi{}, c5{}).begin()[4] == 4U);

  constexpr auto iti1 = generate_view<fi, c5>(c5{}).begin();
  static_assert(*iti1 == 0U);
  static_assert(iti1[5] == 5U);

  static constexpr auto f0 = [](auto&& r)
  {
    return [r = std::tuple {std::forward<decltype(r)>(r)}](auto i)
    {
      return values::operation {std::multiplies<>{}, values::operation {std::multiplies<>{},
        values::operation {std::plus<>{}, values::operation {std::plus<>{}, get(std::get<0>(r), c0{}), i}, c1{}},
        values::operation {std::plus<>{}, values::operation {std::plus<>{}, get(std::get<0>(r), c1{}), i}, c1{}}},
        values::operation {std::plus<>{}, values::operation {std::plus<>{}, get(std::get<0>(r), c2{}), i}, c1{}}};
    };
  };
  constexpr auto t0 = generate_view(f0(generate_view(fi{}, c3{})), c2{});
  static_assert(std::tuple_size_v<std::decay_t<decltype(t0)>> == 2U);
  static_assert(values::fixed_number_of_v<std::tuple_element_t<0, std::decay_t<decltype(t0)>>> == 6U);
  static_assert(values::fixed_number_of_v<std::tuple_element_t<1, std::decay_t<decltype(t0)>>> == 24U);
  static_assert(get(t0, c0{}) == 6U);
  static_assert(get(t0, c1{}) == 24U);

  auto it0 = t0.begin();
  EXPECT_EQ(*it0, 6U);
  EXPECT_EQ(*++it0, 24U);
  EXPECT_EQ(*(it0 - 1), 6U);
  EXPECT_EQ(it0--[1], 60U);
  EXPECT_EQ(*it0, 6U);
  EXPECT_EQ(*(it0 + 2), 60U);
  static_assert(*t0.begin() == 6U);
  static_assert(*++t0.begin() == 24U);
  static_assert(t0.begin()[2] == 60U);

  constexpr auto f0x = f0(generate_view(fi{}, c5{}));
  static_assert(generate_view(f0x, c3{}).begin()[2] == 60U);

  static constexpr auto f1 = [](auto&& r)
  {
    return [r = std::tuple{std::forward<decltype(r)>(r)}](auto i)
    {
      return (get(std::get<0>(r), c0{}) + i + 1) * (get(std::get<0>(r), c1{}) + i + 1) * (get(std::get<0>(r), c2{}) + i + 1);
    };
  };
  constexpr auto t1 = views::generate(f1(generate_view(fi{}, c3{})), c2{});
  static_assert(std::tuple_size_v<std::decay_t<decltype(t1)>> == 2);
  static_assert(std::is_same_v<std::tuple_element_t<0, std::decay_t<decltype(t1)>>, std::size_t>);
  static_assert(std::is_same_v<std::tuple_element_t<1, std::decay_t<decltype(t1)>>, std::size_t>);
  static_assert(get(t1, c0{}) == 6U);
  static_assert(get(t1, c1{}) == 24U);

  auto it1 = t1.begin();
  EXPECT_EQ(*it1, 6U);
  EXPECT_EQ(*++it1, 24U);
  EXPECT_EQ(*(it1 - 1), 6U);
  EXPECT_EQ(it1--[1], 60U);
  EXPECT_EQ(*it1, 6U);
  EXPECT_EQ(*(it1 + 2), 60U);
  static_assert(*t1.begin() == 6U);
  static_assert(*++t1.begin() == 24U);
  static_assert(t1.begin()[2] == 60U);
  constexpr auto f1x = f1(generate_view(fi{}, c4{}));
  static_assert(views::generate(f1x, c3{}).begin()[2] == 60U);

}


TEST(collections, generate_view_unsized)
{
  constexpr auto i0 = generate_view(fi{});
  using I0 = std::decay_t<decltype(i0)>;
  static_assert(rg::view<I0>);
  static_assert(not sized<I0>);
  static_assert(uniformly_gettable<I0>);
  static_assert(collection_view<I0>);
  static_assert(values::fixed_number_of_v<std::tuple_element_t<0, std::decay_t<I0>>> == 0);
  static_assert(values::fixed_number_of_v<std::tuple_element_t<1000, std::decay_t<I0>>> == 1000);
  static_assert(get(i0, c0{}) == 0U);
  static_assert(get(i0, c1{}) == 1U);
  static_assert(get(i0, c2{}) == 2U);
  static_assert(get(i0, 0U) == 0U);
  static_assert(get(i0, 1U) == 1U);
  static_assert(get(i0, 1000U) == 1000U);
  static_assert(get(generate_view<fi>(), 100U) == 100U);
  static_assert(get(generate_view<fi>(fi{}), 101U) == 101U);
  static_assert(get(views::generate(fi{}), 102U) == 102U);

  auto iti0 = i0.begin();
  EXPECT_EQ(*iti0, 0U);
  EXPECT_EQ(*++iti0, 1U);
  EXPECT_EQ(*(iti0 - 1), 0U);
  EXPECT_EQ(iti0--[1], 2U);
  EXPECT_EQ(*iti0, 0U);
  EXPECT_EQ(*(iti0 + 2), 2U);
  static_assert(*i0.begin() == 0U);
  static_assert(*++i0.begin() == 1U);
  static_assert(i0.begin()[5000] == 5000U);
  static_assert(*generate_view(fi{}).begin() == 0U);
  static_assert(*++generate_view(fi{}).begin() == 1U);
  static_assert(generate_view(fi{}).begin()[4000] == 4000U);

  constexpr auto iti1 = generate_view<fi>().begin();
  static_assert(*iti1 == 0U);
  static_assert(iti1[50] == 50U);

  static constexpr auto f0 = [](auto&& r)
  {
    return [r = std::tuple {std::forward<decltype(r)>(r)}](auto i)
    {
      return values::operation {std::multiplies<>{}, values::operation {std::multiplies<>{},
        values::operation {std::plus<>{}, values::operation {std::plus<>{}, get(std::get<0>(r), c0{}), i}, c1{}},
        values::operation {std::plus<>{}, values::operation {std::plus<>{}, get(std::get<0>(r), c1{}), i}, c1{}}},
        values::operation {std::plus<>{}, values::operation {std::plus<>{}, get(std::get<0>(r), c2{}), i}, c1{}}};
    };
  };
  constexpr auto t0 = generate_view(f0(generate_view(fi{})));
  static_assert(values::fixed_number_of_v<std::tuple_element_t<0, std::decay_t<decltype(t0)>>> == 6U);
  static_assert(values::fixed_number_of_v<std::tuple_element_t<1, std::decay_t<decltype(t0)>>> == 24U);
  static_assert(values::fixed_number_of_v<std::tuple_element_t<100, std::decay_t<decltype(t0)>>> == 1061106U);
  static_assert(get(t0, c0{}) == 6U);
  static_assert(get(t0, c1{}) == 24U);

  auto it0 = t0.begin();
  EXPECT_EQ(*it0, 6U);
  EXPECT_EQ(*++it0, 24U);
  EXPECT_EQ(*(it0 - 1), 6U);
  EXPECT_EQ(it0--[1], 60U);
  EXPECT_EQ(*it0, 6U);
  EXPECT_EQ(*(it0 + 2), 60U);
  static_assert(*t0.begin() == 6U);
  static_assert(*++t0.begin() == 24U);
  static_assert(t0.begin()[100] == 1061106U);

  constexpr auto f0x = f0(generate_view(fi{}));
  static_assert(generate_view(f0x).begin()[2] == 60U);

  static constexpr auto f1 = [](auto&& r)
  {
    return [r = std::tuple{std::forward<decltype(r)>(r)}](auto i)
    {
      return (get(std::get<0>(r), c0{}) + i + 1) * (get(std::get<0>(r), c1{}) + i + 1) * (get(std::get<0>(r), c2{}) + i + 1);
    };
  };
  constexpr auto t1 = views::generate(f1(generate_view(fi{})));
  static_assert(std::is_same_v<std::tuple_element_t<0, std::decay_t<decltype(t1)>>, std::size_t>);
  static_assert(std::is_same_v<std::tuple_element_t<1, std::decay_t<decltype(t1)>>, std::size_t>);
  static_assert(get(t1, c0{}) == 6U);
  static_assert(get(t1, c1{}) == 24U);

  auto it1 = t1.begin();
  EXPECT_EQ(*it1, 6U);
  EXPECT_EQ(*++it1, 24U);
  EXPECT_EQ(*(it1 - 1), 6U);
  EXPECT_EQ(it1--[1], 60U);
  EXPECT_EQ(*it1, 6U);
  EXPECT_EQ(*(it1 + 2), 60U);
  static_assert(*t1.begin() == 6U);
  static_assert(*++t1.begin() == 24U);
  static_assert(t1.begin()[2] == 60U);
  constexpr auto f1x = f1(generate_view(fi{}, c4{}));
  static_assert(views::generate(f1x, c3{}).begin()[2] == 60U);

}