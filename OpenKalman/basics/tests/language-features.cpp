/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for language-features.hpp
 */

#include <gtest/gtest.h>
#include "basics/compatibility/language-features.hpp"


TEST(basics, uz_literal)
{
  static_assert(std::is_same_v<decltype(5_uz), std::size_t>);
}


#ifndef __cpp_lib_remove_cvref
TEST(basics, remove_cvref)
{
  static_assert(std::is_same_v<OpenKalman::remove_cvref_t<int[5]>, int[5]>);
  static_assert(std::is_same_v<OpenKalman::remove_cvref_t<const int[5]>, int[5]>);
}
#endif


#ifndef __cpp_lib_bounded_array_traits
TEST(basics, bounded_array)
{
  static_assert(OpenKalman::is_bounded_array_v<int[5]>);
  static_assert(OpenKalman::is_bounded_array_v<const int[5]>);
  static_assert(not OpenKalman::is_bounded_array_v<int[]>);
  static_assert(not OpenKalman::is_bounded_array_v<int[][5]>);
  static_assert(not OpenKalman::is_bounded_array_v<int*>);
}
#endif


#if __cplusplus < 202002L
TEST(basics, reference_wrapper)
{
  static constexpr auto i = 5;
  constexpr auto r = OpenKalman::reference_wrapper<const int> {i};
  static_assert(r.get() == 5);
  auto j = 6;
  EXPECT_EQ(OpenKalman::ref(j), 6);
  ++j;
  EXPECT_EQ(OpenKalman::cref(j), 7);
}
#endif


struct tup1
{
  int val1 = 1;
  double val2 = 3.;

  template<std::size_t i>
  constexpr decltype(auto) get() { return i == 0 ? val1 : val2; }
};

struct tup2
{
  int val1 = 4;
  double val2 = 5.;

  template<std::size_t i>
  friend constexpr decltype(auto) get(const tup2& t) { return i == 0 ? t.val1 : t.val2; }
};

#include <tuple>

TEST(basics, generalized_std_get)
{
  using OpenKalman::internal::generalized_std_get;
  constexpr auto t1 = std::tuple{1, 2., 3.f, 4u};
  static_assert(generalized_std_get<1>(t1) == 2);
  static_assert(std::is_same_v<decltype(generalized_std_get<1>(t1)), const double&>);
  static_assert(generalized_std_get<2>(std::tuple{1, 2., 3.f, 4u}) == 3);
  static_assert(std::is_same_v<decltype(generalized_std_get<2>(std::tuple{1, 2., 3.f, 4u})), float&&>);
  static_assert(generalized_std_get<0>(tup1{}) == 1);
  static_assert(std::is_same_v<decltype(generalized_std_get<1>(tup1{})), double>);
  static_assert(generalized_std_get<0>(tup2{}) == 4);
  static_assert(std::is_same_v<decltype(generalized_std_get<1>(tup2{})), double>);
}
