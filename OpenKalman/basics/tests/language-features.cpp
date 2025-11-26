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
  using namespace OpenKalman;
  static_assert(std::is_same_v<decltype(5_uz), std::size_t>);
}


TEST(basics, remove_cvref)
{
  using OpenKalman::stdex::remove_cvref_t;
  static_assert(std::is_same_v<remove_cvref_t<int[5]>, int[5]>);
  static_assert(std::is_same_v<remove_cvref_t<const int[5]>, int[5]>);
}


TEST(basics, bounded_array)
{
  using OpenKalman::stdex::is_bounded_array_v;
  static_assert(is_bounded_array_v<int[5]>);
  static_assert(is_bounded_array_v<const int[5]>);
  static_assert(not is_bounded_array_v<int[]>);
  static_assert(not is_bounded_array_v<int[][5]>);
  static_assert(not is_bounded_array_v<int*>);
}


TEST(basics, reference_wrapper)
{
  static constexpr auto i = 5;
  constexpr auto r = OpenKalman::stdex::reference_wrapper<const int> {i};
  static_assert(r.get() == 5);
  auto j = 6;
  EXPECT_EQ(OpenKalman::stdex::ref(j), 6);
  ++j;
  EXPECT_EQ(OpenKalman::stdex::cref(j), 7);
}
