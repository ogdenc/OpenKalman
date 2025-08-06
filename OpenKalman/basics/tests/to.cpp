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
 * \brief Tests for \ref ranges::to
 */

#include "tests.hpp"
#include "basics/compatibility/views/to.hpp"

using namespace OpenKalman;

TEST(basics, ranges_to)
{
  static_assert(std::is_same_v<decltype(stdcompat::ranges::to<std::vector<int>>(std::declval<std::array<int, 5>&>())), std::vector<int>>);
  static_assert(std::is_same_v<decltype(stdcompat::ranges::to<std::vector<int>>(std::declval<std::array<int, 5>>())), std::vector<int>>);
  static_assert(std::is_same_v<decltype(stdcompat::ranges::to<std::vector>(std::declval<std::array<int, 5>>())), std::vector<int>>);
  static_assert(std::is_same_v<decltype(stdcompat::ranges::to<std::vector>(std::declval<std::array<int, 5>&>())), std::vector<int>>);

  auto a0 = std::array {1, 2, 3, 4, 5};
  auto vec0a = stdcompat::ranges::to<std::vector<int>>(a0);
  static_assert(std::is_same_v<decltype(vec0a), std::vector<int>>);
  EXPECT_EQ(vec0a[2], 3);
  auto vec0b = a0 | stdcompat::ranges::to<std::vector<int>>();
  static_assert(std::is_same_v<decltype(vec0b), std::vector<int>>);
  EXPECT_EQ(vec0b[4], 5);

  auto vec1a = stdcompat::ranges::to<std::vector<int>>(std::array {1, 2, 3, 4, 5});
  static_assert(std::is_same_v<decltype(vec1a), std::vector<int>>);
  EXPECT_EQ(vec1a[2], 3);
  auto vec1b = std::array {1, 2, 3, 4, 5} | stdcompat::ranges::to<std::vector<int>>();
  static_assert(std::is_same_v<decltype(vec1b), std::vector<int>>);
  EXPECT_EQ(vec1b[4], 5);

  auto vec2a = stdcompat::ranges::to<std::vector>(a0);
  static_assert(std::is_same_v<decltype(vec2a), std::vector<int>>);
  EXPECT_EQ(vec2a[2], 3);
  auto vec2b = a0 | stdcompat::ranges::to<std::vector>();
  static_assert(std::is_same_v<decltype(vec2b), std::vector<int>>);
  EXPECT_EQ(vec2b[4], 5);

  auto vec3a = stdcompat::ranges::to<std::vector>(std::array {1, 2, 3, 4, 5});
  static_assert(std::is_same_v<decltype(vec3a), std::vector<int>>);
  EXPECT_EQ(vec3a[2], 3);
  auto vec3b = std::array {1, 2, 3, 4, 5} | stdcompat::ranges::to<std::vector>();
  static_assert(std::is_same_v<decltype(vec3b), std::vector<int>>);
  EXPECT_EQ(vec3b[4], 5);

  auto vec4a = stdcompat::ranges::to<std::tuple<std::array<int, 5>, int, int>>(a0, 8, 9);
  static_assert(std::is_same_v<decltype(vec4a), std::tuple<std::array<int, 5>, int, int>>);
  EXPECT_EQ(std::get<0>(vec4a)[2], 3);
  EXPECT_EQ(std::get<1>(vec4a), 8);
  auto vec4b = a0 | stdcompat::ranges::to<std::tuple<std::array<int, 5>, int, int>>(8, 9);
  static_assert(std::is_same_v<decltype(vec4b), std::tuple<std::array<int, 5>, int, int>>);
  EXPECT_EQ(std::get<0>(vec4b)[3], 4);
  EXPECT_EQ(std::get<2>(vec4b), 9);

  auto vec5a = stdcompat::ranges::to<std::tuple>(a0, 8, 9);
  static_assert(std::is_same_v<decltype(vec5a), std::tuple<std::array<int, 5>, int, int>>);
  EXPECT_EQ(std::get<0>(vec5a)[2], 3);
  EXPECT_EQ(std::get<1>(vec5a), 8);
  auto vec5b = a0 | stdcompat::ranges::to<std::tuple>(8, 9);
  static_assert(std::is_same_v<decltype(vec5b), std::tuple<std::array<int, 5>, int, int>>);
  EXPECT_EQ(std::get<0>(vec5b)[3], 4);
  EXPECT_EQ(std::get<2>(vec5b), 9);

}

