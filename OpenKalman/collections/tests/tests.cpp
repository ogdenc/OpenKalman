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
 * \brief Tests for tests.hpp
 */

#include <tuple>
#include <array>
#include "collections/tests/tests.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;

TEST(collections, tests_tuple_like)
{
  EXPECT_TRUE(is_near(std::tuple{1.0, 2.0, 3.0}, std::tuple{1.0, 2.0, 3.0}, std::array{0.0, 0.0, 0.0}));
  EXPECT_TRUE(is_near(std::tuple{1.0, 2.0, 3.0}, std::array{1.1, 1.8, 3.3}, std::tuple{0.11, 0.21, 0.31}));
  EXPECT_FALSE(is_near(std::tuple{1.0, 2.0, 3.0}, std::tuple{1.1, 1.7, 3.3}, std::tuple{0.11, 0.21, 0.31}));

  EXPECT_TRUE(is_near(std::tuple{1.0, 2.0, 3.0}, std::tuple{1.0, 2.0, 3.0}, 0.0));
  EXPECT_TRUE(is_near(std::array{1.0, 2.0, 3.0}, std::tuple{1.1, 1.9, 3.3}, 0.31));
  EXPECT_FALSE(is_near(std::tuple{1.0, 2.0, 3.0}, std::tuple{1.1, 1.7, 3.2}, 0.21));
}


TEST(collections, tests_array)
{
  using A3 = double[3];
  using A3c = const double[3];
  A3 a3 {1, 2, 3};
  A3c a3c {1, 2, 3};
  A3 b3 {1.1, 1.8, 3.3};
  A3 c3 {1.1, 1.7, 3.3};
  A3 e3_0 {0, 0, 0};
  A3 e3 {0.11, 0.21, 0.31};
  EXPECT_TRUE(is_near(a3, a3c, e3_0));
  EXPECT_TRUE(is_near(a3, b3, e3));
  EXPECT_FALSE(is_near(a3, c3, e3));

  EXPECT_TRUE(is_near(a3, a3c, 0.0));
  EXPECT_TRUE(is_near(a3, b3, 0.31));
  EXPECT_FALSE(is_near(a3, c3, 0.29));

  using A23 = double[2][3];
  using A23c = const double[2][3];
  A23 a23 {{1, 2, 3}, {4, 5, 6}};
  A23c a23c {{1, 2, 3}, {4, 5, 6}};
  A23 b23 {{1.1, 1.8, 3.3}, {4.4, 5.5, 6.6}};
  A23 c23 {{1.1, 1.7, 3.3}, {4.4, 4.5, 6.6}};
  A23 e23_0 {{0, 0, 0}, {0, 0, 0}};
  A23 e23 {{0.11, 0.21, 0.31}, {0.41, 0.51, 0.61}};
  EXPECT_TRUE(is_near(a23, a23c, e23_0));
  EXPECT_TRUE(is_near(a23, b23, e23));
  EXPECT_FALSE(is_near(a23, c23, e23));

  EXPECT_TRUE(is_near(a23, a23c, 0.0));
  EXPECT_TRUE(is_near(a23, b23, 0.61));
  EXPECT_FALSE(is_near(a23, c23, 0.59));
}


TEST(collections, tests_sized_random_access_range)
{
  EXPECT_TRUE(is_near(std::vector{1.0, 2.0, 3.0}, std::vector{1.0, 2.0, 3.0}, std::vector{0.0, 0.0, 0.0}));
  EXPECT_TRUE(is_near(std::vector{1.0, 2.0, 3.0}, std::vector{1.1, 1.8, 3.3}, std::vector{0.11, 0.21, 0.31}));
  EXPECT_FALSE(is_near(std::vector{1.0, 2.0, 3.0}, std::vector{1.1, 1.7, 3.3}, std::vector{0.11, 0.21, 0.31}));

  EXPECT_TRUE(is_near(std::vector{1.0, 2.0, 3.0}, std::vector{1.0, 2.0, 3.0}, 0.0));
  EXPECT_TRUE(is_near(std::vector{1.0, 2.0, 3.0}, std::vector{1.1, 1.9, 3.3}, 0.31));
  EXPECT_FALSE(is_near(std::vector{1.0, 2.0, 3.0}, std::vector{1.1, 1.7, 3.2}, 0.21));

  EXPECT_TRUE(is_near(std::vector{1.0, 2.0, 3.0}, std::array{1.0, 2.0, 3.0}, 0.0));
}
