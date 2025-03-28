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

TEST(collections, tests_tuple)
{
  EXPECT_TRUE(is_near(std::tuple{1.0, 2.0, 3.0}, std::tuple{1.0, 2.0, 3.0}, std::array{0.0, 0.0, 0.0}));
  EXPECT_TRUE(is_near(std::tuple{1.0, 2.0, 3.0}, std::array{1.1, 1.8, 3.3}, std::tuple{0.11, 0.21, 0.31}));
  EXPECT_FALSE(is_near(std::tuple{1.0, 2.0, 3.0}, std::tuple{1.1, 1.7, 3.3}, std::tuple{0.11, 0.21, 0.31}));

  EXPECT_TRUE(is_near(std::tuple{1.0, 2.0, 3.0}, std::tuple{1.0, 2.0, 3.0}, 0.0));
  EXPECT_TRUE(is_near(std::array{1.0, 2.0, 3.0}, std::tuple{1.1, 1.9, 3.3}, 0.31));
  EXPECT_FALSE(is_near(std::tuple{1.0, 2.0, 3.0}, std::tuple{1.1, 1.7, 3.2}, 0.21));
}
