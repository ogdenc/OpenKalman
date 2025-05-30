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

#include <complex>
#include "values/tests/tests.hpp"
#include "values/classes/Fixed.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;

TEST(values, TestComparison)
{
  EXPECT_TRUE(is_near(1.0, 1.0, 0.0));
  EXPECT_TRUE(is_near(2, 1.0, 2));
  EXPECT_FALSE(is_near(1.0, 2.0, 0.5));
  EXPECT_TRUE(is_near(1.0, 1.1, 0.11));
}


TEST(values, tests_complex)
{
  EXPECT_TRUE(test::is_near(10., 10., 1e-6));
  EXPECT_TRUE(test::is_near(std::complex<double>{3, 4}, std::complex<double>{3 - 1e-9, 4 - 1e-9}, 1e-6));
  EXPECT_FALSE(test::is_near(std::complex<double>{3, 4}, std::complex<double>{3 + 1e-6, 4 + 1e-6}, 1e-9));
  EXPECT_FALSE(test::is_near(std::complex<double>{3, 4}, std::complex<double>{3 - 1e-6, 4 - 1e-6}, 1e-9));
}


TEST(values, tests_fixed)
{
  EXPECT_TRUE(test::is_near(values::Fixed<double, 4>{}, values::Fixed<double, 5>{}, 2));
  EXPECT_TRUE(test::is_near(std::integral_constant<int, 4>{}, std::integral_constant<int, 5>{}, 2));
  EXPECT_TRUE(test::is_near(std::integral_constant<int, 4>{}, 5, 2));
  EXPECT_FALSE(test::is_near(std::integral_constant<int, 4>{}, 6, 1));
  EXPECT_TRUE(test::is_near(values::Fixed<std::complex<double>, 3, 4>{}, values::Fixed<std::complex<double>, 4, 5>{}, 2));
  EXPECT_TRUE(test::is_near(values::Fixed<std::complex<double>, 3, 4>{}, values::Fixed<std::complex<double>, 4, 5>{}, values::Fixed<std::complex<double>, 2, 2>{}));
  EXPECT_TRUE(test::is_near(std::integral_constant<int, 4>{}, values::Fixed<std::complex<double>, 4, 1>{}, 2));
}

