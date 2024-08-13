/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, scalar_product)
{
  auto m23a = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  auto c22_2 = (M11::Identity() + M11::Identity()).replicate<2,2>();
  auto cxx_22_2 = (M11::Identity() + M11::Identity()).replicate(2,2);

  // Zero * anything
  static_assert(zero<decltype(scalar_product(std::declval<Z23>(), std::declval<double>()))>);
  static_assert(zero<decltype(scalar_product(std::declval<Z23>(), std::declval<std::integral_constant<int, 2>>()))>);
  EXPECT_TRUE(constant_coefficient{scalar_product(M23::Identity() - M23::Identity(), std::integral_constant<int, 2>{})} == 0);
  EXPECT_TRUE(constant_coefficient{scalar_product(M23::Identity() - M23::Identity(), 5)} == 0);

  // Constant * runtime value
  static_assert(constant_matrix<decltype(scalar_product(std::declval<C22_2>(), std::declval<double>())), ConstantType::dynamic_constant>);
  EXPECT_TRUE(constant_coefficient{scalar_product(c22_2, 5)} == 10);
  EXPECT_TRUE(constant_coefficient{scalar_product(cxx_22_2, 5)} == 10);

  // Any object * runtime 0
  EXPECT_TRUE(is_near(scalar_product(m23a, 0), m23a * 0));

  // general case
  EXPECT_TRUE(is_near(scalar_product(m23a, 1), m23a));
  EXPECT_TRUE(is_near(scalar_product(m23a, 3), m23a * 3));
  EXPECT_TRUE(is_near(scalar_product(m23a, std::integral_constant<int, 3>{}), m23a * 3));
}


TEST(eigen3, scalar_quotient)
{
  auto m23a = make_dense_object_from<M23>(12, 24, 36, 48, 60, 72);
  auto c22_2 = (M11::Identity() + M11::Identity()).replicate<2,2>();
  auto cxx_22_2 = (M11::Identity() + M11::Identity()).replicate(2,2);

  // Zero / anything
  static_assert(zero<decltype(scalar_quotient(std::declval<Z23>(), std::declval<double>()))>);
  static_assert(zero<decltype(scalar_quotient(std::declval<Z23>(), std::declval<std::integral_constant<int, 2>>()))>);
  EXPECT_TRUE(constant_coefficient{scalar_quotient(M23::Identity() - M23::Identity(), std::integral_constant<int, 2>{})} == 0);
  EXPECT_TRUE(constant_coefficient{scalar_quotient(M23::Identity() - M23::Identity(), 5)} == 0);

  // Constant / runtime value
  static_assert(constant_matrix<decltype(scalar_quotient(std::declval<C22_2>(), std::declval<double>())), ConstantType::dynamic_constant>);
  EXPECT_TRUE(constant_coefficient{scalar_quotient(c22_2, 2)} == 1);
  EXPECT_TRUE(constant_coefficient{scalar_quotient(cxx_22_2, 2)} == 1);

  // general case
  EXPECT_TRUE(is_near(scalar_quotient(m23a, 1), m23a));
  EXPECT_TRUE(is_near(scalar_quotient(m23a, 3), m23a / 3));
  EXPECT_TRUE(is_near(scalar_quotient(m23a, std::integral_constant<int, 3>{}), m23a / 3));
}
