/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear-algebra/interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, sum)
{
  auto m23a = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  auto m23b = make_dense_object_from<M23>(7, 8, 9, 10, 11, 12);
  auto m23c = make_dense_object_from<M23>(8, 10, 12, 14, 16, 18);

  M2x m2x_3a {m23a};
  M2x m2x_3b {m23b};
  Mx3 mx3_2a {m23a};
  Mx3 mx3_2b {m23b};
  Mxx mxx_23a {m23a};
  Mxx mxx_23b {m23b};

  // single argument

  EXPECT_TRUE(is_near(sum(m23a), m23a));

  // zero

  static_assert(zero<decltype(sum(std::declval<Z22>()))>);
  static_assert(zero<decltype(sum(std::declval<Z23>(), std::declval<Z23>()))>);
  static_assert(zero<decltype(sum(std::declval<Z11>(), std::declval<Z11>(), std::declval<Z11>()))>);

  // constant

  static_assert(constant_value_v<decltype(sum(std::declval<C22_2>()))> == 2);
  static_assert(constant_value_v<decltype(sum(std::declval<Z22>(), std::declval<C22_m2>()))> == -2);
  static_assert(constant_value_v<decltype(sum(std::declval<Z22>(), std::declval<Z22>(), std::declval<C22_m2>()))> == -2);

  // constant diagonal

  static_assert(constant_diagonal_value_v<decltype(sum(std::declval<Cd22_2>()))> == 2);
  static_assert(constant_diagonal_value_v<decltype(sum(std::declval<Z22>(), std::declval<Cd22_3>()))> == 3);
  static_assert(constant_diagonal_value_v<decltype(sum(std::declval<Z22>(), std::declval<Cd22_m2>(), std::declval<Z22>()))> == -2);

  // diagonal

  static_assert(not diagonal_matrix<decltype(sum(mxx_23a))>);
  static_assert(not diagonal_matrix<decltype(sum(mxx_23a, mxx_23b))>);
  static_assert(diagonal_matrix<decltype(sum(std::declval<DW21>()))>);
  static_assert(diagonal_matrix<decltype(sum(std::declval<DW2x>()))>);
  static_assert(diagonal_matrix<decltype(sum(std::declval<DWx1>()))>);
  static_assert(diagonal_matrix<decltype(sum(std::declval<DWxx>()))>);
  static_assert(diagonal_matrix<decltype(sum(std::declval<DW21>(), std::declval<DM2>()))>);
  static_assert(diagonal_matrix<decltype(sum(std::declval<DW21>(), std::declval<DMx>()))>);
  static_assert(diagonal_matrix<decltype(sum(std::declval<DW2x>(), std::declval<DMx>()))>);
  static_assert(diagonal_matrix<decltype(sum(std::declval<DWx1>(), std::declval<DMx>()))>);
  static_assert(diagonal_matrix<decltype(sum(std::declval<DWxx>(), std::declval<DMx>()))>);

  // general

  EXPECT_TRUE(is_near(sum(m23a, m23b), m23c));
  EXPECT_TRUE(is_near(sum(m2x_3a, mx3_2b), m23c));
  EXPECT_TRUE(is_near(sum(mx3_2a, m2x_3b), m23c));
  EXPECT_TRUE(is_near(sum(mxx_23a, mxx_23b), m23c));
  static_assert(not dynamic_dimension<decltype(sum(m2x_3a, mx3_2b)), 0>);
  static_assert(not dynamic_dimension<decltype(sum(m2x_3a, mx3_2b)), 1>);
  static_assert(not dynamic_dimension<decltype(sum(m2x_3a, m2x_3b)), 0>);
  static_assert(dynamic_dimension<decltype(sum(m2x_3a, m2x_3b)), 1>);
  static_assert(dynamic_dimension<decltype(sum(mx3_2a, mx3_2b)), 0>);
  static_assert(not dynamic_dimension<decltype(sum(mx3_2a, mx3_2b)), 1>);
  static_assert(not dynamic_dimension<decltype(sum(mx3_2a, m2x_3b)), 0>);
  static_assert(not dynamic_dimension<decltype(sum(mx3_2a, m2x_3b)), 1>);
  static_assert(not dynamic_dimension<decltype(sum(mxx_23a, m23b)), 0>);
  static_assert(not dynamic_dimension<decltype(sum(mxx_23a, m23b)), 1>);
  static_assert(not dynamic_dimension<decltype(sum(m23a, mxx_23b)), 0>);
  static_assert(not dynamic_dimension<decltype(sum(m23a, mxx_23b)), 1>);
}

