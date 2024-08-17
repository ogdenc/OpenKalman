/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, Eigen_Block)
{
  static_assert(eigen_matrix_general<decltype(std::declval<C22_2>().matrix().block<2,1>(0, 0)), true>);
  static_assert(eigen_array_general<decltype(std::declval<C22_2>().block<2,1>(0, 0)), true>);
  static_assert(not eigen_matrix_general<decltype(std::declval<C22_2>().block<2,1>(0, 0)), true>);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().block<2, 1>(0, 0))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().block(2, 1, 0, 0))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().block<1, 2>(0, 0))> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().block<1, 1>(0, 0))> == 0);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().block<1, 1>(0, 0))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().block<1, 1>(0, 0))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().block<2, 2>(0, 0))> == 0);

  static_assert(zero<decltype(std::declval<Z22>().block<2, 1>(0, 0))>);
  static_assert(zero<decltype(std::declval<Z22>().block(2, 1, 0, 0))>);
  static_assert(zero<decltype(std::declval<Z22>().block<1, 1>(0, 0))>);

  static_assert(writable<Eigen::Block<M33, 3, 1, true>>);
  static_assert(writable<Eigen::Block<M33, 3, 2, true>>);
}

