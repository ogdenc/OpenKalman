/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear-algebra/interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, Eigen_VectorBlock)
{
  static_assert(eigen_matrix_general<Eigen::VectorBlock<Eigen::Matrix<double, 2, 1>, 1>, true>);
  static_assert(eigen_array_general<Eigen::VectorBlock<Eigen::Array<double, 2, 1>, 1>, true>);
  static_assert(std::is_same_v<double, typename Eigen::VectorBlock<Eigen::Matrix<double, 2, 1>, 0>::Scalar>);

  static_assert(eigen_general<decltype(std::declval<C21_2>().segment<1>(0)), true>);

  static_assert(constant_value_v<decltype(std::declval<C21_2>().segment<1>(0))> == 2);
  static_assert(constant_value_v<decltype(std::declval<C21_m2>().segment(1, 0))> == -2);

  static_assert(constant_diagonal_value_v<decltype(std::declval<C11_2>().segment<1>(0))> == 2);
  static_assert(constant_diagonal_value_v<decltype(std::declval<C11_m2>().segment<1>(0))> == -2);

  static_assert(zero<decltype(std::declval<Z21>().segment<1>(0))>);
  static_assert(zero<decltype(std::declval<Z21>().segment(1, 0))>);

}
