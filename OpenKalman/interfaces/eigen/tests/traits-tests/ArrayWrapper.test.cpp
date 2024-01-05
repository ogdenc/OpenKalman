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


TEST(eigen3, Eigen_ArrayWrapper)
{
  static_assert(eigen_array_general<Eigen::ArrayWrapper<M32>, true>);
  static_assert(not eigen_matrix_general<Eigen::ArrayWrapper<M32>, true>);
  static_assert(self_contained<Eigen::ArrayWrapper<I22>>);
  static_assert(not self_contained<Eigen::ArrayWrapper<M32>>);

  static_assert(std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen::ArrayWrapper<M32>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen::ArrayWrapper<I22>>()))>);

  static_assert(constant_coefficient_v<Eigen::ArrayWrapper<C22_2>> == 2);
  static_assert(constant_coefficient_v<Eigen::ArrayWrapper<C2x_2>> == 2);
  static_assert(constant_coefficient_v<Eigen::ArrayWrapper<Cx2_2>> == 2);
  static_assert(constant_coefficient_v<Eigen::ArrayWrapper<Cxx_2>> == 2);

  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<Cd22_2>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<Cd2x_2>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<Cdx2_2>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<Cdxx_2>> == 2);

  static_assert(zero<Eigen::ArrayWrapper<Z22>>);
  static_assert(zero<Eigen::ArrayWrapper<Z21>>);
  static_assert(zero<Eigen::ArrayWrapper<Z23>>);

  static_assert(diagonal_matrix<DW21>);
  static_assert(not hermitian_adapter<Eigen::ArrayWrapper<Z22>>);
  static_assert(not hermitian_adapter<Eigen::ArrayWrapper<C22_2>>);
  static_assert(triangular_matrix<DW21, TriangleType::lower>);
  static_assert(triangular_matrix<DW21, TriangleType::upper>);
}

