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


TEST(eigen3, Eigen_MatrixWrapper)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix())> == 2);
  static_assert(zero<decltype(std::declval<Z23>().matrix())>);
  static_assert(identity_matrix<decltype(std::declval<I22>().matrix())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd22_2>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array().matrix())>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().array().matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().matrix()), TriangleType::upper>);

  static_assert(std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen::MatrixWrapper<M32>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen::MatrixWrapper<I22>>()))>);
}
