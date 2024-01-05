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


TEST(eigen3, Eigen_Product)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_2>().matrix())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C12_2>().matrix() * std::declval<C21_2>().matrix())> == 8);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_1>().matrix() * std::declval<C11_2>().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_1>().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<C22_m2>().matrix())> == -8);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<C22_2>().matrix())> == 8);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().matrix() * std::declval<C22_2>().matrix())> == 8);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<Cx2_2>().matrix())> == 8);
  static_assert(constant_matrix<decltype(std::declval<C2x_2>().matrix() * std::declval<Cx2_2>().matrix()), ConstantType::dynamic_constant>);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<I22>().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<I22>().matrix() * std::declval<C22_2>().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<C22_2>().matrix())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<M22>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<M22>().matrix())> == 0);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_2>().matrix())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_1>().matrix() * std::declval<C11_2>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_1>().matrix())> == 2);

  static_assert(scalar_constant<std::integral_constant<long long unsigned int, 2>>);
  static_assert(not OpenKalman::detail::internal_constant<std::integral_constant<long long unsigned int, 2>>);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix() * std::declval<Cd22_3>().matrix())> == 6);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix() * std::declval<I22>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<I22>().matrix() * std::declval<Cd22_2>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C12_2>().matrix() * std::declval<C21_2>().matrix())> == 8);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<C22_2>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<M22>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<M22>().matrix() * std::declval<Z22>().matrix())> == 0);

  static_assert(zero<decltype(std::declval<Z11>().matrix() * std::declval<Z11>().matrix())>);
  static_assert(zero<decltype(std::declval<Z12>().matrix() * std::declval<Z21>().matrix())>);

  static_assert(diagonal_matrix<decltype(std::declval<DW21>().matrix() * std::declval<DW21>().matrix())>);
  static_assert(diagonal_matrix<decltype(std::declval<DW21>().matrix() * std::declval<Cd22_2>().matrix())>);

  static_assert(hermitian_matrix<decltype(std::declval<Cd22_2>().matrix() * std::declval<Salv22>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().matrix() * std::declval<Cd22_2>().matrix())>);

  static_assert(hermitian_matrix<decltype(std::declval<Cd22_2>().matrix() * std::declval<Sauv22>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().matrix() * std::declval<Cd22_2>().matrix())>);

  static_assert(not hermitian_matrix<decltype((std::declval<Cd22_2>()*cdouble{1,1}).matrix() * std::declval<Salv22>().matrix())>);
  static_assert(not hermitian_matrix<decltype(std::declval<Salv22>().matrix() * (std::declval<Cd22_2>()*cdouble{1,1}).matrix())>);

  static_assert(not hermitian_matrix<decltype((std::declval<Cd22_2>()*cdouble{1,1}).matrix() * std::declval<Sauv22>().matrix())>);
  static_assert(not hermitian_matrix<decltype(std::declval<Sauv22>().matrix() * (std::declval<Cd22_2>()*cdouble{1,1}).matrix())>);

  static_assert(triangular_matrix<decltype(std::declval<Cd22_2>().matrix() * std::declval<Tlv22>().matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().matrix() * std::declval<Cd22_2>().matrix()), TriangleType::lower>);

  static_assert(triangular_matrix<decltype(std::declval<Cd22_2>().matrix() * std::declval<Tuv22>().matrix()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().matrix() * std::declval<Cd22_2>().matrix()), TriangleType::upper>);

  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().matrix() * std::declval<Tlv22>().matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().matrix() * std::declval<Tuv22>().matrix()), TriangleType::upper>);
}

