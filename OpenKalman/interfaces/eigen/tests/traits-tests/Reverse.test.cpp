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


TEST(eigen3, Eigen_Reverse)
{
  static_assert(index_dimension_of_v<Eigen::Reverse<M2x, Eigen::Vertical>, 0> == 2);
  static_assert(index_dimension_of_v<Eigen::Reverse<Mx2, Eigen::Vertical>, 1> == 2);
  static_assert(index_dimension_of_v<Eigen::Reverse<M2x, Eigen::Horizontal>, 0> == 2);
  static_assert(index_dimension_of_v<Eigen::Reverse<Mx2, Eigen::BothDirections>, 1> == 2);

  static_assert(one_dimensional<Eigen::Reverse<M11, Eigen::Vertical>>);
  static_assert(one_dimensional<Eigen::Reverse<M1x, Eigen::Horizontal>, Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<Eigen::Reverse<Mx1, Eigen::BothDirections>, Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<Eigen::Reverse<Mxx, Eigen::Vertical>, Qualification::depends_on_dynamic_shape>);

  static_assert(square_shaped<Eigen::Reverse<M22, Eigen::BothDirections>>);
  static_assert(square_shaped<Eigen::Reverse<M2x, Eigen::BothDirections>, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Eigen::Reverse<Mx2, Eigen::BothDirections>, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Eigen::Reverse<Mxx, Eigen::BothDirections>, Qualification::depends_on_dynamic_shape>);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().reverse())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().reverse())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C2x_2>().reverse())> == 2);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().reverse())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().reverse())> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reverse<Z22, Eigen::Vertical>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reverse<M11::IdentityReturnType, Eigen::Horizontal>> == 1);

  static_assert(zero<decltype(std::declval<Z23>().reverse())>);

  static_assert(identity_matrix<Eigen::Reverse<I22, Eigen::BothDirections>>);

  static_assert(diagonal_matrix<decltype(std::declval<DW21>().reverse())>);
  static_assert(diagonal_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<C1x_2, Eigen::Vertical>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<Cx1_2, Eigen::Vertical>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<Cxx_2, Eigen::Vertical>>);
  static_assert(diagonal_matrix<Eigen::Reverse<C11_2, Eigen::Horizontal>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<C1x_2, Eigen::Horizontal>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<Cx1_2, Eigen::Horizontal>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<Cxx_2, Eigen::Horizontal>>);

  static_assert(diagonal_matrix<Eigen::Reverse<Cd22_2, Eigen::BothDirections>>);
  static_assert(diagonal_matrix<Eigen::Reverse<Cd2x_2, Eigen::BothDirections>>);
  static_assert(diagonal_matrix<Eigen::Reverse<Cdx2_2, Eigen::BothDirections>>);
  static_assert(diagonal_matrix<Eigen::Reverse<Cdxx_2, Eigen::BothDirections>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<Cd22_2, Eigen::Vertical>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<Cd22_2, Eigen::Horizontal>>);

  static_assert(diagonal_matrix<Eigen::Reverse<DW21, Eigen::BothDirections>>);
  static_assert(diagonal_matrix<Eigen::Reverse<DW2x, Eigen::BothDirections>>);
  static_assert(diagonal_matrix<Eigen::Reverse<DWx1, Eigen::BothDirections>>);
  static_assert(diagonal_matrix<Eigen::Reverse<DWxx, Eigen::BothDirections>>);

  static_assert(not diagonal_matrix<Eigen::Reverse<DW21, Eigen::Horizontal>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<DW21, Eigen::Vertical>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<DW2x, Eigen::Horizontal>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<DW2x, Eigen::Vertical>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<DWx1, Eigen::Horizontal>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<DWx1, Eigen::Vertical>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<DWxx, Eigen::Horizontal>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<DWxx, Eigen::Vertical>>);

  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().reverse())>);
  static_assert(hermitian_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);

  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().reverse())>);
  static_assert(hermitian_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);

  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().reverse()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv2x>().reverse()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuvx2>().reverse()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuvxx>().reverse()), TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::Reverse<M11, Eigen::Vertical>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::Reverse<M11, Eigen::Horizontal>, TriangleType::lower>);

  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().reverse()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv2x>().reverse()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tlvx2>().reverse()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tlvxx>().reverse()), TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::Reverse<M11, Eigen::Vertical>, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::Reverse<M11, Eigen::Horizontal>, TriangleType::upper>);
}

