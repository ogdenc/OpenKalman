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


TEST(eigen3, Eigen_Select)
{
  auto br = make_eigen_matrix<bool, 2, 2>(true, false, true, false);
  auto bsa = eigen_matrix_t<bool, 2, 2>::Identity();

  static_assert(one_dimensional<decltype(std::declval<B11_true>().select(std::declval<Mxx>(), std::declval<Mxx>()))>);
  static_assert(one_dimensional<decltype(std::declval<B1x_true>().select(std::declval<Mxx>(), std::declval<Mx1>()))>);
  static_assert(one_dimensional<decltype(std::declval<Bx1_true>().select(std::declval<M1x>(), std::declval<Mxx>()))>);
  static_assert(one_dimensional<decltype(std::declval<Bxx_true>().select(std::declval<M1x>(), std::declval<Mx1>()))>);
  static_assert(one_dimensional<decltype(std::declval<Bxx_true>().select(std::declval<M1x>(), std::declval<DMx>()))>);

  static_assert(square_shaped<decltype(std::declval<B22_true>().select(std::declval<Mxx>(), std::declval<Mxx>()))>);
  static_assert(square_shaped<decltype(std::declval<B2x_true>().select(std::declval<Mxx>(), std::declval<Mx2>()))>);
  static_assert(square_shaped<decltype(std::declval<Bx2_true>().select(std::declval<M2x>(), std::declval<Mxx>()))>);
  static_assert(square_shaped<decltype(std::declval<Bxx_true>().select(std::declval<M2x>(), std::declval<Mx2>()))>);
  static_assert(square_shaped<decltype(std::declval<Bxx_true>().select(std::declval<DMx>(), std::declval<Mxx>()))>);

  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<C22_2>(), std::declval<Z22>()))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<C22_2>(), std::declval<M22>()))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<C22_2>(), std::declval<Z22>()))> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Z22>()))> == 0);
  static_assert(constant_coefficient_v<decltype(br.select(std::declval<C22_2>(), std::declval<C22_2>()))> == 2);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<Cd22_2>(), std::declval<Z22>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<Cd22_2>(), std::declval<M22>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<Cd22_2>(), std::declval<Z22>()))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(br.select(std::declval<Cd22_2>(), std::declval<Cd22_2>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<C22_2>(), std::declval<Z22>()))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Z22>()))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B11_true>().select(std::declval<C11_2>(), std::declval<Z22>()))> == 2);

  static_assert(zero<decltype(std::declval<B22_true>().select(std::declval<Z22>(), M22::Identity()))>);
  static_assert(not zero<decltype(std::declval<B22_true>().select(std::declval<I22>(), std::declval<Z22>()))>);
  static_assert(zero<decltype(std::declval<B22_false>().select(std::declval<I22>(), std::declval<Z22>()))>);
  static_assert(zero<decltype(br.select(std::declval<Z22>(), std::declval<Z22>()))>);

  static_assert(diagonal_matrix<decltype(std::declval<B22_true>().select(std::declval<DW21>(), std::declval<M22>()))>);
  static_assert(diagonal_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<DW21>()))>);

  static_assert(hermitian_matrix<decltype(std::declval<B22_true>().select(std::declval<Salv22>(), std::declval<M22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<B22_true>().select(std::declval<Sauv22>(), std::declval<M22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Salv22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Sauv22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Salv22>(), std::declval<Salv22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Sauv22>(), std::declval<Sauv22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Salv22>(), std::declval<Sauv22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Sauv22>(), std::declval<Salv22>()))>);

  static_assert(triangular_matrix<decltype(std::declval<B22_true>().select(std::declval<Tlv22>(), std::declval<M22>())), triangle_type::lower>);
  static_assert(triangular_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Tlv22>())), triangle_type::lower>);

  static_assert(triangular_matrix<decltype(std::declval<B22_true>().select(std::declval<Tuv22>(), std::declval<M22>())), triangle_type::upper>);
  static_assert(triangular_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Tuv22>())), triangle_type::upper>);
}

