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


TEST(eigen3, Eigen_Transpose)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().transpose())> == 2);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().transpose())> == 2);

  static_assert(zero<decltype((std::declval<Z23>()).transpose())>);

  static_assert(identity_matrix<Eigen::Transpose<I22>>);

  static_assert(diagonal_matrix<decltype(std::declval<DW21>().transpose())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().transpose())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().transpose())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().transpose()), triangle_type::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().transpose()), triangle_type::upper>);
}


