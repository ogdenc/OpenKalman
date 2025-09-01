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


TEST(eigen3, Eigen_DiagonalMatrix)
{
  static_assert(index_dimension_of_v<DM2, 0> == 2);
  static_assert(index_dimension_of_v<DMx, 0> == dynamic_size);

  static_assert(index_dimension_of_v<DM2, 1> == 2);
  static_assert(index_dimension_of_v<DMx, 1> == dynamic_size);

  static_assert(square_shaped<DMx>);

  static_assert(diagonal_matrix<DM2>);
  static_assert(diagonal_matrix<DMx>);

  static_assert(triangular_matrix<DMx, triangle_type::lower>);

  static_assert(std::is_same_v<nested_object_of_t<Eigen::DiagonalMatrix<double, 2>>, M21&&>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalMatrix<double, 3>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalMatrix<double, Eigen::Dynamic>>, Mxx>);

  static_assert(diagonal_matrix<Eigen::DiagonalMatrix<double, 3>>);

  static_assert(not writable<Eigen::DiagonalMatrix<double, 3>>);

  /*static_assert(element_gettable<Eigen::DiagonalMatrix<double, 2>, 2>);
  static_assert(not element_gettable<Eigen::DiagonalMatrix<double, 2>, 1>);
  static_assert(element_gettable<Eigen::DiagonalMatrix<double, Eigen::Dynamic>, 2>);
  static_assert(not element_gettable<Eigen::DiagonalMatrix<double, Eigen::Dynamic>, 1>);

  static_assert(element_gettable<Eigen::DiagonalWrapper<M21>, 2>);
  static_assert(not element_gettable<Eigen::DiagonalWrapper<M21>, 1>);
  static_assert(element_gettable<Eigen::DiagonalWrapper<M2x>, 2>);
  static_assert(not element_gettable<Eigen::DiagonalWrapper<M2x>, 1>);
  static_assert(element_gettable<Eigen::DiagonalWrapper<Mx1>, 2>);
  static_assert(not element_gettable<Eigen::DiagonalWrapper<Mx1>, 1>);*/

  static_assert(Eigen3::eigen_general<Eigen::DiagonalMatrix<double, 3>, true>);
  static_assert(not Eigen3::eigen_dense_general<Eigen::DiagonalMatrix<double, 3>, true>);
}

