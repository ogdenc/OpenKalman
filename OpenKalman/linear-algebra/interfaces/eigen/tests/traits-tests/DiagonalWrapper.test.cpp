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


TEST(eigen3, Eigen_DiagonalWrapper)
{
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M31>, 0> == 3);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M3x>, 0> == stdex::dynamic_extent);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<Mx1>, 0> == stdex::dynamic_extent);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<Mxx>, 0> == stdex::dynamic_extent);

  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M13>, 0> == 3);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M1x>, 0> == stdex::dynamic_extent);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<Mx3>, 0> == stdex::dynamic_extent);

  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M31>, 1> == 3);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M3x>, 1> == stdex::dynamic_extent);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<Mx1>, 1> == stdex::dynamic_extent);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<Mxx>, 1> == stdex::dynamic_extent);

  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M13>, 1> == 3);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<M1x>, 1> == stdex::dynamic_extent);
  static_assert(index_dimension_of_v<Eigen::DiagonalWrapper<Mx3>, 1> == stdex::dynamic_extent);

  static_assert(patterns::dimension_of_v<decltype(std::get<0>(all_vector_space_descriptors(std::declval<Eigen::DiagonalWrapper<M31>>())))> == 3);
  static_assert(patterns::dimension_of_v<decltype(std::get<1>(all_vector_space_descriptors(std::declval<Eigen::DiagonalWrapper<M31>>())))> == 3);

  static_assert(square_shaped<Eigen::DiagonalWrapper<Mxx>>);

  static_assert(std::is_same_v<nested_object_of_t<Eigen::DiagonalWrapper<M21>>, M21&>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M31>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M22>>, M44>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M3x>>, Mxx>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<Mx1>>, Mxx>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<Mxx>>, Mxx>);

  static_assert(constant_value_v<decltype(std::declval<C11_2>().matrix().asDiagonal())> == 2);
  static_assert(constant_value_v<decltype(std::declval<Z21>().matrix().asDiagonal())> == 0);
  static_assert(not constant_matrix<decltype(std::declval<C21_2>().matrix().asDiagonal())>);

  static_assert(constant_diagonal_value_v<decltype(std::declval<C11_2>().matrix().asDiagonal())> == 2);
  static_assert(constant_diagonal_value_v<decltype(std::declval<Z21>().matrix().asDiagonal())> == 0);
  static_assert(constant_diagonal_value_v<decltype(std::declval<C21_2>().matrix().asDiagonal())> == 2);

  static_assert(not zero<decltype(std::declval<C11_1>())>);
  static_assert(not zero<decltype(std::declval<C11_2>())>);
  static_assert(not zero<decltype(std::declval<C21_2>())>);

  static_assert(zero<decltype(std::declval<Z11>().matrix().asDiagonal())>);
  static_assert(zero<decltype(std::declval<Z21>().matrix().asDiagonal())>);

  static_assert(identity_matrix<decltype(std::declval<C11_1>().matrix().asDiagonal())>);
  static_assert(identity_matrix<decltype(std::declval<C21_1>().matrix().asDiagonal())>);

  static_assert(diagonal_matrix<Eigen::DiagonalWrapper<M31>>);
  static_assert(diagonal_matrix<decltype(std::declval<C11_2>().matrix().asDiagonal())>);
  static_assert(diagonal_matrix<decltype(std::declval<C21_2>().matrix().asDiagonal())>);

  static_assert(not writable<Eigen::DiagonalWrapper<M31>>);
}

