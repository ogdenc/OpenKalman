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


TEST(eigen3, Eigen_Diagonal)
{
  static_assert(not self_contained<decltype(std::declval<M22>().diagonal())>);
  static_assert(self_contained<decltype(std::declval<C22_2>().matrix().diagonal())>);

  static_assert(constant_coefficient_v<decltype(M22::Identity().diagonal())> == 1);
  static_assert(constant_coefficient_v<decltype(M22::Identity().diagonal<1>())> == 0);
  static_assert(constant_coefficient_v<decltype(M22::Identity().diagonal<-1>())> == 0);
  static_assert(constant_matrix<decltype(M22::Identity().diagonal<Eigen::DynamicIndex>()), ConstantType::dynamic_constant>);
  static_assert(not constant_matrix<decltype(M22::Identity().diagonal<2>())>);

  static_assert(constant_coefficient_v<decltype(M2x::Identity().diagonal())> == 1);
  static_assert(constant_coefficient_v<decltype(M2x::Identity().diagonal<-1>())> == 0);
  static_assert(constant_coefficient_v<decltype(M2x::Identity().diagonal<1>())> == 0); // may throw out-of-range exception at runtime
  static_assert(not constant_matrix<decltype(M2x::Identity().diagonal<-2>())>);

  static_assert(constant_coefficient_v<decltype(Mx2::Identity().diagonal())> == 1);
  static_assert(constant_coefficient_v<decltype(Mx2::Identity().diagonal<1>())> == 0);
  static_assert(constant_coefficient_v<decltype(Mx2::Identity().diagonal<-1>())> == 0); // may throw out-of-range exception at runtime
  static_assert(not constant_matrix<decltype(Mx2::Identity().diagonal<2>())>);

  static_assert(constant_coefficient_v<decltype(Mxx::Identity().diagonal())> == 1);
  static_assert(constant_coefficient_v<decltype(Mxx::Identity().diagonal<1>())> == 0); // may throw out-of-range exception at runtime
  static_assert(constant_coefficient_v<decltype(Mxx::Identity().diagonal<-1>())> == 0); // may throw out-of-range exception at runtime

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<1>())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<-1>())> == 2);

  static_assert(constant_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal<1>())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal<-1>())> == 0);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix().diagonal())> == 2);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<C1x_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cx1_2>().matrix().diagonal())>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C12_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().matrix().diagonal())> == 2);

  static_assert(not constant_diagonal_matrix<decltype(std::declval<C11_2>().matrix().diagonal<1>())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<C11_2>().matrix().diagonal<-1>())>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<1>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<-1>())> == 2);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<C22_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<C22_2>().matrix().diagonal<2>())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<C22_2>().matrix().diagonal<-2>())>);

  static_assert(zero<decltype(std::declval<Z23>().matrix().diagonal())>);
  static_assert(zero<decltype(std::declval<Z23>().matrix().diagonal<1>())>);
  static_assert(zero<decltype(std::declval<Z23>().matrix().diagonal<-1>())>);
  static_assert(zero<decltype(std::declval<Z23>().matrix().diagonal(1))>);
  static_assert(zero<decltype(std::declval<Z23>().matrix().diagonal(-1))>);
  static_assert(zero<decltype(std::declval<Zxx>().matrix().diagonal(2))>);
  static_assert(zero<decltype(std::declval<Zxx>().matrix().diagonal(-2))>);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal<-1>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal<1>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd2x_2>().matrix().diagonal<-1>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd2x_2>().matrix().diagonal<1>())> == 0);

  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd2x_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cdx2_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cdxx_2>().matrix().diagonal())>);

  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, -2>, 0, 0>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, -1>, 0, 1>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, 0>, 0, 2>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, 1>, 0, 2>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, 2>, 0, 1>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, 3>, 0, 0>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M23, Eigen::DynamicIndex>, 0, dynamic_size>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, 2>, 0, 0>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, 1>, 0, 1>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, 0>, 0, 2>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, -1>, 0, 2>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, -2>, 0, 1>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, -3>, 0, 0>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M32, Eigen::DynamicIndex>, 0, dynamic_size>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<M2x, 0>, 0, dynamic_size>);
  static_assert(dimension_size_of_index_is<Eigen::Diagonal<Mx2, 0>, 0, dynamic_size>);

  static_assert(one_dimensional<Eigen::Diagonal<M11, 0>>);
  static_assert(one_dimensional<Eigen::Diagonal<M13, 0>>);
  static_assert(one_dimensional<Eigen::Diagonal<M13, 1>>);
  static_assert(not one_dimensional<Eigen::Diagonal<M1x, 0>>);
  static_assert(one_dimensional<Eigen::Diagonal<M1x, 0>, Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<Eigen::Diagonal<M31, 0>>);
  static_assert(one_dimensional<Eigen::Diagonal<M31, -1>>);
  static_assert(not one_dimensional<Eigen::Diagonal<Mx1, 0>>);
  static_assert(one_dimensional<Eigen::Diagonal<Mx1, 0>, Qualification::depends_on_dynamic_shape>);
  static_assert(not one_dimensional<Eigen::Diagonal<M22, Eigen::DynamicIndex>>);
  static_assert(one_dimensional<Eigen::Diagonal<M22, Eigen::Dynamic>, Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<Eigen::Diagonal<M2x, 0>, Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<Eigen::Diagonal<Mx2, 0>, Qualification::depends_on_dynamic_shape>);
}

