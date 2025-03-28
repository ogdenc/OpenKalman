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


TEST(eigen3, Eigen_Replicate)
{
  auto z11 = I11{1,1} - I11{1,1};
  auto z2x_1 = Eigen::Replicate<Z11, 2, Eigen::Dynamic> {z11, 2, 1};
  auto zx1_2 = Eigen::Replicate<Z11, Eigen::Dynamic, 1> {z11, 2, 1};
  auto zxx_21 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 1};
  auto z22 = Eigen::Replicate<Z11, 2, 2> {z11};
  auto z2x_2 = Eigen::Replicate<Z11, 2, Eigen::Dynamic> {z11, 2, 2};
  auto zx2_2 = Eigen::Replicate<Z11, Eigen::Dynamic, 2> {z11, 2, 2};
  auto zxx_22 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  static_assert(Eigen3::eigen_general<Zxx, true>);
  static_assert(index_count_v<Zxx> == 2);
  static_assert(index_dimension_of_v<Zxx, 0> == dynamic_size);
  static_assert(index_dimension_of_v<Zxx, 1> == dynamic_size);
  EXPECT_EQ(get_vector_space_descriptor<0>(zxx_21), 2);
  EXPECT_EQ(get_vector_space_descriptor<1>(zxx_21), 1);
  static_assert(std::is_same_v<typename interface::indexible_object_traits<std::decay_t<Zxx>>::scalar_type, double>);

  static_assert(one_dimensional<Eigen::Replicate<M11, 1, 1>>);
  static_assert(one_dimensional<Eigen::Replicate<Mxx, 1, 1>, Applicability::permitted>);
  static_assert(one_dimensional<Eigen::Replicate<M1x, 1, 1>, Applicability::permitted>);
  static_assert(one_dimensional<Eigen::Replicate<Mx1, 1, 1>, Applicability::permitted>);
  static_assert(not one_dimensional<Eigen::Replicate<M2x, Eigen::Dynamic, Eigen::Dynamic>, Applicability::permitted>);
  static_assert(not one_dimensional<Eigen::Replicate<Mx2, Eigen::Dynamic, Eigen::Dynamic>, Applicability::permitted>);
  static_assert(not one_dimensional<Eigen::Replicate<Mxx, 1, 1>>);
  static_assert(one_dimensional<Eigen::Replicate<M11, Eigen::Dynamic, Eigen::Dynamic>, Applicability::permitted>);
  static_assert(not one_dimensional<Eigen::Replicate<M11, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(square_shaped<Eigen::Replicate<M22, 3, 3>>);
  static_assert(square_shaped<Eigen::Replicate<M22, Eigen::Dynamic, 3>, Applicability::permitted>);
  static_assert(not square_shaped<Eigen::Replicate<M22, Eigen::Dynamic, 3>>);
  static_assert(square_shaped<Eigen::Replicate<M22, 3, Eigen::Dynamic>, Applicability::permitted>);
  static_assert(not square_shaped<Eigen::Replicate<M22, 3, Eigen::Dynamic>>);
  static_assert(square_shaped<Eigen::Replicate<M32, 2, 3>>);
  static_assert(square_shaped<Eigen::Replicate<M32, Eigen::Dynamic, 3>, Applicability::permitted>);
  static_assert(not square_shaped<Eigen::Replicate<M32, Eigen::Dynamic, 3>>);
  static_assert(square_shaped<Eigen::Replicate<M32, 2, Eigen::Dynamic>, Applicability::permitted>);
  static_assert(not square_shaped<Eigen::Replicate<M32, 2, Eigen::Dynamic>>);
  static_assert(not square_shaped<Eigen::Replicate<M32, 5, Eigen::Dynamic>, Applicability::permitted>);
  static_assert(not square_shaped<Eigen::Replicate<M32, Eigen::Dynamic, 2>, Applicability::permitted>);
  static_assert(square_shaped<Eigen::Replicate<M3x, 2, 3>, Applicability::permitted>);
  static_assert(not square_shaped<Eigen::Replicate<M3x, 2, 3>>);
  static_assert(square_shaped<Eigen::Replicate<Mx2, 2, 3>, Applicability::permitted>);
  static_assert(not square_shaped<Eigen::Replicate<Mx2, 2, 3>>);
  static_assert(not square_shaped<Eigen::Replicate<M2x, 2, 3>, Applicability::permitted>);
  static_assert(not square_shaped<Eigen::Replicate<Mx3, 2, 3>, Applicability::permitted>);

  static_assert(constant_coefficient_v<Eigen::Replicate<Z11, 1, 2>> == 0);
  static_assert(constant_coefficient_v<decltype(z2x_1)> == 0);
  static_assert(constant_coefficient_v<decltype(zx1_2)> == 0);
  static_assert(constant_coefficient_v<Eigen::Replicate<C2x_2, 1, 2>> == 2);
  static_assert(constant_coefficient_v<Eigen::Replicate<Cx2_2, 1, 2>> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().replicate<5,5>())> == 2);
  static_assert(not constant_matrix<Eigen::Replicate<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().replicate<1,1>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().replicate<5,5>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(z22)> == 0);
  static_assert(constant_diagonal_matrix<decltype(z2x_2)>);
  static_assert(constant_diagonal_matrix<decltype(zx2_2)>);
  static_assert(constant_diagonal_matrix<decltype(zxx_22)>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().replicate<1,1>())> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Replicate<C22_2, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(identity_matrix<Eigen::Replicate<I22, 1, 1>>);

  static_assert(diagonal_matrix<decltype(std::declval<DW21>().replicate<1, 1>())>);
  static_assert(diagonal_matrix<decltype(z11.replicate<2, 2>())>);
  static_assert(diagonal_matrix<decltype(z11.replicate<2, Eigen::Dynamic>())>);
  static_assert(diagonal_matrix<decltype(z11.replicate<Eigen::Dynamic, 2>())>);
  static_assert(diagonal_matrix<decltype(z11.replicate<Eigen::Dynamic, Eigen::Dynamic>())>);

  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().replicate<1, 1>()), TriangleType::lower>);

  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().replicate<1, 1>()), TriangleType::upper>);

  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().replicate<1, 1>())>);

  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().replicate<1, 1>())>);
}

