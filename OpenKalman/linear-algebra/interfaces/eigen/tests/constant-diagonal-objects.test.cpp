/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;


TEST(eigen3, constant_diagonal_objects)
{
  static_assert(constant_diagonal_matrix<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, M22>>);
  static_assert(value::dynamic<constant_diagonal_coefficient<M11>>);
  static_assert(not constant_diagonal_matrix<M1x>);
  static_assert(not constant_diagonal_matrix<Mx1>);
  static_assert(not constant_diagonal_matrix<Mxx>);
  static_assert(not constant_diagonal_matrix<M21>);
  EXPECT_EQ(constant_diagonal_coefficient{make_dense_object_from<M11>(5.5)}(), 5.5);

  static_assert(value::fixed<constant_diagonal_coefficient<Z11>>);
  static_assert(value::fixed<constant_diagonal_coefficient<Z1x>>);
  static_assert(value::fixed<constant_diagonal_coefficient<Z2x>>);
  static_assert(value::fixed<constant_diagonal_coefficient<Zx2>>);
  static_assert(value::fixed<constant_diagonal_coefficient<Zx1>>);
  static_assert(value::fixed<constant_diagonal_coefficient<Zxx>>);
  static_assert(value::fixed<constant_diagonal_coefficient<Z21>>);
  static_assert(value::fixed<constant_diagonal_coefficient<Z12>>);
  static_assert(value::fixed<constant_diagonal_coefficient<Z23>>);
  static_assert(constant_diagonal_coefficient_v<C11_1> == 1);
  static_assert(constant_diagonal_coefficient_v<C11_m1> == -1);
  static_assert(constant_diagonal_coefficient_v<C11_2> == 2);
  static_assert(constant_diagonal_coefficient_v<C11_m2> == -2);
  static_assert(value::fixed<constant_diagonal_coefficient<C11_1>>);
  static_assert(not constant_diagonal_matrix<C21_1>);
  static_assert(not constant_diagonal_matrix<C2x_1>);
  static_assert(not constant_diagonal_matrix<C1x_1>);
  static_assert(not constant_diagonal_matrix<Cx1_1>);
  static_assert(not constant_diagonal_matrix<Cxx_1>);
  static_assert(constant_diagonal_coefficient_v<I22> == 1);
  static_assert(constant_diagonal_coefficient_v<I2x> == 1);
  static_assert(constant_diagonal_coefficient_v<Cd22_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cd2x_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cdx2_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cdxx_2> == 2);

  constant_diagonal_coefficient<Cd22_3> cd3;
  static_assert(std::decay_t<decltype(+cd3)>::value == 3);
  static_assert(std::decay_t<decltype(-cd3)>::value == -3);
  static_assert(std::decay_t<decltype(cd3 + std::integral_constant<int, 2>{})>::value == 5);
  static_assert(std::decay_t<decltype(cd3 - std::integral_constant<int, 2>{})>::value == 1);
  static_assert(std::decay_t<decltype(cd3 * std::integral_constant<int, 2>{})>::value == 6);
  static_assert(std::decay_t<decltype(cd3 / std::integral_constant<int, 2>{})>::value == 1.5);

  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value + cd3 == 5);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value - cd3 == -1);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value * cd3 == 6);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 9>{})>::value / cd3 == 3);

  constant_diagonal_coefficient cd3u {Eigen::DiagonalWrapper {M21::Constant(3)}};
  EXPECT_EQ(+cd3u, 3);
  EXPECT_EQ(-cd3u, -3);
  EXPECT_EQ((cd3u + std::integral_constant<int, 2>{}), 5);
  EXPECT_EQ((cd3u - std::integral_constant<int, 2>{}), 1);
  EXPECT_EQ((cd3u * std::integral_constant<int, 2>{}), 6);
  EXPECT_EQ((cd3u / std::integral_constant<int, 2>{}), 1.5);

  EXPECT_EQ((std::integral_constant<int, 2>{} + cd3u), 5);
  EXPECT_EQ((std::integral_constant<int, 2>{} - cd3u), -1);
  EXPECT_EQ((std::integral_constant<int, 2>{} * cd3u), 6);
  EXPECT_EQ((std::integral_constant<int, 6>{} / cd3u), 2);

  constant_coefficient<C21_3> c3;
  auto sc3 = value::Fixed<double, 3>{};
  auto sco3 = value::operation{std::minus{}, value::Fixed<double, 7>{}, std::integral_constant<int, 4>{}};
  static_assert(std::decay_t<decltype(c3 + cd3)>::value == 6);
  static_assert(std::decay_t<decltype(sc3 - cd3)>::value == 0);
  static_assert(std::decay_t<decltype(c3 * sco3)>::value == 9);
  static_assert(std::decay_t<decltype(sco3 / sc3)>::value == 1);

  static_assert(not identity_matrix<C21_1>);
  static_assert(not identity_matrix<C2x_1>);
  static_assert(identity_matrix<I22>);
  static_assert(identity_matrix<I2x>);
  static_assert(not identity_matrix<Cd22_2>);
  static_assert(not identity_matrix<Cd22_3>);
  static_assert(identity_matrix<C11_1>);
  static_assert(not identity_matrix<C1x_1>);
  static_assert(not identity_matrix<Cx1_1>);
  static_assert(not identity_matrix<Cxx_1>);
  static_assert(not identity_matrix<C21_1>);
  static_assert(not identity_matrix<C2x_1>);
  static_assert(not identity_matrix<C1x_1>);
  static_assert(not identity_matrix<Cx1_1>);
  static_assert(not identity_matrix<Cxx_1>);

  static_assert(identity_matrix<M00>);
  static_assert(identity_matrix<M0x>);
  static_assert(identity_matrix<Mx0>);
  static_assert(not identity_matrix<Mxx>);

}

