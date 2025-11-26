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
  static_assert(values::dynamic<constant_diagonal_value<M11>>);
  static_assert(not constant_diagonal_matrix<M1x>);
  static_assert(not constant_diagonal_matrix<Mx1>);
  static_assert(not constant_diagonal_matrix<Mxx>);
  static_assert(not constant_diagonal_matrix<M21>);
  EXPECT_EQ(constant_diagonal_value{make_dense_object_from<M11>(5.5)}(), 5.5);

  static_assert(values::fixed<constant_diagonal_value<Z11>>);
  static_assert(values::fixed<constant_diagonal_value<Z1x>>);
  static_assert(values::fixed<constant_diagonal_value<Z2x>>);
  static_assert(values::fixed<constant_diagonal_value<Zx2>>);
  static_assert(values::fixed<constant_diagonal_value<Zx1>>);
  static_assert(values::fixed<constant_diagonal_value<Zxx>>);
  static_assert(values::fixed<constant_diagonal_value<Z21>>);
  static_assert(values::fixed<constant_diagonal_value<Z12>>);
  static_assert(values::fixed<constant_diagonal_value<Z23>>);
  static_assert(constant_diagonal_value_v<C11_1> == 1);
  static_assert(constant_diagonal_value_v<C11_m1> == -1);
  static_assert(constant_diagonal_value_v<C11_2> == 2);
  static_assert(constant_diagonal_value_v<C11_m2> == -2);
  static_assert(values::fixed<constant_diagonal_value<C11_1>>);
  static_assert(not constant_diagonal_matrix<C21_1>);
  static_assert(not constant_diagonal_matrix<C2x_1>);
  static_assert(not constant_diagonal_matrix<C1x_1>);
  static_assert(not constant_diagonal_matrix<Cx1_1>);
  static_assert(not constant_diagonal_matrix<Cxx_1>);
  static_assert(constant_diagonal_value_v<I22> == 1);
  static_assert(constant_diagonal_value_v<I2x> == 1);
  static_assert(constant_diagonal_value_v<Cd22_2> == 2);
  static_assert(constant_diagonal_value_v<Cd2x_2> == 2);
  static_assert(constant_diagonal_value_v<Cdx2_2> == 2);
  static_assert(constant_diagonal_value_v<Cdxx_2> == 2);

  constant_diagonal_value<Cd22_3> cd3;
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

  constant_diagonal_value cd3u {Eigen::DiagonalWrapper {M21::Constant(3)}};
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

  constant_value<C21_3> c3;
  auto sc3 = values::fixed_value<double, 3>{};
  auto sco3 = values::operation(std::minus{}, values::fixed_value<double, 7>{}, std::integral_constant<int, 4>{});
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

