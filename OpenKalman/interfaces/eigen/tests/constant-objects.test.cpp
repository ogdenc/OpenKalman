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


TEST(eigen3, constant_objects)
{
  static_assert(not constant_matrix<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, M22>>);

  static_assert(constant_matrix<Z21, ConstantType::static_constant>);
  static_assert(scalar_constant<constant_coefficient<Z21>, ConstantType::static_constant>);
  static_assert(not scalar_constant<constant_coefficient<Z21>, ConstantType::dynamic_constant>);

  static_assert(constant_matrix<M11, ConstantType::dynamic_constant>);
  static_assert(not constant_matrix<M1x>);
  static_assert(not constant_matrix<Mx1>);
  static_assert(not constant_matrix<Mxx>);
  static_assert(not constant_matrix<M21>);
  EXPECT_EQ(constant_coefficient{make_dense_object_from<M11>(5.5)}(), 5.5);

  static_assert(constant_coefficient_v<Z21> == 0);
  static_assert(constant_coefficient_v<Z12> == 0);
  static_assert(constant_coefficient_v<Z23> == 0);
  static_assert(constant_coefficient_v<Z2x> == 0);
  static_assert(constant_coefficient_v<Zx2> == 0);

  static_assert(constant_coefficient_v<Zxx> == 0);
  static_assert(constant_coefficient_v<Zx1> == 0);
  static_assert(constant_coefficient_v<C11_1> == 1);
  static_assert(constant_coefficient_v<C11_m1> == -1);
  static_assert(constant_coefficient_v<C11_2> == 2);
  static_assert(constant_coefficient_v<C11_m2> == -2);
  static_assert(constant_coefficient_v<C11_3> == 3);
  static_assert(constant_coefficient_v<C2x_2> == 2);
  static_assert(constant_coefficient_v<Cx2_2> == 2);
  static_assert(constant_coefficient_v<Cxx_2> == 2);
  static_assert(constant_coefficient_v<B22_true> == true);
  static_assert(constant_coefficient_v<B22_false> == false);
  static_assert(not constant_matrix<Cd22_2>);
  static_assert(not constant_matrix<Cd2x_2>);
  static_assert(not constant_matrix<Cdx2_2>);
  static_assert(not constant_matrix<Cd22_3>);
  static_assert(not constant_matrix<Cd22_2>);
  static_assert(not constant_matrix<Cd2x_2>);
  static_assert(not constant_matrix<Cdx2_2>);
  static_assert(not constant_matrix<Cd22_3>);

  constant_coefficient<C21_3> c3;
  static_assert(std::decay_t<decltype(+c3)>::value == 3);
  static_assert(std::decay_t<decltype(-c3)>::value == -3);
  static_assert(std::decay_t<decltype(c3 + std::integral_constant<int, 2>{})>::value == 5);
  static_assert(std::decay_t<decltype(c3 - std::integral_constant<int, 2>{})>::value == 1);
  static_assert(std::decay_t<decltype(c3 * std::integral_constant<int, 2>{})>::value == 6);
  static_assert(std::decay_t<decltype(c3 / std::integral_constant<int, 2>{})>::value == 1.5);

  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value + c3 == 5);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value - c3 == -1);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 2>{})>::value * c3 == 6);
  static_assert(std::decay_t<decltype(std::integral_constant<int, 9>{})>::value / c3 == 3);

  constant_coefficient c3u {M21::Constant(3)};
  EXPECT_EQ(+c3u, 3);
  EXPECT_EQ(-c3u, -3);
  EXPECT_EQ((c3u + std::integral_constant<int, 2>{}), 5);
  EXPECT_EQ((c3u - std::integral_constant<int, 2>{}), 1);
  EXPECT_EQ((c3u * std::integral_constant<int, 2>{}), 6);
  EXPECT_EQ((c3u / std::integral_constant<int, 2>{}), 1.5);

  EXPECT_EQ((std::integral_constant<int, 2>{} + c3u), 5);
  EXPECT_EQ((std::integral_constant<int, 2>{} - c3u), -1);
  EXPECT_EQ((std::integral_constant<int, 2>{} * c3u), 6);
  EXPECT_EQ((std::integral_constant<int, 6>{} / c3u), 2);

  static_assert(zero<Z21>);
  static_assert(zero<Eigen::DiagonalWrapper<Z21>>);
  static_assert(zero<Z23>);
  static_assert(zero<Z2x>);
  static_assert(zero<Zx2>);
  static_assert(zero<B22_false>);
  static_assert(not zero<Cd22_2>);
  static_assert(zero<Z11>);
  static_assert(zero<Zxx>);
}

