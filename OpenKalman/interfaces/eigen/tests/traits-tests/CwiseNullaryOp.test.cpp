/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using numbers::pi;


TEST(eigen3, Eigen_CwiseNullaryOp)
{
  static_assert(eigen_matrix_general<M33::ConstantReturnType, true>);

  static_assert(constant_coefficient_v<C11_1> == 1);
  static_assert(constant_matrix<typename Mxx::ConstantReturnType, ConstantType::dynamic_constant>);
  static_assert(not constant_matrix<typename Mxx::ConstantReturnType, ConstantType::static_constant>);
  static_assert(constant_coefficient_v<Z11> == 0);
  static_assert(constant_coefficient_v<Z22> == 0);

  EXPECT_EQ(constant_coefficient{M22::Constant(3)}(), 3);
  EXPECT_EQ(constant_coefficient{M2x::Constant(2, 2, 3)}(), 3);
  EXPECT_EQ(constant_coefficient{Mx2::Constant(2, 2, 3)}(), 3);
  EXPECT_EQ(constant_coefficient{Mxx::Constant(2, 2, 3)}(), 3);
  EXPECT_EQ(constant_coefficient{M1x::Constant(1, 1, 3)}(), 3);
  EXPECT_EQ(constant_coefficient{Mx1::Constant(1, 1, 3)}(), 3);
  EXPECT_EQ(constant_coefficient{Mxx::Constant(1, 1, 3)}(), 3);

  EXPECT_EQ(constant_coefficient{M11::Identity()}(), 1);
  static_assert(not constant_matrix<decltype(M1x::Identity(1, 1))>);
  static_assert(not constant_matrix<decltype(Mx1::Identity(1, 1))>);
  static_assert(not constant_matrix<decltype(Mxx::Identity(1, 1))>);

  EXPECT_EQ((constant_coefficient{M22{}.NullaryExpr([]{ return 7.; })}()), 7.);
  EXPECT_EQ((constant_coefficient{M2x{2, 2}.NullaryExpr(2, 2, []{ return 7.; })}()), 7.);
  EXPECT_EQ((constant_coefficient{Mx2{2, 2}.NullaryExpr(2, 2, []{ return 7.; })}()), 7.);
  EXPECT_EQ((constant_coefficient{Mxx{2, 2}.NullaryExpr(2, 2, []{ return 7.; })}()), 7.);

  static_assert(constant_coefficient_v<I11> == 1);
  static_assert(not constant_matrix<I1x> == 1);
  static_assert(not constant_matrix<Ix1> == 1);
  static_assert(not constant_matrix<Ixx> == 1);

  static_assert(not zero<typename Mxx::ConstantReturnType>);
  static_assert(zero<Z11>);
  static_assert(zero<Z22>);

  static_assert(constant_diagonal_coefficient_v<I11> == 1);
  static_assert(constant_diagonal_coefficient_v<I1x> == 1);
  static_assert(constant_diagonal_coefficient_v<Ix1> == 1);
  static_assert(constant_diagonal_coefficient_v<Ixx> == 1);

  static_assert(constant_diagonal_coefficient_v<C11_1> == 1);
  static_assert(constant_diagonal_coefficient_v<I22> == 1);
  static_assert(constant_diagonal_matrix<typename M11::ConstantReturnType, ConstantType::dynamic_constant>);
  static_assert(not constant_diagonal_matrix<typename M1x::ConstantReturnType>);
  static_assert(not constant_diagonal_matrix<typename Mx1::ConstantReturnType>);
  static_assert(not constant_diagonal_matrix<typename Mxx::ConstantReturnType>);
  static_assert(constant_diagonal_coefficient_v<Z11> == 0);
  static_assert(constant_diagonal_coefficient_v<Z22> == 0);

  static_assert(constant_diagonal_coefficient_v<I11> == 1);
  static_assert(constant_diagonal_coefficient_v<I1x> == 1);
  static_assert(constant_diagonal_coefficient_v<Ix1> == 1);
  static_assert(constant_diagonal_coefficient_v<Ixx> == 1);

  static_assert(constant_diagonal_matrix<typename M33::IdentityReturnType, ConstantType::static_constant>);
  static_assert(constant_diagonal_matrix<typename M3x::IdentityReturnType, ConstantType::static_constant>);
  static_assert(constant_diagonal_matrix<typename Mx3::IdentityReturnType, ConstantType::static_constant>);
  static_assert(constant_diagonal_matrix<typename Mxx::IdentityReturnType, ConstantType::static_constant>);
  static_assert(constant_diagonal_matrix<typename M3x::IdentityReturnType>);
  static_assert(constant_diagonal_matrix<typename Mx3::IdentityReturnType>);
  static_assert(constant_diagonal_matrix<typename Mxx::IdentityReturnType>);

  EXPECT_EQ(constant_diagonal_coefficient{M22::Identity()}(), 1);
  EXPECT_EQ(constant_diagonal_coefficient{M2x::Identity(2, 2)}(), 1);
  EXPECT_EQ(constant_diagonal_coefficient{Mx2::Identity(2, 2)}(), 1);
  EXPECT_EQ(constant_diagonal_coefficient{Mxx::Identity(2, 2)}(), 1);
  EXPECT_EQ(constant_diagonal_coefficient{M11::Constant(3)}(), 3);
  static_assert(not constant_diagonal_matrix<decltype(M1x::Constant(1, 1, 3))>);
  static_assert(not constant_diagonal_matrix<decltype(Mx1::Constant(1, 1, 3))>);
  static_assert(not constant_diagonal_matrix<decltype(Mxx::Constant(1, 1, 3))>);

  static_assert(identity_matrix<typename M33::IdentityReturnType>);
  static_assert(identity_matrix<typename M3x::IdentityReturnType>);
  static_assert(identity_matrix<typename Mx3::IdentityReturnType>);
  static_assert(identity_matrix<typename Mxx::IdentityReturnType>);
  static_assert(identity_matrix<typename M3x::IdentityReturnType>);
  static_assert(identity_matrix<typename Mx3::IdentityReturnType>);
  static_assert(identity_matrix<typename Mxx::IdentityReturnType>);
  static_assert(identity_matrix<C11_1>);
  static_assert(not identity_matrix<Z22>);

  static_assert(diagonal_matrix<typename M33::IdentityReturnType>);

  static_assert(hermitian_matrix<M33::ConstantReturnType>);

  static_assert(hermitian_matrix<M33::ConstantReturnType>);
  static_assert(not hermitian_matrix<M21::ConstantReturnType>);
  static_assert(hermitian_matrix<typename M33::IdentityReturnType>);
  static_assert(hermitian_matrix<Z22>);
  static_assert(hermitian_matrix<C11_2>);

  static_assert(triangular_matrix<Z22, TriangleType::lower>);

  static_assert(triangular_matrix<Z22, TriangleType::upper>);

  static_assert(square_shaped<Z11>);
  static_assert(square_shaped<C11_1>);

  static_assert(square_shaped<Z11, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<Z2x, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<Z21, Qualification::depends_on_dynamic_shape>);
  static_assert(square_shaped<C22_1, Qualification::depends_on_dynamic_shape>);
  static_assert(not square_shaped<C21_1, Qualification::depends_on_dynamic_shape>);

  static_assert(one_dimensional<Z11>);
  static_assert(one_dimensional<Z1x, Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<Zx1, Qualification::depends_on_dynamic_shape>);
  static_assert(one_dimensional<C11_1>);

  static_assert(not writable<Mx2::ConstantReturnType>);
  static_assert(not writable<M2x::IdentityReturnType>);
}

