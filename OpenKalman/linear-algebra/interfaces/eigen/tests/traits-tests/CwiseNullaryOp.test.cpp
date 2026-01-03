/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear-algebra/interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using stdex::numbers::pi;


TEST(eigen3, Eigen_CwiseNullaryOp)
{
  static_assert(eigen_matrix_general<M33::ConstantReturnType, true>);

  static_assert(constant_value_v<C11_1> == 1);
  static_assert(values::dynamic<constant_value<typename Mxx::ConstantReturnType>>);
  static_assert(not values::fixed<constant_value<typename Mxx::ConstantReturnType>>);
  static_assert(constant_value_v<Z11> == 0);
  static_assert(constant_value_v<Z22> == 0);

  EXPECT_EQ(constant_value{M22::Constant(3)}(), 3);
  EXPECT_EQ(constant_value{M2x::Constant(2, 2, 3)}(), 3);
  EXPECT_EQ(constant_value{Mx2::Constant(2, 2, 3)}(), 3);
  EXPECT_EQ(constant_value{Mxx::Constant(2, 2, 3)}(), 3);
  EXPECT_EQ(constant_value{M1x::Constant(1, 1, 3)}(), 3);
  EXPECT_EQ(constant_value{Mx1::Constant(1, 1, 3)}(), 3);
  EXPECT_EQ(constant_value{Mxx::Constant(1, 1, 3)}(), 3);

  EXPECT_EQ(constant_value{M11::Identity()}(), 1);
  static_assert(not constant_matrix<decltype(M1x::Identity(1, 1))>);
  static_assert(not constant_matrix<decltype(Mx1::Identity(1, 1))>);
  static_assert(not constant_matrix<decltype(Mxx::Identity(1, 1))>);

  EXPECT_EQ((constant_value{M22{}.NullaryExpr([]{ return 7.; })}()), 7.);
  EXPECT_EQ((constant_value{M2x{2, 2}.NullaryExpr(2, 2, []{ return 7.; })}()), 7.);
  EXPECT_EQ((constant_value{Mx2{2, 2}.NullaryExpr(2, 2, []{ return 7.; })}()), 7.);
  EXPECT_EQ((constant_value{Mxx{2, 2}.NullaryExpr(2, 2, []{ return 7.; })}()), 7.);

  static_assert(constant_value_v<I11> == 1);
  static_assert(not constant_matrix<I1x> == 1);
  static_assert(not constant_matrix<Ix1> == 1);
  static_assert(not constant_matrix<Ixx> == 1);

  static_assert(not zero<typename Mxx::ConstantReturnType>);
  static_assert(zero<Z11>);
  static_assert(zero<Z22>);

  static_assert(constant_diagonal_value_v<I11> == 1);
  static_assert(constant_diagonal_value_v<I1x> == 1);
  static_assert(constant_diagonal_value_v<Ix1> == 1);
  static_assert(constant_diagonal_value_v<Ixx> == 1);

  static_assert(constant_diagonal_value_v<C11_1> == 1);
  static_assert(constant_diagonal_value_v<I22> == 1);
  static_assert(values::dynamic<constant_diagonal_value<typename M11::ConstantReturnType>>);
  static_assert(not constant_diagonal_matrix<typename M1x::ConstantReturnType>);
  static_assert(not constant_diagonal_matrix<typename Mx1::ConstantReturnType>);
  static_assert(not constant_diagonal_matrix<typename Mxx::ConstantReturnType>);
  static_assert(constant_diagonal_value_v<Z11> == 0);
  static_assert(constant_diagonal_value_v<Z22> == 0);

  static_assert(constant_diagonal_value_v<I11> == 1);
  static_assert(constant_diagonal_value_v<I1x> == 1);
  static_assert(constant_diagonal_value_v<Ix1> == 1);
  static_assert(constant_diagonal_value_v<Ixx> == 1);

  static_assert(values::fixed<constant_diagonal_value<typename M33::IdentityReturnType>>);
  static_assert(values::fixed<constant_diagonal_value<typename M3x::IdentityReturnType>>);
  static_assert(values::fixed<constant_diagonal_value<typename Mx3::IdentityReturnType>>);
  static_assert(values::fixed<constant_diagonal_value<typename Mxx::IdentityReturnType>>);
  static_assert(constant_diagonal_matrix<typename M3x::IdentityReturnType>);
  static_assert(constant_diagonal_matrix<typename Mx3::IdentityReturnType>);
  static_assert(constant_diagonal_matrix<typename Mxx::IdentityReturnType>);

  EXPECT_EQ(constant_diagonal_value{M22::Identity()}(), 1);
  EXPECT_EQ(constant_diagonal_value{M2x::Identity(2, 2)}(), 1);
  EXPECT_EQ(constant_diagonal_value{Mx2::Identity(2, 2)}(), 1);
  EXPECT_EQ(constant_diagonal_value{Mxx::Identity(2, 2)}(), 1);
  EXPECT_EQ(constant_diagonal_value{M11::Constant(3)}(), 3);
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

  static_assert(triangular_matrix<Z22, triangle_type::lower>);

  static_assert(triangular_matrix<Z22, triangle_type::upper>);

  static_assert(square_shaped<Z11>);
  static_assert(square_shaped<C11_1>);

  static_assert(square_shaped<Z11, values::unbounded_size, applicability::permitted>);
  static_assert(square_shaped<Z2x, values::unbounded_size, applicability::permitted>);
  static_assert(not square_shaped<Z21, values::unbounded_size, applicability::permitted>);
  static_assert(square_shaped<C22_1, values::unbounded_size, applicability::permitted>);
  static_assert(not square_shaped<C21_1, values::unbounded_size, applicability::permitted>);

  static_assert(one_dimensional<Z11>);
  static_assert(one_dimensional<Z1x, values::unbounded_size, applicability::permitted>);
  static_assert(one_dimensional<Zx1, values::unbounded_size, applicability::permitted>);
  static_assert(one_dimensional<C11_1>);

  static_assert(not writable<Mx2::ConstantReturnType>);
  static_assert(not writable<M2x::IdentityReturnType>);
}

