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


TEST(eigen3, Eigen_CwiseUnaryView)
{
  auto id = I22 {2, 2}; // Identity
  auto z = id - id; // Zero
  auto cp2 = (I11 {1, 1} + I11 {1, 1}).replicate(2, 2); // Constant +2
  auto cm2 = (-(I11 {1, 1} + I11 {1, 1})).replicate<2, 2>(); // Constant -2
  auto cxa = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{1, 2}}; // Constant complex
  auto cxb = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{3, 4}}; // Constant complex
  auto cdp2 = id * 2; // Constant diagonal +2
  auto cdm2 = id * -2; // Constant diagonal -2

  static_assert(self_contained<const M22>);

  // scalar_real_ref_op
  static_assert(not self_contained<decltype(std::declval<CA22>().real())>);
  static_assert(not self_contained<decltype(std::declval<C22_2>().real())>);
  static_assert(constant_coefficient_v<decltype(M11::Identity().real())> == 1);
  static_assert(constant_coefficient_v<decltype(std::declval<Z11>().real())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().real())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().real())> == 2);
  static_assert(constant_matrix<decltype(cxa.real())>);
  EXPECT_EQ(constant_coefficient{cxa.real()}(), 1);
  static_assert(constant_matrix<decltype(cxb.real())>);
  EXPECT_EQ(constant_coefficient{cxb.real()}(), 3);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().real())> == 2);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().real()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().real()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(eigen_matrix_t<cdouble, 2, 2>::Identity().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<C11_m1>().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<Z22>().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd22_2>().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd22_2>().real())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().real())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().real())>);
  static_assert(hermitian_matrix<decltype(cxa.real())>);

  // scalar_imag_ref_op
  static_assert(not self_contained<decltype(std::declval<CA22>().imag())>);
  static_assert(constant_coefficient_v<decltype(M11::Identity().imag())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z11>().imag())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().imag())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().imag())> == 0);
  static_assert(constant_matrix<decltype(cxa.imag())>);
  EXPECT_EQ(constant_coefficient{cxa.imag()}(), 2);
  static_assert(constant_matrix<decltype(cxb.imag())>);
  EXPECT_EQ(constant_coefficient{cxb.imag()}(), 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().imag())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().imag())> == 0);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().imag()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().imag()), TriangleType::upper>);
  static_assert(diagonal_matrix<decltype(eigen_matrix_t<cdouble, 2, 2>::Identity().imag())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().imag())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().imag())>);
  static_assert(hermitian_matrix<decltype(cxa.imag())>);

}

