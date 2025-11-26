/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear-algebra/interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, Eigen_CwiseUnaryView)
{
  auto cxa = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{1, 2}}; // Constant complex
  auto cxb = Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{3, 4}}; // Constant complex

  // scalar_real_ref_op
  static_assert(OpenKalman::Eigen3::constexpr_unary_operation_defined<Eigen::internal::scalar_real_ref_op<double>>);
  static_assert(Eigen3::UnaryFunctorTraits<Eigen::internal::scalar_real_ref_op<double>>::constexpr_operation()(5) == 5);
  static_assert(constant_value_v<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<double>, decltype(M11::Identity())>> == 1);
  static_assert(constant_value_v<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<double>, Z11>> == 0);
  static_assert(constant_value_v<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<double>, C11_2>> == 2);
  static_assert(constant_value_v<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<double>, C22_2>> == 2);
  static_assert(constant_matrix<decltype(cxa.real())>);
  EXPECT_EQ(constant_value{cxa.real()}(), 1);
  static_assert(constant_matrix<decltype(cxb.real())>);
  EXPECT_EQ(constant_value{cxb.real()}(), 3);
  static_assert(constant_diagonal_value_v<decltype(std::declval<Cd22_2>().real())> == 2);
  static_assert(triangular_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<double>, Tlv22>, triangle_type::lower>);
  static_assert(triangular_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<double>, Tuv22>, triangle_type::upper>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<cdouble>, decltype(eigen_matrix_t<cdouble, 2, 2>::Identity())>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<double>, C11_m1>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<double>, Z22>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<double>, Cd22_2>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<double>, Cd22_2>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<double>, Salv22>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<double>, Sauv22>>);
  static_assert(hermitian_matrix<decltype(cxa.real())>);

  // scalar_imag_ref_op
  static_assert(constant_matrix<decltype(cxa.imag())>);
  EXPECT_EQ(constant_value{cxa.imag()}(), 2);
  static_assert(constant_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<cdouble>, decltype(cxb)>>);
  EXPECT_EQ(constant_value{cxb.imag()}(), 4);
  static_assert(OpenKalman::Eigen3::constexpr_unary_operation_defined<Eigen::internal::scalar_imag_ref_op<double>>);
  static_assert(Eigen3::UnaryFunctorTraits<Eigen::internal::scalar_imag_ref_op<double>>::constexpr_operation()(5) == 0);
  static_assert(constant_value_v<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<double>, decltype(M11::Identity())>> == 0);
  static_assert(constant_value_v<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<double>, Z11>> == 0);
  static_assert(constant_value_v<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<double>, C11_2>> == 0);
  static_assert(constant_value_v<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<double>, C22_2>> == 0);
  static_assert(constant_diagonal_value_v<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<double>, Cd22_2>> == 0);
  static_assert(constant_diagonal_value_v<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<double>, C22_2>> == 0);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().imag()), triangle_type::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().imag()), triangle_type::upper>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<cdouble>, decltype(eigen_matrix_t<cdouble, 2, 2>::Identity().imag())>>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().imag())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().imag())>);
  static_assert(hermitian_matrix<decltype(cxa.imag())>);

}

