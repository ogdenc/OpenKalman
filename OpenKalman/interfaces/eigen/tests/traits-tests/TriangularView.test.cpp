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


TEST(eigen3, Eigen_TriangularView)
{
  static_assert(std::is_same_v<nested_object_of_t<Eigen::TriangularView<M22, Eigen::Upper>>, M22&>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<M33, Eigen::Upper>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<M3x, Eigen::Upper>>, M3x>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<Mx3, Eigen::Upper>>, Mx3>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<Mxx, Eigen::Upper>>, Mxx>);

  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<C11_2>, Eigen::Lower>> == 2);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Lower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z23>, Eigen::Lower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Zxx>, Eigen::Upper>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z23>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<decltype(M32::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd22_2>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd2x_2>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cdx2_2>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cdxx_2>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<M11, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<M11, Eigen::UnitLower>> == 1);

  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd22_2>, Eigen::Lower>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z2x>, Eigen::StrictlyLower>> == 0);
  static_assert(not constant_diagonal_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Z23>, Eigen::StrictlyLower>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Zx2>, Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M2x::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(Mx2::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(Mxx::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(not constant_diagonal_matrix<Eigen::TriangularView<decltype(M23::Identity()), Eigen::StrictlyLower>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M2x::Identity()), Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(Mx2::Identity()), Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(Mxx::Identity()), Eigen::UnitLower>> == 1);
  static_assert(not constant_diagonal_matrix<Eigen::TriangularView<decltype(M23::Identity()), Eigen::UnitUpper>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Tlv22, Eigen::UnitUpper>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Tuv22, Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<M11, Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<M11, Eigen::UnitLower>> == 1);

  static_assert(zero<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Lower>>);

  static_assert(identity_matrix<Eigen::TriangularView<decltype(M33::Identity()), Eigen::Upper>>);
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(M3x::Identity(3, 3)), Eigen::Lower>>);
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(Mx3::Identity(3, 3)), Eigen::Upper>>);
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(Mxx::Identity(3, 3)), Eigen::Lower>>);
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(M33::Identity()), Eigen::StrictlyUpper>>);
  static_assert(identity_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::UnitUpper>>);

  static_assert(triangular_matrix<Eigen::TriangularView<M33, Eigen::Lower>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::TriangularView<M3x, Eigen::Lower>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::TriangularView<Mx3, Eigen::Lower>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::TriangularView<Mxx, Eigen::Lower>, TriangleType::lower>);
  static_assert(triangular_matrix<Tlvx2, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::TriangularView<M43, Eigen::Lower>, TriangleType::lower>);

  static_assert(triangular_matrix<Eigen::TriangularView<M33, Eigen::Upper>, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::TriangularView<M3x, Eigen::Upper>, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::TriangularView<Mx3, Eigen::Upper>, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::TriangularView<Mxx, Eigen::Upper>, TriangleType::upper>);
  static_assert(triangular_matrix<Tuv2x, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::TriangularView<M34, Eigen::Upper>, TriangleType::upper>);

  static_assert(triangular_matrix<Eigen::TriangularView<Tlv22, Eigen::Upper>, TriangleType::diagonal>);
  static_assert(triangular_matrix<Eigen::TriangularView<Tuv22, Eigen::Lower>, TriangleType::diagonal>);

  static_assert(diagonal_matrix<Eigen::TriangularView<Tlv22, Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tlv2x, Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tlvx2, Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tlvxx, Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tuv22, Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tuv2x, Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tuvx2, Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tuvxx, Eigen::Lower>>);

  static_assert(hermitian_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Cd22_2>, Eigen::Lower>>);
  static_assert(hermitian_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Cd22_2>, Eigen::Upper>>);

  auto m22l = make_eigen_matrix<double, 2, 2>(3, 1, 1, 3);
  auto tlv22 = m22l.template triangularView<Eigen::Lower>();
  EXPECT_EQ(get_component(tlv22, 0, 0), 3);
  EXPECT_EQ(get_component(tlv22, 0, 1), 0);
  EXPECT_EQ(get_component(tlv22, 1, 0), 1);
  EXPECT_EQ(get_component(tlv22, 1, 1), 3);
  static_assert(not element_settable<decltype(tlv22), 2>);
  static_assert(element_settable<decltype(nested_object(tlv22)), 2>);
  set_component(nested_object(tlv22), 4, 1, 0);
  EXPECT_EQ(get_component(tlv22, 1, 0), 4);
  EXPECT_EQ(get_component(tlv22, 0, 1), 0);

  auto m22u = make_eigen_matrix<double, 2, 2>(3, 1, 1, 3);
  auto tuv22 = m22u.template triangularView<Eigen::Upper>();
  EXPECT_EQ(get_component(tuv22, 0, 0), 3);
  EXPECT_EQ(get_component(tuv22, 0, 1), 1);
  EXPECT_EQ(get_component(tuv22, 1, 0), 0);
  EXPECT_EQ(get_component(tuv22, 1, 1), 3);
  static_assert(not element_settable<decltype(tuv22), 2>);
  static_assert(element_settable<decltype(nested_object(tuv22)), 2>);
  set_component(nested_object(tuv22), 4, 0, 1);
  EXPECT_EQ(get_component(tuv22, 0, 1), 4);
  EXPECT_EQ(get_component(tuv22, 1, 0), 0);

  auto m22lc = make_eigen_matrix<std::complex<double>, 2, 2>(std::complex<double>{3, 0.3}, 0, std::complex<double>{1, 0.1}, 3);
  auto tlv22c = m22lc.template triangularView<Eigen::Lower>();
  EXPECT_EQ(std::real(get_component(tlv22c, 0, 0)), 3);
  EXPECT_EQ(std::real(get_component(tlv22c, 0, 1)), 0);
  EXPECT_EQ(std::imag(get_component(tlv22c, 0, 1)), 0);
  EXPECT_EQ(std::real(get_component(tlv22c, 1, 0)), 1);
  EXPECT_EQ(std::imag(get_component(tlv22c, 1, 0)), 0.1);
  EXPECT_EQ(std::real(get_component(tlv22c, 1, 1)), 3);
  static_assert(not element_settable<decltype(tlv22c), 2>);
  static_assert(element_settable<decltype(nested_object(tlv22c)), 2>);
  set_component(nested_object(tlv22c), std::complex<double>{4, 0.4}, 1, 0);
  EXPECT_EQ(std::imag(get_component(tlv22c, 1, 0)), 0.4);
  EXPECT_EQ(std::imag(get_component(tlv22c, 0, 1)), 0);
}

