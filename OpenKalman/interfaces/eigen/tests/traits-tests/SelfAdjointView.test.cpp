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


TEST(eigen3, Eigen_SelfAdjointView)
{
  static_assert(std::is_same_v<nested_object_of_t<Eigen::SelfAdjointView<M22, Eigen::Lower>>, M22&>);

  static_assert(not eigen_matrix_general<Eigen::SelfAdjointView<M33, Eigen::Lower>, true>);
  static_assert(not writable<Eigen::SelfAdjointView<M33, Eigen::Lower>>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M33, Eigen::Lower>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M3x, Eigen::Lower>>, M3x>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<Mx3, Eigen::Lower>>, Mx3>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<Mxx, Eigen::Lower>>, Mxx>);

  static_assert(not has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M22, Eigen::Lower>>>);
  static_assert(not has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M22, Eigen::Upper>>>);
  static_assert(has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<Mx2, Eigen::Lower>>>);
  static_assert(has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<Mx2, Eigen::Upper>>>);
  static_assert(has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M2x, Eigen::Lower>>>);
  static_assert(has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M2x, Eigen::Upper>>>);

  static_assert(constant_coefficient_v<Eigen::SelfAdjointView<Eigen::MatrixWrapper<C22_2>, Eigen::Upper>> == 2);

  static_assert(constant_matrix<C11_1cx>);
  static_assert(std::real(constant_coefficient_v<C11_1cx>) == 1);
  static_assert(std::imag(constant_coefficient_v<C11_1cx>) == 0);
  static_assert(constant_matrix<Eigen::SelfAdjointView<Eigen::MatrixWrapper<C11_1cx>, Eigen::Lower>>);

  static_assert(constant_matrix<C11_2cx>);
  EXPECT_EQ(std::real(constant_coefficient{CM11::Identity() + CM11::Identity()}()), 2);
  EXPECT_EQ(std::imag(constant_coefficient{CM11::Identity() + CM11::Identity()}()), 0);
  static_assert(constant_matrix<Eigen::SelfAdjointView<Eigen::MatrixWrapper<C11_2cx>, Eigen::Lower>>);

  static_assert(constant_diagonal_coefficient_v<Eigen::SelfAdjointView<Eigen::MatrixWrapper<Cd22_2>, Eigen::Upper>> == 2);

  static_assert(zero<Eigen::SelfAdjointView<Eigen::MatrixWrapper<Z22>, Eigen::Upper>>);

  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M33::Identity()), Eigen::Upper>>);
  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M3x::Identity(3, 3)), Eigen::Lower>>);
  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(Mx3::Identity(3, 3)), Eigen::Upper>>);
  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(Mxx::Identity(3, 3)), Eigen::Lower>>);

  static_assert(diagonal_matrix<Sadv22>);
  static_assert(diagonal_matrix<Sadv2x>);
  static_assert(diagonal_matrix<Sadvx2>);
  static_assert(diagonal_matrix<Sadvxx>);

  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Lower>, HermitianAdapterType::lower>);
  static_assert(not hermitian_adapter<Eigen::SelfAdjointView<CM22, Eigen::Lower>, HermitianAdapterType::lower>); // the diagonal must be real

  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Upper>, HermitianAdapterType::upper>);
  static_assert(not hermitian_adapter<Eigen::SelfAdjointView<CM22, Eigen::Upper>, HermitianAdapterType::upper>); // the diagonal must be real

  static_assert(hermitian_matrix<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(hermitian_matrix<Eigen::SelfAdjointView<M33, Eigen::Upper>>);

  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Upper>>);
  static_assert(not hermitian_adapter<Eigen::SelfAdjointView<M43, Eigen::Lower>, HermitianAdapterType::lower>);
  static_assert(not hermitian_adapter<Eigen::SelfAdjointView<M34, Eigen::Upper>, HermitianAdapterType::upper>);

  auto m22l = make_eigen_matrix<double, 2, 2>(9, 0, 3, 10);
  auto salv22 = m22l.template selfadjointView<Eigen::Lower>();
  EXPECT_EQ(get_component(salv22, 0, 0), 9);
  EXPECT_EQ(get_component(salv22, 0, 1), 3);
  EXPECT_EQ(get_component(salv22, 1, 0), 3);
  EXPECT_EQ(get_component(salv22, 1, 1), 10);
  static_assert(std::is_lvalue_reference_v<decltype(get_component(salv22, 0, 1))>);
  static_assert(element_settable<decltype(salv22), 2>);
  set_component(salv22, 4, 0, 1);
  EXPECT_EQ(get_component(salv22, 0, 1), 4);
  EXPECT_EQ(get_component(salv22, 1, 0), 4);

  auto m22u = make_eigen_matrix<double, 2, 2>(9, 3, 0, 10);
  auto sauv22 = m22u.template selfadjointView<Eigen::Upper>();
  EXPECT_EQ(get_component(sauv22, 0, 0), 9);
  EXPECT_EQ(get_component(sauv22, 0, 1), 3);
  EXPECT_EQ(get_component(sauv22, 1, 0), 3);
  EXPECT_EQ(get_component(sauv22, 1, 1), 10);
  static_assert(std::is_lvalue_reference_v<decltype(get_component(sauv22, 1, 0))>);
  static_assert(element_settable<decltype(sauv22), 2>);
  set_component(sauv22, 4, 1, 0);
  EXPECT_EQ(get_component(sauv22, 0, 1), 4);
  EXPECT_EQ(get_component(sauv22, 1, 0), 4);

  auto m22lc = make_eigen_matrix<std::complex<double>, 2, 2>(std::complex<double>{9, 0.9}, 0, std::complex<double>{3, 0.3}, 10);
  auto salv22c = m22lc.template selfadjointView<Eigen::Lower>();
  EXPECT_EQ(std::real(get_component(salv22c, 0, 0)), 9);
  EXPECT_EQ(std::real(get_component(salv22c, 0, 1)), 3);
  EXPECT_EQ(std::imag(get_component(salv22c, 0, 1)), -0.3);
  EXPECT_EQ(std::real(get_component(salv22c, 1, 0)), 3);
  EXPECT_EQ(std::imag(get_component(salv22c, 1, 0)), 0.3);
  EXPECT_EQ(std::real(get_component(salv22c, 1, 1)), 10);
  static_assert(not std::is_lvalue_reference_v<decltype(get_component(salv22c, 0, 1))>);
  static_assert(element_settable<decltype(salv22c), 2>);
  set_component(salv22c, std::complex<double>{4, 0.4}, 0, 1);
  EXPECT_EQ(std::imag(get_component(salv22c, 0, 1)), 0.4);
  EXPECT_EQ(std::imag(get_component(salv22c, 1, 0)), -0.4);
}

