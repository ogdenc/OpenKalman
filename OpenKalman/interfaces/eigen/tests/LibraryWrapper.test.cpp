/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, LibraryWrapper)
{
  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen3::EigenWrapper<M22>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen3::EigenWrapper<I22>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen3::EigenWrapper<M22&>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen3::EigenWrapper<I22&>>()))>);

  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<EigenWrapper<Eigen::DiagonalMatrix<double, 3>>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_object(std::declval<EigenWrapper<Eigen::DiagonalMatrix<double, 3>&>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<EigenWrapper<Eigen::DiagonalWrapper<M31>>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_object(std::declval<EigenWrapper<Eigen::DiagonalWrapper<M31>&>>()))>);

  static_assert(std::is_same_v<decltype(nested_object(std::declval<Eigen3::EigenWrapper<Eigen::DiagonalMatrix<double, 3>>&&>())), Eigen::DiagonalMatrix<double, 3>&&>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_object(std::declval<Eigen3::EigenWrapper<Eigen::DiagonalMatrix<double, 3>>>()))>);

  static_assert(Eigen3::eigen_general<Eigen3::EigenWrapper<Eigen::DiagonalMatrix<double, 3>>>);
  static_assert(Eigen3::eigen_wrapper<Eigen3::EigenWrapper<Eigen::DiagonalMatrix<double, 3>>>);
}

