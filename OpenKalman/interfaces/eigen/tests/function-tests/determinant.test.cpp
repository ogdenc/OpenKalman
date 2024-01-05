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


TEST(eigen3, determinant)
{
  auto m22 = make_dense_object_from<M22>(1, 2, 3, 4);

  EXPECT_NEAR(determinant(m22), -2, 1e-6);
  EXPECT_NEAR(determinant(m22.array()), -2, 1e-6);
  EXPECT_NEAR(determinant(Eigen3::make_eigen_wrapper(m22)), -2, 1e-6);
  EXPECT_NEAR(determinant(M2x {m22}), -2, 1e-6);
  EXPECT_NEAR(determinant(Mx2 {m22}), -2, 1e-6);
  EXPECT_NEAR(determinant(Mxx {m22}), -2, 1e-6);

  const M00 m00 {};

  EXPECT_NEAR(determinant(m00), 1, 1e-6);
  EXPECT_NEAR(determinant(M0x {m00}), 1, 1e-6);
  EXPECT_NEAR(determinant(Mx0 {m00}), 1, 1e-6);
  EXPECT_NEAR(determinant(Mxx {m00}), 1, 1e-6);

  EXPECT_NEAR(determinant(M11{2}), 2, 1e-6);
  EXPECT_NEAR(determinant((M1x(1,1) << 2).finished()), 2, 1e-6);
  EXPECT_NEAR(determinant((Mx1(1,1) << 2).finished()), 2, 1e-6);
  EXPECT_NEAR(determinant((Mxx(1,1) << 2).finished()), 2, 1e-6);

  auto cm22 = make_dense_object_from<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1});

  EXPECT_NEAR(std::real(determinant(cm22)), 0, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22)), 4, 1e-6);

  EXPECT_NEAR(std::real(determinant(cm22.array())), 0, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22.array())), 4, 1e-6);

  EXPECT_NEAR(std::real(determinant(Eigen3::make_eigen_wrapper(cm22))), 0, 1e-6);
  EXPECT_NEAR(std::imag(determinant(Eigen3::make_eigen_wrapper(cm22))), 4, 1e-6);

  auto m22_93310 = make_dense_object_from<M22>(9, 3, 3, 10);

  EXPECT_NEAR(determinant(m22_93310.template selfadjointView<Eigen::Lower>()), 81, 1e-6);
  EXPECT_NEAR(determinant(Eigen3::make_eigen_wrapper(m22_93310.template selfadjointView<Eigen::Lower>())), 81, 1e-6);

  auto m22_3103 = make_dense_object_from<M22>(3, 1, 0, 3);

  EXPECT_NEAR(determinant(m22_3103.template triangularView<Eigen::Upper>()), 9, 1e-6);

  auto m21 = M21 {1, 4};

  EXPECT_NEAR(determinant(Eigen::DiagonalMatrix<double, 2> {m21}), 4, 1e-6);
  EXPECT_NEAR(determinant(Eigen3::make_eigen_wrapper(Eigen::DiagonalMatrix<double, 2> {m21})), 4, 1e-6);

  EXPECT_NEAR(determinant(Eigen::DiagonalWrapper<M21> {m21}), 4, 1e-6);
  EXPECT_NEAR(determinant(Eigen3::make_eigen_wrapper(Eigen::DiagonalWrapper<M21> {m21})), 4, 1e-6);

  auto cm22_93310 = make_dense_object_from<CM22>(9, cdouble(3,1), cdouble(3,-1), 10);

  EXPECT_NEAR(std::real(determinant(cm22_93310)), 80, 1e-6);
  EXPECT_NEAR(std::real(determinant(cm22_93310.template selfadjointView<Eigen::Lower>())), 80, 1e-6);
  EXPECT_NEAR(std::real(determinant(cm22_93310.template selfadjointView<Eigen::Upper>())), 80, 1e-6);

  auto cm22_3103 = make_dense_object_from<CM22>(cdouble(3,-1), cdouble(1,-1), 0, 3);
  auto cm22_3013 = make_dense_object_from<CM22>(cdouble(3,-1), 0, cdouble(1,-1), 3);

  EXPECT_NEAR(std::real(determinant(cm22_3103.template triangularView<Eigen::Upper>())), 9, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22_3103.template triangularView<Eigen::Upper>())), -3, 1e-6);
  EXPECT_NEAR(std::real(determinant(cm22_3013.template triangularView<Eigen::Lower>())), 9, 1e-6);
  EXPECT_NEAR(std::imag(determinant(cm22_3013.template triangularView<Eigen::Lower>())), -3, 1e-6);
}

