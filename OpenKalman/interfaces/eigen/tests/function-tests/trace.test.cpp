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


TEST(eigen3, trace)
{
  auto m22 = make_dense_object_from<M22>(1, 2, 3, 4);

  EXPECT_NEAR(trace(m22), 5, 1e-6);
  EXPECT_NEAR(trace(m22.array()), 5, 1e-6);
  EXPECT_NEAR(trace(Eigen3::make_eigen_wrapper(m22)), 5, 1e-6);
  EXPECT_NEAR(trace(M2x {m22}), 5, 1e-6);
  EXPECT_NEAR(trace(Mx2 {m22}), 5, 1e-6);
  EXPECT_NEAR(trace(Mxx {m22}), 5, 1e-6);

  const M00 m00 {};

  EXPECT_NEAR(trace(m00), 0, 1e-6);
  //EXPECT_NEAR(trace(M0x{m00}), 0, 1e-6); // Tested with special matrices
  //EXPECT_NEAR(trace(Mx0{m00}), 0, 1e-6); // Tested with special matrices
  EXPECT_NEAR(trace(Mxx{m00}), 0, 1e-6);

  EXPECT_NEAR(trace(M11{3}), 3, 1e-6);
  EXPECT_NEAR(trace((M1x(1,1) << 3).finished()), 3, 1e-6);
  EXPECT_NEAR(trace((Mx1(1,1) << 3).finished()), 3, 1e-6);
  EXPECT_NEAR(trace((Mxx(1,1) << 3).finished()), 3, 1e-6);

  auto cm22 = make_dense_object_from<CM22>(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1});

  EXPECT_NEAR(std::real(trace(cm22)), 5, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22)), 5, 1e-6);

  EXPECT_NEAR(std::real(trace(cm22.array())), 5, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22.array())), 5, 1e-6);

  EXPECT_NEAR(std::real(trace(Eigen3::make_eigen_wrapper(cm22))), 5, 1e-6);
  EXPECT_NEAR(std::imag(trace(Eigen3::make_eigen_wrapper(cm22))), 5, 1e-6);

  auto m22_93310 = make_dense_object_from<M22>(9, 3, 3, 10);

  EXPECT_NEAR(trace(m22_93310.template selfadjointView<Eigen::Lower>()), 19, 1e-6);
  EXPECT_NEAR(trace(Eigen3::make_eigen_wrapper(m22_93310.template selfadjointView<Eigen::Lower>())), 19, 1e-6);

  auto m22_3103 = make_dense_object_from<M22>(3, 1, 0, 3);

  EXPECT_NEAR(trace(m22_3103.template triangularView<Eigen::Upper>()), 6, 1e-6);

  auto m21 = M21 {1, 4};

  EXPECT_NEAR(trace(Eigen::DiagonalMatrix<double, 2> {m21}), 5, 1e-6);
  EXPECT_NEAR(trace(Eigen3::make_eigen_wrapper(Eigen::DiagonalMatrix<double, 2> {m21})), 5, 1e-6);

  EXPECT_NEAR(trace(Eigen::DiagonalWrapper<M21> {m21}), 5, 1e-6);
  EXPECT_NEAR(trace(Eigen3::make_eigen_wrapper(Eigen::DiagonalWrapper<M21> {m21})), 5, 1e-6);

  auto cm22_93310 = make_dense_object_from<CM22>(9, cdouble(3,1), cdouble(3,-1), 10);

  EXPECT_NEAR(std::real(trace(cm22_93310.template selfadjointView<Eigen::Lower>())), 19, 1e-6);
  EXPECT_NEAR(std::real(trace(cm22_93310.template selfadjointView<Eigen::Upper>())), 19, 1e-6);

  auto cm22_3103 = make_dense_object_from<CM22>(cdouble(3,-1), cdouble(1,-1), 0, 3);
  auto cm22_3013 = make_dense_object_from<CM22>(cdouble(3,-1), 0, cdouble(1,-1), 3);

  EXPECT_NEAR(std::real(trace(cm22_3013.template triangularView<Eigen::Lower>())), 6, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22_3013.template triangularView<Eigen::Lower>())), -1, 1e-6);
  EXPECT_NEAR(std::real(trace(cm22_3103.template triangularView<Eigen::Upper>())), 6, 1e-6);
  EXPECT_NEAR(std::imag(trace(cm22_3103.template triangularView<Eigen::Upper>())), -1, 1e-6);
}

