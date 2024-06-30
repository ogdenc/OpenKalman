/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, cholesky_diagonal)
{
  // zero
  auto z33 = M33::Identity() - M33::Identity();
  static_assert(constant_coefficient{cholesky_factor<TriangleType::lower>(z33)} == 0);
  static_assert(constant_coefficient{cholesky_factor<TriangleType::upper>(z33)} == 0);
  static_assert(constant_coefficient{cholesky_square(z33)} == 0);

  // identity
  static_assert(identity_matrix<decltype(cholesky_factor<TriangleType::lower>(M33::Identity()))>);
  static_assert(identity_matrix<decltype(cholesky_factor<TriangleType::upper>(M33::Identity()))>);
  static_assert(identity_matrix<decltype(cholesky_square(M33::Identity()))>);

  // constant_diagonal, constant, and diagonal require creation of special matrices.
}


TEST(eigen3, cholesky_hermitian)
{
  auto m22_93310 = make_dense_object_from<M22>(9, 3, 3, 10);
  auto hl22 = Eigen::SelfAdjointView<M22, Eigen::Lower> {m22_93310};
  auto hu22 = Eigen::SelfAdjointView<M22, Eigen::Upper> {m22_93310};
  auto m22_3013 = make_dense_object_from<M22>(3, 0, 1, 3);
  auto m22_3103 = make_dense_object_from<M22>(3, 1, 0, 3);
  auto tl22 = Eigen::TriangularView<M22, Eigen::Lower> {m22_3013};
  auto tu22 = Eigen::TriangularView<M22, Eigen::Upper> {m22_3103};

  EXPECT_TRUE(is_near(cholesky_square(tl22), hl22));
  EXPECT_TRUE(is_near(cholesky_square(tl22), hu22));
  EXPECT_TRUE(is_near(cholesky_square(tu22), hl22));
  EXPECT_TRUE(is_near(cholesky_square(tu22), hu22));
}


TEST(eigen3, LQ_and_QR_decomp_triangular)
{
  auto m22_lq = make_dense_object_from<M22>(-0.1, 0, 1.096, -1.272);
  EXPECT_TRUE(is_near(LQ_decomposition(m22_lq.triangularView<Eigen::Lower>()), m22_lq));
  EXPECT_TRUE(is_near(QR_decomposition(adjoint(m22_lq).triangularView<Eigen::Upper>()), adjoint(m22_lq)));
}


TEST(eigen3, LQ_and_QR_decomp_zero)
{
  auto z11 = M11::Identity() - M11::Identity();
  auto z22 = M22::Identity() - M22::Identity();
  auto z33 = M33::Identity() - M33::Identity();

  EXPECT_TRUE(is_near(LQ_decomposition(z11), z11));
  EXPECT_TRUE(is_near(LQ_decomposition(z22), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z33), z33));

  EXPECT_TRUE(is_near(QR_decomposition(z11), z11));
  EXPECT_TRUE(is_near(QR_decomposition(z22), z22));
  EXPECT_TRUE(is_near(QR_decomposition(z33), z33));
}

