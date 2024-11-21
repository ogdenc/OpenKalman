/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear-algebra/interfaces/eigen/tests/eigen.gtest.hpp"
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

