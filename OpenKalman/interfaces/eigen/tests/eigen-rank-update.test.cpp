/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

TEST(eigen3, rank_update_self_adjoint)
{
  // u is zero:

  const auto m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  M22 m22 = m22_93310;
  const auto z22 = M22::Identity() - M22::Identity(); static_assert(zero_matrix<decltype(z22)>);

  auto hl22 = Eigen::SelfAdjointView<M22, Eigen::Lower> {m22};
  static_assert(eigen_SelfAdjointView<decltype(rank_update_self_adjoint(hl22, z22, 4))>);
  static_assert(eigen_SelfAdjointView<decltype(rank_update_self_adjoint(std::as_const(hl22), z22, 4))>);
  EXPECT_TRUE(is_near(rank_update_self_adjoint(hl22, z22, 4), m22_93310));
  EXPECT_TRUE(is_near(m22, m22_93310));

  auto hu22 = Eigen::SelfAdjointView<M22, Eigen::Upper> {m22};
  static_assert(eigen_SelfAdjointView<decltype(rank_update_self_adjoint(hu22, z22, 4))>);
  static_assert(eigen_SelfAdjointView<decltype(rank_update_self_adjoint(std::as_const(hu22), z22, 4))>);
  EXPECT_TRUE(is_near(rank_update_self_adjoint(hu22, z22, 4), m22_93310));
  EXPECT_TRUE(is_near(m22, m22_93310));

  // a is 1-by-1 (Dymamic requires creating a special matrix. Tests are in special_matrices tests.):

  const auto m11_2 = M11(2);
  const auto m14_1 = make_dense_writable_matrix_from<M14>(1, 1, 1, 1);
  const auto m11_25 = M11(25);

  auto m11 = M11(9);

  EXPECT_TRUE(is_near(rank_update_self_adjoint(std::as_const(m11), m11_2, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(std::as_const(m11), M1x {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M11 {m11}, Mx1 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M11 {m11}, Mxx {m11_2}, 4), m11_25));

  EXPECT_TRUE(is_near(rank_update_self_adjoint(std::as_const(m11), m14_1, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(std::as_const(m11), M1x {m14_1}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M11 {m11}, Mx4 {m14_1}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M11 {m11}, Mxx {m14_1}, 4), m11_25));

  rank_update_self_adjoint(m11, m11_2, 4); EXPECT_TRUE(is_near(m11, m11_25));
  m11 = M11(9);
  rank_update_self_adjoint(m11, m14_1, 4); EXPECT_TRUE(is_near(m11, m11_25));

  // Other tests require creation of DiagonalMatrix. See special_matrix tests.
}


TEST(eigen3, rank_update_triangular)
{
  // u is zero:

  const auto m22_3013 = make_dense_writable_matrix_from<M22>(3, 0, 1, 3);
  const auto m22_3103 = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);
  const auto z22 = M22::Identity() - M22::Identity(); static_assert(zero_matrix<decltype(z22)>);

  M22 m22 = m22_3013;
  auto tl22 = Eigen::TriangularView<M22, Eigen::Lower> {m22};
  static_assert(eigen_TriangularView<decltype(rank_update_triangular(tl22, z22, 4))>);
  static_assert(eigen_TriangularView<decltype(rank_update_triangular(std::as_const(tl22), z22, 4))>);
  EXPECT_TRUE(is_near(rank_update_triangular(tl22, z22, 4), m22_3013));
  EXPECT_TRUE(is_near(m22, m22_3013));

  m22 = m22_3103;
  auto tu22 = Eigen::TriangularView<M22, Eigen::Upper> {m22};
  static_assert(eigen_TriangularView<decltype(rank_update_triangular(tu22, z22, 4))>);
  static_assert(eigen_TriangularView<decltype(rank_update_triangular(std::as_const(tu22), z22, 4))>);
  EXPECT_TRUE(is_near(rank_update_triangular(tu22, z22, 4), m22_3103));
  EXPECT_TRUE(is_near(m22, m22_3103));

  // a is 1-by-1 (Dymamic requires creating a special matrix. Tests are in special_matrices tests.):

  const auto m11_2 = M11(2);
  const auto m14_1 = make_dense_writable_matrix_from<M14>(1, 1, 1, 1);
  const auto m11_5 = M11(5);

  auto m11 = M11(3);

  EXPECT_TRUE(is_near(rank_update_triangular(M11 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M11 {m11}, M1x {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(std::as_const(m11), Mx1 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(std::as_const(m11), Mxx {m11_2}, 4), m11_5));

  EXPECT_TRUE(is_near(rank_update_triangular(M11 {m11}, m14_1, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M11 {m11}, M1x {m14_1}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(std::as_const(m11), Mx4 {m14_1}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(std::as_const(m11), Mxx {m14_1}, 4), m11_5));

  rank_update_triangular(m11, m11_2, 4); EXPECT_TRUE(is_near(m11, m11_5));
  m11 = M11(3);
  rank_update_triangular(m11, m14_1, 4); EXPECT_TRUE(is_near(m11, m11_5));

  // Other tests require creation of DiagonalMatrix. See special_matrix tests.
}
