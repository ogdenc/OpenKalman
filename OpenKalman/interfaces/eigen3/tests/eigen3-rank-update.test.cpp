/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen3.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


namespace
{
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M12 = eigen_matrix_t<double, 1, 2>;
  using M13 = eigen_matrix_t<double, 1, 3>;
  using M14 = eigen_matrix_t<double, 1, 4>;
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M23 = eigen_matrix_t<double, 2, 3>;
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M34 = eigen_matrix_t<double, 3, 4>;
  using M43 = eigen_matrix_t<double, 4, 3>;
  using M44 = eigen_matrix_t<double, 4, 4>;
  using M55 = eigen_matrix_t<double, 5, 5>;

  using M00 = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using M10 = eigen_matrix_t<double, 1, dynamic_size>;
  using M01 = eigen_matrix_t<double, dynamic_size, 1>;
  using M20 = eigen_matrix_t<double, 2, dynamic_size>;
  using M02 = eigen_matrix_t<double, dynamic_size, 2>;
  using M30 = eigen_matrix_t<double, 3, dynamic_size>;
  using M03 = eigen_matrix_t<double, dynamic_size, 3>;
  using M04 = eigen_matrix_t<double, dynamic_size, 4>;
  using M50 = eigen_matrix_t<double, 5, dynamic_size>;
  using M05 = eigen_matrix_t<double, dynamic_size, 5>;

  using cdouble = std::complex<double>;

  using CM21 = eigen_matrix_t<cdouble, 2, 1>;
  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;
}


TEST(eigen3, rank_update_self_adjoint)
{
  // u is zero:

  const auto m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  M22 m22 = m22_93310;
  const auto m22_2012 = make_dense_writable_matrix_from<M22>(2, 0, 1, 2);
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
  const auto m14_05 = make_dense_writable_matrix_from<M14>(0.5, 0.5, 0.5, 0.5);
  const auto m11_25 = M11(25);

  auto m11 = M11(9);

  EXPECT_TRUE(is_near(rank_update_self_adjoint(std::as_const(m11), m11_2, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(std::as_const(m11), M10 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M11 {m11}, M01 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M11 {m11}, M00 {m11_2}, 4), m11_25));

  EXPECT_TRUE(is_near(rank_update_self_adjoint(std::as_const(m11), m14_05, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(std::as_const(m11), M10 {m14_05}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M11 {m11}, M01 {m14_05}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M11 {m11}, M00 {m14_05}, 4), m11_25));

  rank_update_self_adjoint(m11, m11_2, 4); EXPECT_TRUE(is_near(m11, m11_25));
  m11 = M11(9);
  rank_update_self_adjoint(m11, m14_05, 4); EXPECT_TRUE(is_near(m11, m11_25));

  // Other tests require creation of DiagonalMatrix. See special_matrix tests.
}


TEST(eigen3, rank_update_triangular)
{
  // u is zero:

  const auto m22_3013 = make_dense_writable_matrix_from<M22>(3, 0, 1, 3);
  const auto m22_3103 = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);
  const auto m22_2012 = make_dense_writable_matrix_from<M22>(2, 0, 1, 2);
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
  const auto m14_05 = make_dense_writable_matrix_from<M14>(0.5, 0.5, 0.5, 0.5);
  const auto m11_5 = M11(5);

  auto m11 = M11(3);

  EXPECT_TRUE(is_near(rank_update_triangular(M11 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M11 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(std::as_const(m11), M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(std::as_const(m11), M00 {m11_2}, 4), m11_5));

  EXPECT_TRUE(is_near(rank_update_triangular(M11 {m11}, m14_05, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M11 {m11}, M10 {m14_05}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(std::as_const(m11), M01 {m14_05}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(std::as_const(m11), M00 {m14_05}, 4), m11_5));

  rank_update_triangular(m11, m11_2, 4); EXPECT_TRUE(is_near(m11, m11_5));
  m11 = M11(3);
  rank_update_triangular(m11, m14_05, 4); EXPECT_TRUE(is_near(m11, m11_5));

  // Other tests require creation of DiagonalMatrix. See special_matrix tests.
}
