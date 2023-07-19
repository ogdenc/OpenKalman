/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for decomposition functions.
 */

#include "special-matrices.gtest.hpp"

#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

namespace
{
  using cdouble = std::complex<double>;

  using Mxx = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M1x = eigen_matrix_t<double, 1, dynamic_size>;
  using Mx1 = eigen_matrix_t<double, dynamic_size, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M2x = eigen_matrix_t<double, 2, dynamic_size>;
  using Mx2 = eigen_matrix_t<double, dynamic_size, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M44 = eigen_matrix_t<double, 4, 4>;
  using M99 = eigen_matrix_t<double, 4, 4>;

  using C21 = eigen_matrix_t<cdouble, 2, 1>;
  using C22 = eigen_matrix_t<cdouble, 2, 2>;
  using C20 = eigen_matrix_t<cdouble, 2, dynamic_size>;
  using C02 = eigen_matrix_t<cdouble, dynamic_size, 2>;
  using C01 = eigen_matrix_t<cdouble, dynamic_size, 1>;
  using C00 = eigen_matrix_t<cdouble, dynamic_size, dynamic_size>;

  using D2 = DiagonalMatrix<eigen_matrix_t<double, 2, 1>>;
  using D0 = DiagonalMatrix<eigen_matrix_t<double, dynamic_size, 1>>;

  using L22 = SelfAdjointMatrix<M22, HermitianAdapterType::lower>;
  using L20 = SelfAdjointMatrix<M2x, HermitianAdapterType::lower>;
  using L02 = SelfAdjointMatrix<Mx2, HermitianAdapterType::lower>;
  using L00 = SelfAdjointMatrix<Mxx, HermitianAdapterType::lower>;

  using U22 = SelfAdjointMatrix<M22, HermitianAdapterType::upper>;
  using U20 = SelfAdjointMatrix<M2x, HermitianAdapterType::upper>;
  using U02 = SelfAdjointMatrix<Mx2, HermitianAdapterType::upper>;
  using U00 = SelfAdjointMatrix<Mxx, HermitianAdapterType::upper>;
  
  using CL22 = SelfAdjointMatrix<C22, HermitianAdapterType::lower>;
  using CU22 = SelfAdjointMatrix<C22, HermitianAdapterType::upper>;

  using DL2 = SelfAdjointMatrix<D2, HermitianAdapterType::lower>;
  using DL0 = SelfAdjointMatrix<D0, HermitianAdapterType::lower>;

  template<typename...Args>
  inline auto mat22(Args...args) { return make_dense_writable_matrix_from<M22>(args...); }

  auto m_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m_4225 = make_dense_writable_matrix_from<M22>(4, 2, 2, 5);

  template<typename T> using D = DiagonalMatrix<T>;
  template<typename T> using Tl = TriangularMatrix<T, TriangleType::lower>;
  template<typename T> using Tu = TriangularMatrix<T, TriangleType::upper>;
  template<typename T> using SAl = SelfAdjointMatrix<T, HermitianAdapterType::lower>;
  template<typename T> using SAu = SelfAdjointMatrix<T, HermitianAdapterType::upper>;
}


TEST(special_matrices, decompositions_constant)
{
  auto z11 = M11::Identity() - M11::Identity();
  auto z22 = M22::Identity() - M22::Identity();
  auto z33 = M33::Identity() - M33::Identity();
  auto z23 = z11.replicate<2, 3>();
  auto z32 = z11.replicate<3, 2>();
  EXPECT_TRUE(is_near(LQ_decomposition(z23), z22));
  EXPECT_TRUE(is_near(LQ_decomposition(z32), z33));
  EXPECT_TRUE(is_near(QR_decomposition(z23), z33));
  EXPECT_TRUE(is_near(QR_decomposition(z32), z22));

  auto c11 = M11::Identity() + M11::Identity();
  auto l44 = make_dense_writable_matrix_from<M44>(
    6, 0, 0, 0,
    6, 0, 0, 0,
    6, 0, 0, 0,
    6, 0, 0, 0);
  auto u44 = make_dense_writable_matrix_from<M44>(
    6, 6, 6, 6,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0);
  auto l99 = make_dense_writable_matrix_from<M99>(
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0);
  auto u99 = make_dense_writable_matrix_from<M99>(
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0);

  auto c49 = c11.replicate<4, 9>();
  auto c40_9 = Eigen::Replicate<decltype(c11), 4, Eigen::Dynamic> {c11, 4, 9};
  auto c09_4 = Eigen::Replicate<decltype(c11), Eigen::Dynamic, 9> {c11, 4, 9};
  auto c00_49 = Eigen::Replicate<decltype(c11), Eigen::Dynamic, Eigen::Dynamic> {c11, 4, 9};

  auto c94 = c11.replicate<9, 4>();
  auto c90_4 = Eigen::Replicate<decltype(c11), 9, Eigen::Dynamic> {c11, 9, 4};
  auto c04_9 = Eigen::Replicate<decltype(c11), Eigen::Dynamic, 4> {c11, 9, 4};
  auto c00_94 = Eigen::Replicate<decltype(c11), Eigen::Dynamic, Eigen::Dynamic> {c11, 9, 4};

  EXPECT_TRUE(is_near(LQ_decomposition(c49), l44));
  EXPECT_TRUE(is_near(LQ_decomposition(c40_9), l44));
  EXPECT_TRUE(is_near(LQ_decomposition(c09_4), l44));
  EXPECT_TRUE(is_near(LQ_decomposition(c00_49), l44));

  EXPECT_TRUE(is_near(LQ_decomposition(c94), l99));
  EXPECT_TRUE(is_near(LQ_decomposition(c90_4), l99));
  EXPECT_TRUE(is_near(LQ_decomposition(c04_9), l99));
  EXPECT_TRUE(is_near(LQ_decomposition(c00_94), l99));

  EXPECT_TRUE(is_near(QR_decomposition(c49), u99));
  EXPECT_TRUE(is_near(QR_decomposition(c40_9), u99));
  EXPECT_TRUE(is_near(QR_decomposition(c09_4), u99));
  EXPECT_TRUE(is_near(QR_decomposition(c00_49), u99));

  EXPECT_TRUE(is_near(QR_decomposition(c94), u44));
  EXPECT_TRUE(is_near(QR_decomposition(c90_4), u44));
  EXPECT_TRUE(is_near(QR_decomposition(c04_9), u44));
  EXPECT_TRUE(is_near(QR_decomposition(c00_94), u44));
}

