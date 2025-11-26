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

#include "adapters.gtest.hpp"

#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

namespace
{
  using cdouble = std::complex<double>;

  using Mxx = eigen_matrix_t<double, dynamic_size_v, dynamic_size_v>;
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M1x = eigen_matrix_t<double, 1, dynamic_size_v>;
  using Mx1 = eigen_matrix_t<double, dynamic_size_v, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M2x = eigen_matrix_t<double, 2, dynamic_size_v>;
  using Mx2 = eigen_matrix_t<double, dynamic_size_v, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M44 = eigen_matrix_t<double, 4, 4>;
  using M99 = eigen_matrix_t<double, 4, 4>;

  using C21 = eigen_matrix_t<cdouble, 2, 1>;
  using C22 = eigen_matrix_t<cdouble, 2, 2>;
  using C20 = eigen_matrix_t<cdouble, 2, dynamic_size_v>;
  using C02 = eigen_matrix_t<cdouble, dynamic_size_v, 2>;
  using C01 = eigen_matrix_t<cdouble, dynamic_size_v, 1>;
  using C00 = eigen_matrix_t<cdouble, dynamic_size_v, dynamic_size_v>;

  using D2 = diagonal_adapter<eigen_matrix_t<double, 2, 1>>;
  using D0 = diagonal_adapter<eigen_matrix_t<double, dynamic_size_v, 1>>;

  using L22 = HermitianAdapter<M22, HermitianAdapterType::lower>;
  using L20 = HermitianAdapter<M2x, HermitianAdapterType::lower>;
  using L02 = HermitianAdapter<Mx2, HermitianAdapterType::lower>;
  using L00 = HermitianAdapter<Mxx, HermitianAdapterType::lower>;

  using U22 = HermitianAdapter<M22, HermitianAdapterType::upper>;
  using U20 = HermitianAdapter<M2x, HermitianAdapterType::upper>;
  using U02 = HermitianAdapter<Mx2, HermitianAdapterType::upper>;
  using U00 = HermitianAdapter<Mxx, HermitianAdapterType::upper>;
  
  using CL22 = HermitianAdapter<C22, HermitianAdapterType::lower>;
  using CU22 = HermitianAdapter<C22, HermitianAdapterType::upper>;

  using DL2 = HermitianAdapter<D2, HermitianAdapterType::lower>;
  using DL0 = HermitianAdapter<D0, HermitianAdapterType::lower>;

  template<typename...Args>
  inline auto mat22(Args...args) { return make_dense_writable_matrix_from<M22>(args...); }

  auto m_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m_4225 = make_dense_writable_matrix_from<M22>(4, 2, 2, 5);

  template<typename T> using D = diagonal_adapter<T>;
  template<typename T> using Tl = TriangularAdapter<T, triangle_type::lower>;
  template<typename T> using Tu = TriangularAdapter<T, triangle_type::upper>;
  template<typename T> using SAl = HermitianAdapter<T, HermitianAdapterType::lower>;
  template<typename T> using SAu = HermitianAdapter<T, HermitianAdapterType::upper>;
}


TEST(special_matrices, cholesky_diagonal)
{
  // constant diagonal
  static_assert(constant_diagonal_value{cholesky_factor<triangle_type::lower>(to_diagonal(make_constant<M41>(std::integral_constant<int, 4>{})))} == 2);

  // constant
  EXPECT_TRUE(is_near(cholesky_factor<triangle_type::lower>(make_constant<M22>(std::integral_constant<int, 4>{})), M22{2, 2, 0, 0}));

  auto d22 = Eigen::DiagonalMatrix<double, 2> {9, 16};
  const auto d22_sqrt = Eigen::DiagonalMatrix<double, 2> {3, 4};

  EXPECT_TRUE(is_near(cholesky_factor<triangle_type::lower>(d22), d22_sqrt));
  EXPECT_TRUE(is_near(cholesky_factor<triangle_type::upper>(d22), d22_sqrt));

  EXPECT_TRUE(is_near(cholesky_square(d22_sqrt), d22));
  EXPECT_TRUE(is_near(cholesky_square(d22_sqrt), d22));
}


TEST(special_matrices, cholesky_hermitian)
{
  auto m22_93310 = make_dense_object_from<M22>(9, 3, 3, 10);
  auto hl22 = Eigen::SelfAdjointView<M22, Eigen::Lower> {m22_93310};
  auto hu22 = Eigen::SelfAdjointView<M22, Eigen::Upper> {m22_93310};
  auto m22_3013 = make_dense_object_from<M22>(3, 0, 1, 3);
  auto m22_3103 = make_dense_object_from<M22>(3, 1, 0, 3);
  auto tl22 = Eigen::TriangularView<M22, Eigen::Lower> {m22_3013};
  auto tu22 = Eigen::TriangularView<M22, Eigen::Upper> {m22_3103};

  EXPECT_TRUE(is_near(cholesky_factor<triangle_type::lower>(hl22), tl22));
  EXPECT_TRUE(is_near(cholesky_factor<triangle_type::upper>(hl22), tu22));
  EXPECT_TRUE(is_near(cholesky_factor<triangle_type::lower>(hu22), tl22));
  EXPECT_TRUE(is_near(cholesky_factor<triangle_type::upper>(hu22), tu22));
}


// \todo Add triangular and hermitian cholesky tests


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

