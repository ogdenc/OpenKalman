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
 * \brief Tests for linear algebra functions.
 */

#include "special-matrices.gtest.hpp"

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
  using M23 = eigen_matrix_t<double, 2, 3>;
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

  using D2 = DiagonalMatrix<eigen_matrix_t<double, 2, 1>>;
  using D0 = DiagonalMatrix<eigen_matrix_t<double, dynamic_size_v, 1>>;

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


TEST(special_matrices, contract)
{
  auto m31a = make_dense_writable_matrix_from<M31>(2, 3, 4);
  auto m01_3a = Mx1{m31a};
  auto m31b = make_dense_writable_matrix_from<M31>(5, 6, 7);
  auto m01_3b = Mx1{m31b};

  auto dm3a = Eigen::DiagonalMatrix<double, 3>{m31a};
  auto dm0_3a = Eigen::DiagonalMatrix<double, Eigen::Dynamic>{m31a};
  auto dw3a = Eigen::DiagonalWrapper{m31a};
  auto dw0_3a = Eigen::DiagonalWrapper{m01_3a};

  auto dm3b = Eigen::DiagonalMatrix<double, 3>{m31b};
  auto dm0_3b = Eigen::DiagonalMatrix<double, Eigen::Dynamic>{m31b};
  auto dw3b = Eigen::DiagonalWrapper{m31b};
  auto dw0_3b = Eigen::DiagonalWrapper{m01_3b};

  M33 d3c {make_dense_writable_matrix_from<M31>(10, 18, 28).asDiagonal()};

  EXPECT_TRUE(is_near(contract(dm3a, dm3b), d3c));
  EXPECT_TRUE(is_near(contract(EigenWrapper {dm3a}, EigenWrapper {dm3b}), d3c));
  EXPECT_TRUE(is_near(contract(dm0_3a, dm3b), d3c));
  EXPECT_TRUE(is_near(contract(dm3a, dm0_3b), d3c));
  EXPECT_TRUE(is_near(contract(dm0_3a, dm0_3b), d3c));
  EXPECT_TRUE(is_near(contract(dw3a, dw3b), d3c));
  EXPECT_TRUE(is_near(contract(dw0_3a, dw3b), d3c));
  EXPECT_TRUE(is_near(contract(dw3a, dw0_3b), d3c));
  EXPECT_TRUE(is_near(contract(dw0_3a, dw0_3b), d3c));

  auto m23_468 = make_dense_writable_matrix_from<M23>(4, 6, 8, 4, 6, 8);

  EXPECT_TRUE(is_near(contract(make_constant_matrix_like<M23, 2>(), dm3a), m23_468));
  EXPECT_TRUE(is_near(contract(make_constant_matrix_like<M23, 2>(), dw3a), m23_468));

  auto m23_151821 = make_dense_writable_matrix_from<M33>(15, 15, 15, 18, 18, 18, 21, 21, 21);

  EXPECT_TRUE(is_near(contract(dm3b, make_constant_matrix_like<M33, 3>()), m23_151821));
  EXPECT_TRUE(is_near(contract(dw3b, make_constant_matrix_like<M33, 3>()), m23_151821));
}

