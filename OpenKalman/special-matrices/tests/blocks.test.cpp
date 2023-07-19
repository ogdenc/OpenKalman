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
 * \brief Tests for block functions.
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

  using C21 = eigen_matrix_t<cdouble, 2, 1>;
  using C22 = eigen_matrix_t<cdouble, 2, 2>;
  using C20 = eigen_matrix_t<cdouble, 2, dynamic_size>;
  using C02 = eigen_matrix_t<cdouble, dynamic_size, 2>;
  using C01 = eigen_matrix_t<cdouble, dynamic_size, 1>;
  using C00 = eigen_matrix_t<cdouble, dynamic_size, dynamic_size>;

  using D2 = DiagonalMatrix<eigen_matrix_t<double, 2, 1>>;
  using D0 = DiagonalMatrix<eigen_matrix_t<double, dynamic_size, 1>>;

  using L22 = SelfAdjointMatrix<M22, TriangleType::lower>;
  using L20 = SelfAdjointMatrix<M2x, TriangleType::lower>;
  using L02 = SelfAdjointMatrix<Mx2, TriangleType::lower>;
  using L00 = SelfAdjointMatrix<Mxx, TriangleType::lower>;

  using U22 = SelfAdjointMatrix<M22, TriangleType::upper>;
  using U20 = SelfAdjointMatrix<M2x, TriangleType::upper>;
  using U02 = SelfAdjointMatrix<Mx2, TriangleType::upper>;
  using U00 = SelfAdjointMatrix<Mxx, TriangleType::upper>;
  
  using CL22 = SelfAdjointMatrix<C22, TriangleType::lower>;
  using CU22 = SelfAdjointMatrix<C22, TriangleType::upper>;

  using DM22 = SelfAdjointMatrix<M22, TriangleType::diagonal>;
  using DM20 = SelfAdjointMatrix<M2x, TriangleType::diagonal>;
  using DM02 = SelfAdjointMatrix<Mx2, TriangleType::diagonal>;
  using DM00 = SelfAdjointMatrix<Mxx, TriangleType::diagonal>;
  
  using DD2 = SelfAdjointMatrix<D2, TriangleType::diagonal>;
  using DD0 = SelfAdjointMatrix<D0, TriangleType::diagonal>;
  
  using DL2 = SelfAdjointMatrix<D2, TriangleType::lower>;
  using DL0 = SelfAdjointMatrix<D0, TriangleType::lower>;

  template<typename...Args>
  inline auto mat22(Args...args) { return make_dense_writable_matrix_from<M22>(args...); }

  auto m_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m_4225 = make_dense_writable_matrix_from<M22>(4, 2, 2, 5);

  template<typename T> using D = DiagonalMatrix<T>;
  template<typename T> using Tl = TriangularMatrix<T, TriangleType::lower>;
  template<typename T> using Tu = TriangularMatrix<T, TriangleType::upper>;
  template<typename T> using SAl = SelfAdjointMatrix<T, TriangleType::lower>;
  template<typename T> using SAu = SelfAdjointMatrix<T, TriangleType::upper>;
}


TEST(special_matrices, set_triangle)
{
  const auto a33 = make_dense_writable_matrix_from<M33>(
    1, 2, 3,
    2, 4, 5,
    3, 5, 6);
  const auto b33 = make_dense_writable_matrix_from<M33>(
    1.5, 2.5, 3.5,
    2.5, 4.5, 5.5,
    3.5, 5.5, 6.5);
  const auto d33 = make_dense_writable_matrix_from<M33>(
    1.5, 2, 3,
    2, 4.5, 5,
    3, 5, 6.5);
  const auto d31 = make_dense_writable_matrix_from<M31>(1, 4, 6);
  const auto e31 = make_dense_writable_matrix_from<M31>(1.5, 4.5, 6.5);
  const auto e33 = e31.asDiagonal();

  M33 a;

  a = a33; internal::set_triangle<TriangleType::diagonal>(a, b33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(internal::set_triangle<TriangleType::diagonal>(a33, b33), d33));

  a = a33; internal::set_triangle<TriangleType::diagonal>(a.triangularView<Eigen::Lower>(), b33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(nested_matrix(internal::set_triangle<TriangleType::diagonal>(a33.triangularView<Eigen::Lower>(), b33)), d33));

  a = a33; internal::set_triangle<TriangleType::diagonal>(EigenWrapper {a.triangularView<Eigen::Upper>()}, b33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(nested_matrix(internal::set_triangle<TriangleType::diagonal>(EigenWrapper {a33.triangularView<Eigen::Upper>()}, b33)), d33));

  a = a33; internal::set_triangle<TriangleType::diagonal>(a.selfadjointView<Eigen::Upper>(), b33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(internal::set_triangle<TriangleType::diagonal>(a33.selfadjointView<Eigen::Upper>(), b33), d33));

  a = a33; internal::set_triangle<TriangleType::diagonal>(EigenWrapper {a.selfadjointView<Eigen::Lower>()}, b33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(internal::set_triangle<TriangleType::diagonal>(EigenWrapper {a33.selfadjointView<Eigen::Lower>()}, b33), d33));

  a = a33; internal::set_triangle(a, e33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(internal::set_triangle(a33, e33), d33));

  a = a33; internal::set_triangle(a.triangularView<Eigen::Lower>(), e33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(nested_matrix(internal::set_triangle(a33.triangularView<Eigen::Lower>(), e33)), d33));

  a = a33; internal::set_triangle(EigenWrapper {a.triangularView<Eigen::Upper>()}, e33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(nested_matrix(internal::set_triangle(EigenWrapper {a33.triangularView<Eigen::Upper>()}, e33)), d33));

  a = a33; internal::set_triangle(a.selfadjointView<Eigen::Upper>(), e33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(internal::set_triangle(a33.selfadjointView<Eigen::Upper>(), e33), d33));

  a = a33; internal::set_triangle(EigenWrapper {a.selfadjointView<Eigen::Lower>()}, e33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(internal::set_triangle(EigenWrapper {a33.selfadjointView<Eigen::Lower>()}, e33), d33));

  M31 d;

  d = d31; internal::set_triangle(Eigen::DiagonalWrapper<M31>{d}, b33);
  EXPECT_TRUE(is_near(d, e31));
  EXPECT_TRUE(is_near(internal::set_triangle<TriangleType::diagonal>(d31.asDiagonal(), b33), e33));
  EXPECT_TRUE(is_near(internal::set_triangle(d31.asDiagonal(), b33), e33));
  EXPECT_TRUE(is_near(internal::set_triangle(Eigen::DiagonalMatrix<double, 3>{d31}, b33), e33));

  d = d31; internal::set_triangle(Eigen::DiagonalWrapper<M31>{d}, e33);
  EXPECT_TRUE(is_near(d, e31));
  EXPECT_TRUE(is_near(internal::set_triangle<TriangleType::diagonal>(d31.asDiagonal(), e33), e33));
  EXPECT_TRUE(is_near(internal::set_triangle(d31.asDiagonal(), e33), e33));
  EXPECT_TRUE(is_near(internal::set_triangle(Eigen::DiagonalMatrix<double, 3>{d31}, e33), e33));
}

