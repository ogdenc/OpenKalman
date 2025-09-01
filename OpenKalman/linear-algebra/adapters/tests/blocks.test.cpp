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

  using C21 = eigen_matrix_t<cdouble, 2, 1>;
  using C22 = eigen_matrix_t<cdouble, 2, 2>;
  using C20 = eigen_matrix_t<cdouble, 2, dynamic_size_v>;
  using C02 = eigen_matrix_t<cdouble, dynamic_size_v, 2>;
  using C01 = eigen_matrix_t<cdouble, dynamic_size_v, 1>;
  using C00 = eigen_matrix_t<cdouble, dynamic_size_v, dynamic_size_v>;

  using D2 = diagonal_adapter<eigen_matrix_t<double, 2, 1>>;
  using D0 = diagonal_adapter<eigen_matrix_t<double, dynamic_size_v, 1>>;

  using L22 = HermitianAdapter<M22, triangle_type::lower>;
  using L20 = HermitianAdapter<M2x, triangle_type::lower>;
  using L02 = HermitianAdapter<Mx2, triangle_type::lower>;
  using L00 = HermitianAdapter<Mxx, triangle_type::lower>;

  using U22 = HermitianAdapter<M22, triangle_type::upper>;
  using U20 = HermitianAdapter<M2x, triangle_type::upper>;
  using U02 = HermitianAdapter<Mx2, triangle_type::upper>;
  using U00 = HermitianAdapter<Mxx, triangle_type::upper>;
  
  using CL22 = HermitianAdapter<C22, triangle_type::lower>;
  using CU22 = HermitianAdapter<C22, triangle_type::upper>;

  using DM22 = HermitianAdapter<M22, triangle_type::diagonal>;
  using DM20 = HermitianAdapter<M2x, triangle_type::diagonal>;
  using DM02 = HermitianAdapter<Mx2, triangle_type::diagonal>;
  using DM00 = HermitianAdapter<Mxx, triangle_type::diagonal>;
  
  using DD2 = HermitianAdapter<D2, triangle_type::diagonal>;
  using DD0 = HermitianAdapter<D0, triangle_type::diagonal>;
  
  using DL2 = HermitianAdapter<D2, triangle_type::lower>;
  using DL0 = HermitianAdapter<D0, triangle_type::lower>;

  template<typename...Args>
  inline auto mat22(Args...args) { return make_dense_writable_matrix_from<M22>(args...); }

  auto m_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m_4225 = make_dense_writable_matrix_from<M22>(4, 2, 2, 5);

  template<typename T> using D = diagonal_adapter<T>;
  template<typename T> using Tl = TriangularAdapter<T, triangle_type::lower>;
  template<typename T> using Tu = TriangularAdapter<T, triangle_type::upper>;
  template<typename T> using SAl = HermitianAdapter<T, triangle_type::lower>;
  template<typename T> using SAu = HermitianAdapter<T, triangle_type::upper>;
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

  a = a33; internal::set_triangle<triangle_type::diagonal>(a, b33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(internal::set_triangle<triangle_type::diagonal>(a33, b33), d33));

  a = a33; internal::set_triangle<triangle_type::diagonal>(a.triangularView<Eigen::Lower>(), b33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(nested_matrix(internal::set_triangle<triangle_type::diagonal>(a33.triangularView<Eigen::Lower>(), b33)), d33));

  a = a33; internal::set_triangle<triangle_type::diagonal>(EigenWrapper {a.triangularView<Eigen::Upper>()}, b33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(nested_matrix(internal::set_triangle<triangle_type::diagonal>(EigenWrapper {a33.triangularView<Eigen::Upper>()}, b33)), d33));

  a = a33; internal::set_triangle<triangle_type::diagonal>(a.selfadjointView<Eigen::Upper>(), b33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(internal::set_triangle<triangle_type::diagonal>(a33.selfadjointView<Eigen::Upper>(), b33), d33));

  a = a33; internal::set_triangle<triangle_type::diagonal>(EigenWrapper {a.selfadjointView<Eigen::Lower>()}, b33);
  EXPECT_TRUE(is_near(a, d33));
  EXPECT_TRUE(is_near(internal::set_triangle<triangle_type::diagonal>(EigenWrapper {a33.selfadjointView<Eigen::Lower>()}, b33), d33));

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
  EXPECT_TRUE(is_near(internal::set_triangle<triangle_type::diagonal>(d31.asDiagonal(), b33), e33));
  EXPECT_TRUE(is_near(internal::set_triangle(d31.asDiagonal(), b33), e33));
  EXPECT_TRUE(is_near(internal::set_triangle(Eigen::DiagonalMatrix<double, 3>{d31}, b33), e33));

  d = d31; internal::set_triangle(Eigen::DiagonalWrapper<M31>{d}, e33);
  EXPECT_TRUE(is_near(d, e31));
  EXPECT_TRUE(is_near(internal::set_triangle<triangle_type::diagonal>(d31.asDiagonal(), e33), e33));
  EXPECT_TRUE(is_near(internal::set_triangle(d31.asDiagonal(), e33), e33));
  EXPECT_TRUE(is_near(internal::set_triangle(Eigen::DiagonalMatrix<double, 3>{d31}, e33), e33));
}

