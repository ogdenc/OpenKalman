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
 * \brief Tests for rank-update functions.
 */

#include "special-matrices.gtest.hpp"

#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

namespace
{
  using cdouble = std::complex<double>;

  using M21 = eigen_matrix_t<double, 2, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M20 = eigen_matrix_t<double, 2, dynamic_size>;
  using M02 = eigen_matrix_t<double, dynamic_size, 2>;
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M10 = eigen_matrix_t<double, 1, dynamic_size>;
  using M01 = eigen_matrix_t<double, dynamic_size, 1>;
  using M00 = eigen_matrix_t<double, dynamic_size, dynamic_size>;

  using C21 = eigen_matrix_t<cdouble, 2, 1>;
  using C22 = eigen_matrix_t<cdouble, 2, 2>;
  using C20 = eigen_matrix_t<cdouble, 2, dynamic_size>;
  using C02 = eigen_matrix_t<cdouble, dynamic_size, 2>;
  using C01 = eigen_matrix_t<cdouble, dynamic_size, 1>;
  using C00 = eigen_matrix_t<cdouble, dynamic_size, dynamic_size>;

  using D2 = DiagonalMatrix<eigen_matrix_t<double, 2, 1>>;
  using D0 = DiagonalMatrix<eigen_matrix_t<double, dynamic_size, 1>>;

  using L22 = SelfAdjointMatrix<M22, TriangleType::lower>;
  using L20 = SelfAdjointMatrix<M20, TriangleType::lower>;
  using L02 = SelfAdjointMatrix<M02, TriangleType::lower>;
  using L00 = SelfAdjointMatrix<M00, TriangleType::lower>;

  using U22 = SelfAdjointMatrix<M22, TriangleType::upper>;
  using U20 = SelfAdjointMatrix<M20, TriangleType::upper>;
  using U02 = SelfAdjointMatrix<M02, TriangleType::upper>;
  using U00 = SelfAdjointMatrix<M00, TriangleType::upper>;
  
  using CL22 = SelfAdjointMatrix<C22, TriangleType::lower>;
  using CU22 = SelfAdjointMatrix<C22, TriangleType::upper>;

  using DM22 = SelfAdjointMatrix<M22, TriangleType::diagonal>;
  using DM20 = SelfAdjointMatrix<M20, TriangleType::diagonal>;
  using DM02 = SelfAdjointMatrix<M02, TriangleType::diagonal>;
  using DM00 = SelfAdjointMatrix<M00, TriangleType::diagonal>;
  
  using DD2 = SelfAdjointMatrix<D2, TriangleType::diagonal>;
  using DD0 = SelfAdjointMatrix<D0, TriangleType::diagonal>;
  
  using DL2 = SelfAdjointMatrix<D2, TriangleType::lower>;
  using DL0 = SelfAdjointMatrix<D0, TriangleType::lower>;

  template<typename...Args>
  inline auto mat22(Args...args) { return MatrixTraits<M22>::make(args...); }

  auto m_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m_4225 = make_dense_writable_matrix_from<M22>(4, 2, 2, 5);

  template<typename T> using D = DiagonalMatrix<T>;
  template<typename T> using Tl = TriangularMatrix<T, TriangleType::lower>;
  template<typename T> using Tu = TriangularMatrix<T, TriangleType::upper>;
  template<typename T> using SAl = SelfAdjointMatrix<T, TriangleType::lower>;
  template<typename T> using SAu = SelfAdjointMatrix<T, TriangleType::upper>;
}


TEST(special_matrices, DiagonalMatrix_rank_update_1x1)
{
  auto s2 = DiagonalMatrix {3., 3};
  rank_update(s2, DiagonalMatrix {2., 2}, 4);
  EXPECT_TRUE(is_near(s2, MatrixTraits<M22>::make(5., 0, 0, 5)));
  s2 = DiagonalMatrix {3., 3};
  rank_update(s2, 2 * M22::Identity(), 4);
  EXPECT_TRUE(is_near(s2, MatrixTraits<M22>::make(5., 0, 0, 5)));
  //
  using M1by1 = eigen_matrix_t<double, 1, 1>;
  auto s2a = M1by1 {3.};
  rank_update(s2a, M1by1 {2.}, 4);
  EXPECT_TRUE(is_near(s2a, M1by1 {5.}));
  //
  EXPECT_TRUE(is_near(rank_update(DiagonalMatrix {3., 3}, DiagonalMatrix {2., 2}, 4), make_eigen_matrix<double, 2, 2>(5., 0, 0, 5.)));
  EXPECT_TRUE(is_near(rank_update(DiagonalMatrix {3., 3}, 2 * M22::Identity(), 4), make_dense_writable_matrix_from<M22>(5., 0, 0, 5.)));
  EXPECT_TRUE(is_near(rank_update(DiagonalMatrix {3., 3}, make_dense_writable_matrix_from<M22>(2, 0, 0, 2.), 4), make_dense_writable_matrix_from<M22>(5., 0, 0, 5.)));
  //
  EXPECT_TRUE(is_near(rank_update(M1by1 {3.}, M1by1 {2.}, 4), M1by1 {5.}));
}


TEST(special_matrices, SelfAdjointMatrix_rank_update_1x1)
{
  auto sl1 = L22 {9., 3, 3, 10};
  rank_update(sl1, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4);
  EXPECT_TRUE(is_near(sl1, mat22(25., 11, 11, 30)));
  auto su1 = U22 {9., 3, 3, 10};
  rank_update(su1, make_dense_writable_matrix_from<M22>(2, 1, 0, 2), 4);
  EXPECT_TRUE(is_near(su1, mat22(29., 11, 11, 26)));
  //
  auto m11 = M11(9);
  auto m10_1 = m11;
  auto m01_1 = m11;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
  auto m00_11 = m11;
#pragma GCC diagnostic pop

  const auto m11_2 = M11{2};
  const auto m11_25 = M11(25);

  EXPECT_TRUE(is_near(rank_update_self_adjoint(M10 {m11}, m11_2, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M01 {m11}, m11_2, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M00 {m11}, m11_2, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M10 {m11}, M10 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M01 {m11}, M10 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M00 {m11}, M10 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M10 {m11}, M01 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M01 {m11}, M01 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M00 {m11}, M01 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M10 {m11}, M00 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M01 {m11}, M00 {m11_2}, 4), m11_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M00 {m11}, M00 {m11_2}, 4), m11_25));

  rank_update_self_adjoint(m10_1, m11_2, 4); EXPECT_TRUE(is_near(m10_1, m11_25));
  rank_update_self_adjoint(m01_1, m11_2, 4); EXPECT_TRUE(is_near(m01_1, m11_25));
  rank_update_self_adjoint(m00_11, m11_2, 4); EXPECT_TRUE(is_near(m00_11, m11_25));
}


TEST(special_matrices, rank_update_triangular_1x1)
{
  auto m11 = M11(3);
  auto m10_1 = M10 {m11};
  auto m01_1 = M01 {m11};
  auto m00_11 = M00 {m11};

  const auto m11_2 = M11{2};
  const auto m11_5 = M11(5);

  EXPECT_TRUE(is_near(rank_update_triangular(M10 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M01 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M00 {m11}, m11_2, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M10 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M01 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M00 {m11}, M10 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M10 {m11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M01 {m11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M00 {m11}, M01 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M10 {m11}, M00 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M01 {m11}, M00 {m11_2}, 4), m11_5));
  EXPECT_TRUE(is_near(rank_update_triangular(M00 {m11}, M00 {m11_2}, 4), m11_5));

  rank_update_triangular(m10_1, m11_2, 4); EXPECT_TRUE(is_near(m10_1, m11_5));
  rank_update_triangular(m01_1, m11_2, 4); EXPECT_TRUE(is_near(m01_1, m11_5));
  rank_update_triangular(m00_11, m11_2, 4); EXPECT_TRUE(is_near(m00_11, m11_5));
}


TEST(eigen3, rank_update_diagonal)
{
  // a is zero, u is diagonal:

  const auto z22 = M22::Identity() - M22::Identity(); static_assert(zero_matrix<decltype(z22)>);
  const auto m22_16 = make_dense_writable_matrix_from<M22>(10.5, 0.5, 0.5, 0.5);

  const Eigen::DiagonalMatrix<double, 2> d22_2 {make_dense_writable_matrix_from<M21>(2, 2)}; static_assert(diagonal_matrix<decltype(d22_2)>);
  const Eigen::DiagonalMatrix<double, 2> d22_16 {make_dense_writable_matrix_from<M21>(16, 16)}; static_assert(diagonal_matrix<decltype(d22_16)>);

  EXPECT_TRUE(is_near(rank_update_self_adjoint(z22, d22_2, 4), d22_16));

  // a and u are both diagonal:

  DiagonalMatrix d22_3 {make_dense_writable_matrix_from<M21>(3, 3)};
  const Eigen::DiagonalMatrix<double, 2> d22_19 {make_dense_writable_matrix_from<M21>(19, 19)};
  EXPECT_TRUE(is_near(rank_update_self_adjoint(std::as_const(d22_3), d22_2, 4), d22_19));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(d22_3, d22_2, 4), d22_19));
  EXPECT_TRUE(is_near(d22_3, d22_19));

  // ----

  auto m21_33 = make_dense_writable_matrix_from<M21>(3, 3);
  auto m20_33 = make_dense_writable_matrix_from<M20>(3, 3);
  auto m01_33 = make_dense_writable_matrix_from<M01>(3, 3);
  auto m00_33 = make_dense_writable_matrix_from<M00>(std::tuple {2, 1}, 3, 3);

  auto m22_2012 = make_dense_writable_matrix_from<M22>(2, 0, 1, 2);
  auto m22_198823 = make_dense_writable_matrix_from<M22>(19, 8, 8, 23);
  auto m22_258829_sqrt = make_dense_writable_matrix_from<M22>(5, 0, 1.6, std::sqrt(26.44));

  EXPECT_TRUE(is_near(rank_update_self_adjoint(Eigen::DiagonalMatrix<double, 2>(m21_33), m22_2012, 4), m22_198823));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(Eigen::DiagonalMatrix<double, Eigen::Dynamic>(m21_33), m22_2012, 4), m22_198823));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(Eigen::DiagonalWrapper {m21_33}, m22_2012, 4), m22_198823));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(Eigen::DiagonalWrapper {m20_33}, m22_2012, 4), m22_198823));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(Eigen::DiagonalWrapper {m01_33}, m22_2012, 4), m22_198823));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(Eigen::DiagonalWrapper {m00_33}, m22_2012, 4), m22_198823));

  EXPECT_TRUE(is_near(rank_update_triangular(Eigen::DiagonalMatrix<double, 2>(m21_33), m22_2012, 4), m22_258829_sqrt));
  EXPECT_TRUE(is_near(rank_update_triangular(Eigen::DiagonalMatrix<double, Eigen::Dynamic>(m21_33), m22_2012, 4), m22_258829_sqrt));
  EXPECT_TRUE(is_near(rank_update_triangular(Eigen::DiagonalWrapper {m21_33}, m22_2012, 4), m22_258829_sqrt));
  EXPECT_TRUE(is_near(rank_update_triangular(Eigen::DiagonalWrapper {m20_33}, m22_2012, 4), m22_258829_sqrt));
  EXPECT_TRUE(is_near(rank_update_triangular(Eigen::DiagonalWrapper {m01_33}, m22_2012, 4), m22_258829_sqrt));
  EXPECT_TRUE(is_near(rank_update_triangular(Eigen::DiagonalWrapper {m00_33}, m22_2012, 4), m22_258829_sqrt));

  static_assert(not std::is_lvalue_reference_v<decltype(rank_update_self_adjoint(Eigen::DiagonalMatrix<double, 2>(m21_33), m22_2012, 4))>);
  static_assert(not std::is_lvalue_reference_v<decltype(rank_update_self_adjoint(Eigen::DiagonalMatrix<double, Eigen::Dynamic>(m21_33), m22_2012, 4))>);
  static_assert(not std::is_lvalue_reference_v<decltype(rank_update_self_adjoint(Eigen::DiagonalWrapper {m00_33}, m22_2012, 4))>);

  static_assert(not std::is_lvalue_reference_v<decltype(rank_update_triangular(Eigen::DiagonalMatrix<double, 2>(m21_33), m22_2012, 4))>);
  static_assert(not std::is_lvalue_reference_v<decltype(rank_update_triangular(Eigen::DiagonalMatrix<double, Eigen::Dynamic>(m21_33), m22_2012, 4))>);
  static_assert(not std::is_lvalue_reference_v<decltype(rank_update_triangular(Eigen::DiagonalWrapper {m00_33}, m22_2012, 4))>);

  const auto cd22_3 = M22::Identity() + M22::Identity() + M22::Identity();
  const auto cd20_3 = Eigen::Replicate<decltype(cd22_3), 1, Eigen::Dynamic> {cd22_3, 1, 1};
  const auto cd02_3 = Eigen::Replicate<decltype(cd22_3), Eigen::Dynamic, 1> {cd22_3, 1, 1};
  const auto cd00_3 = Eigen::Replicate<decltype(cd22_3), Eigen::Dynamic, Eigen::Dynamic> {cd22_3, 1, 1};

  EXPECT_TRUE(is_near(rank_update_self_adjoint(cd22_3.triangularView<Eigen::Lower>(), m22_2012, 4), m22_198823));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(cd20_3.triangularView<Eigen::Upper>(), m22_2012, 4), m22_198823));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(cd02_3.triangularView<Eigen::Lower>(), m22_2012, 4), m22_198823));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(cd00_3.triangularView<Eigen::Upper>(), m22_2012, 4), m22_198823));

  EXPECT_TRUE(is_near(rank_update_triangular(cd22_3.selfadjointView<Eigen::Lower>(), m22_2012, 4), m22_258829_sqrt));
  EXPECT_TRUE(is_near(rank_update_triangular(cd20_3.selfadjointView<Eigen::Upper>(), m22_2012, 4), m22_258829_sqrt));
  EXPECT_TRUE(is_near(rank_update_triangular(cd02_3.selfadjointView<Eigen::Lower>(), m22_2012, 4), m22_258829_sqrt));
  EXPECT_TRUE(is_near(rank_update_triangular(cd00_3.selfadjointView<Eigen::Upper>(), m22_2012, 4), m22_258829_sqrt));
}


TEST(eigen3, rank_update_self_adjoint)
{
  // ---

  const auto sl2 = L22 {9., 3, 3, 10};
  EXPECT_TRUE(is_near(rank_update(sl2, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4), mat22(25., 11, 11, 30)));
  const auto su2 = U22 {9., 3, 3, 10};
  EXPECT_TRUE(is_near(rank_update(su2, make_dense_writable_matrix_from<M22>(2, 1, 0, 2), 4), mat22(29., 11, 11, 26)));
  //
  EXPECT_TRUE(is_near(rank_update(L22 {9., 3, 3, 10}, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4), mat22(25., 11, 11, 30)));
  EXPECT_TRUE(is_near(rank_update(U22 {9., 3, 3, 10}, make_dense_writable_matrix_from<M22>(2, 1, 0, 2), 4), mat22(29., 11, 11, 26)));
  EXPECT_TRUE(is_near(rank_update(DM22 {9., 10}, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4), mat22(25., 8, 8, 30)));
  EXPECT_TRUE(is_near(rank_update(DD2 {9., 10}, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4), mat22(25., 8, 8, 30)));
  EXPECT_TRUE(is_near(rank_update(DL2 {9., 10}, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4), mat22(25., 8, 8, 30)));

  auto m22_2022 = make_dense_writable_matrix_from<M22>(2, 0, 2, 2);
  auto m22_25 = make_dense_writable_matrix_from<M22>(25, 25, 25, 50);

  auto m22 = make_dense_writable_matrix_from<M22>(9, 9, 9, 18);
  auto m20 = make_dense_writable_matrix_from<M20>(m22);
  auto m02 = make_dense_writable_matrix_from<M02>(m22);
  auto m00 = make_dense_writable_matrix_from<M00>(m22);

  rank_update_self_adjoint(m22.selfadjointView<Eigen::Lower>(), m22_2022, 4); EXPECT_TRUE(is_near(m22.selfadjointView<Eigen::Lower>(), m22_25));
  rank_update_self_adjoint(m20.selfadjointView<Eigen::Lower>(), m22_2022, 4); EXPECT_TRUE(is_near(m20.selfadjointView<Eigen::Lower>(), m22_25));
  rank_update_self_adjoint(m02.selfadjointView<Eigen::Lower>(), m22_2022, 4); EXPECT_TRUE(is_near(m02.selfadjointView<Eigen::Lower>(), m22_25));
  rank_update_self_adjoint(m00.selfadjointView<Eigen::Lower>(), m22_2022, 4); EXPECT_TRUE(is_near(m00.selfadjointView<Eigen::Lower>(), m22_25));

  m22 = make_dense_writable_matrix_from<M22>(9, 9, 9, 18);

  EXPECT_TRUE(is_near(rank_update_self_adjoint(M22{m22}.selfadjointView<Eigen::Lower>(), m22_2022, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M20{m22}.selfadjointView<Eigen::Lower>(), m22_2022, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M02{m22}.selfadjointView<Eigen::Lower>(), m22_2022, 4), m22_25));
  EXPECT_TRUE(is_near(rank_update_self_adjoint(M00{m22}.selfadjointView<Eigen::Lower>(), m22_2022, 4), m22_25));

  static_assert(hermitian_adapter<decltype(rank_update_self_adjoint(M22{m22}.selfadjointView<Eigen::Lower>(), m22_2022, 4)), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(rank_update_self_adjoint(M20{m22}.selfadjointView<Eigen::Lower>(), m22_2022, 4)), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(rank_update_self_adjoint(M02{m22}.selfadjointView<Eigen::Lower>(), m22_2022, 4)), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(rank_update_self_adjoint(M00{m22}.selfadjointView<Eigen::Lower>(), m22_2022, 4)), HermitianAdapterType::lower>);

  auto z22 = M22::Identity() - M22::Identity(); static_assert(zero_matrix<decltype(z22)>);

  EXPECT_TRUE(is_near(rank_update_self_adjoint(z22, m22_2022, 4), rank_update_self_adjoint(M22::Zero(), m22_2022, 4)));

  auto m22_2012 = make_dense_writable_matrix_from<M22>(2, 0, 1, 2);
  auto m20_2012 = M20 {m22_2012};
  auto m02_2012 = M02 {m22_2012};
  auto m00_2012 = M00 {m22_2012};

  auto m22_2102 = make_dense_writable_matrix_from<M22>(2, 1, 0, 2);
  auto m20_2102 = M20 {m22_2102};
  auto m02_2102 = M02 {m22_2102};
  auto m00_2102 = M00 {m22_2102};

  auto m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto m20_93310 = M20 {m22_93310};
  auto m02_93310 = M02 {m22_93310};
  auto m00_93310 = M00 {m22_93310};

  auto m22_29111126 = make_dense_writable_matrix_from<M22>(29, 11, 11, 26);
  auto m22_25111130 = make_dense_writable_matrix_from<M22>(25, 11, 11, 30);

  auto ru_93310_2102_4_rvalue = rank_update_self_adjoint(Eigen::SelfAdjointView<M22, Eigen::Upper> {m22_93310}, m22_2102, 4);
  EXPECT_TRUE(is_near(ru_93310_2102_4_rvalue, m22_29111126));
  EXPECT_TRUE(is_near(m22_93310.selfadjointView<Eigen::Upper>(), m22_29111126));
  static_assert(eigen_self_adjoint_expr<decltype(ru_93310_2102_4_rvalue)>);
  static_assert(hermitian_adapter<decltype(ru_93310_2102_4_rvalue), HermitianAdapterType::upper>);
  static_assert(std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru_93310_2102_4_rvalue)>>);

  m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);
  auto sa_93310_2012_4 = Eigen::SelfAdjointView<M22, Eigen::Lower> {m22_93310};

  auto ru_93310_2012_4_lvalue = rank_update_self_adjoint(sa_93310_2012_4, m22_2012, 4);
  EXPECT_TRUE(is_near(ru_93310_2012_4_lvalue, m22_25111130));
  EXPECT_TRUE(is_near(sa_93310_2012_4, m22_25111130));
  EXPECT_TRUE(is_near(m22_93310.template selfadjointView<Eigen::Lower>(), m22_25111130));
  static_assert(eigen_self_adjoint_expr<decltype(ru_93310_2012_4_lvalue)>);
  static_assert(hermitian_adapter<decltype(ru_93310_2012_4_lvalue), HermitianAdapterType::lower>);
  static_assert(std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru_93310_2012_4_lvalue)>>);

  m22_93310 = make_dense_writable_matrix_from<M22>(9, 3, 3, 10);

  auto ru_93310_2012_4_const_lvalue = rank_update_self_adjoint(std::as_const(sa_93310_2012_4), m22_2012, 4);
  EXPECT_TRUE(is_near(ru_93310_2012_4_const_lvalue, m22_25111130));
  EXPECT_TRUE(is_near(sa_93310_2012_4, make_dense_writable_matrix_from<M22>(9, 3, 3, 10)));
  static_assert(eigen_self_adjoint_expr<decltype(ru_93310_2012_4_const_lvalue)>);
  static_assert(not std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru_93310_2012_4_const_lvalue)>>);
}


TEST(eigen3, rank_update_triangular)
{
  auto sl1 = Lower {3., 0, 1, 3};
  rank_update(sl1, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4);
  EXPECT_TRUE(is_near(sl1, mat22(5., 0, 2.2, std::sqrt(25.16))));
  auto su1 = Upper {3., 1, 0, 3};
  rank_update(su1, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4);
  EXPECT_TRUE(is_near(su1, mat22(5., 2.2, 0, std::sqrt(25.16))));
  //
  const auto sl2 = Lower {3., 0, 1, 3};
  EXPECT_TRUE(is_near(rank_update(sl2, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4), mat22(5., 0, 2.2, std::sqrt(25.16))));
  const auto su2 = Upper {3., 1, 0, 3};
  EXPECT_TRUE(is_near(rank_update(su2, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4), mat22(5., 2.2, 0, std::sqrt(25.16))));
  //
  EXPECT_TRUE(is_near(rank_update(Lower {3., 0, 1, 3}, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4), mat22(5., 0, 2.2, std::sqrt(25.16))));
  EXPECT_TRUE(is_near(rank_update(Upper {3., 1, 0, 3}, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4), mat22(5., 2.2, 0, std::sqrt(25.16))));
  EXPECT_TRUE(is_near(rank_update(Diagonal {3., 3}, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4), mat22(5., 0, 1.6, std::sqrt(26.44))));
  EXPECT_TRUE(is_near(rank_update(Diagonal2 {3., 3}, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4), mat22(5., 0, 1.6, std::sqrt(26.44))));
  EXPECT_TRUE(is_near(rank_update(Diagonal3 {3., 3}, make_dense_writable_matrix_from<M22>(2, 0, 1, 2), 4), mat22(5., 0, 1.6, std::sqrt(26.44))));

  const auto m22_3033 = make_dense_writable_matrix_from<M22>(3, 0, 3, 3);
  const auto m20_3033 = make_dense_writable_matrix_from<M20>(m22_3033);
  const auto m02_3033 = make_dense_writable_matrix_from<M02>(m22_3033);
  const auto m00_3033 = make_dense_writable_matrix_from<M00>(m22_3033);

  const auto m22_3303 = make_dense_writable_matrix_from<M22>(3, 3, 0, 3);
  const auto m20_3303 = make_dense_writable_matrix_from<M20>(m22_3303);
  const auto m02_3303 = make_dense_writable_matrix_from<M02>(m22_3303);
  const auto m00_3303 = make_dense_writable_matrix_from<M00>(m22_3303);

  const auto m22_2022 = make_dense_writable_matrix_from<M22>(2, 0, 2, 2);
  const auto m22_5055 = make_dense_writable_matrix_from<M22>(5, 0, 5, 5);
  const auto m22_5505 = make_dense_writable_matrix_from<M22>(5, 5, 0, 5);

  auto m22 = m22_3033;
  auto m20 = m20_3033;
  auto m02 = m02_3033;
  auto m00 = m00_3033;

  rank_update_triangular(m22.triangularView<Eigen::Lower>(), m22_2022, 4); EXPECT_TRUE(is_near(m22, m22_5055));
  rank_update_triangular(m20.triangularView<Eigen::Lower>(), m22_2022, 4); EXPECT_TRUE(is_near(m20, m22_5055));
  rank_update_triangular(m02.triangularView<Eigen::Lower>(), m22_2022, 4); EXPECT_TRUE(is_near(m02, m22_5055));
  rank_update_triangular(m00.triangularView<Eigen::Lower>(), m22_2022, 4); EXPECT_TRUE(is_near(m00, m22_5055));
  static_assert(triangular_matrix<decltype(rank_update_triangular(m22.triangularView<Eigen::Lower>(), m22_2022, 4)), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(rank_update_triangular(m20.triangularView<Eigen::Lower>(), m22_2022, 4)), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(rank_update_triangular(m02.triangularView<Eigen::Lower>(), m22_2022, 4)), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(rank_update_triangular(m00.triangularView<Eigen::Lower>(), m22_2022, 4)), TriangleType::lower>);

  EXPECT_TRUE(is_near(rank_update_triangular(M22 {m22_3033}.triangularView<Eigen::Lower>(), m22_2022, 4), m22_5055));
  EXPECT_TRUE(is_near(rank_update_triangular(M20 {m22_3033}.triangularView<Eigen::Lower>(), m22_2022, 4), m22_5055));
  EXPECT_TRUE(is_near(rank_update_triangular(M02 {m22_3033}.triangularView<Eigen::Lower>(), m22_2022, 4), m22_5055));
  EXPECT_TRUE(is_near(rank_update_triangular(M00 {m22_3033}.triangularView<Eigen::Lower>(), m22_2022, 4), m22_5055));

  static_assert(triangular_matrix<decltype(rank_update_triangular(M22 {m22_3033}.triangularView<Eigen::Lower>(), m22_2022, 4)), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(rank_update_triangular(M20 {m22_3033}.triangularView<Eigen::Lower>(), m22_2022, 4)), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(rank_update_triangular(M02 {m22_3033}.triangularView<Eigen::Lower>(), m22_2022, 4)), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(rank_update_triangular(M00 {m22_3033}.triangularView<Eigen::Lower>(), m22_2022, 4)), TriangleType::lower>);

  const auto constm00 {m00_3033};
  static_assert(triangular_matrix<decltype(rank_update_triangular(constm00.triangularView<Eigen::Lower>(), m22_2022, 4)), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(rank_update_triangular(constm00.triangularView<Eigen::Lower>(), M20{m22_2022}, 4)), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(rank_update_triangular(constm00.triangularView<Eigen::Lower>(), M02{m22_2022}, 4)), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(rank_update_triangular(constm00.triangularView<Eigen::Lower>(), M00{m22_2022}, 4)), TriangleType::lower>);

  auto z22 = M22::Identity() - M22::Identity(); static_assert(zero_matrix<decltype(z22)>);

  EXPECT_TRUE(is_near(rank_update_triangular(z22, m22_2022, 4), make_dense_writable_matrix_from<M22>(4, 0, 4, 4)));

  m22 = m22_3033;
  m20 = m20_3033;
  m02 = m02_3033;
  m00 = m00_3033;

  auto tl22_lvalue = Eigen::TriangularView<M22, Eigen::Lower> {m22};
  auto tl20_lvalue = Eigen::TriangularView<M20, Eigen::Lower> {m20};
  auto tl02_lvalue = Eigen::TriangularView<M02, Eigen::Lower> {m02};
  auto tl00_lvalue = Eigen::TriangularView<M00, Eigen::Lower> {m00};

  auto ru22_lvalue = rank_update_triangular(tl22_lvalue, m22_2022, 4); EXPECT_TRUE(is_near(ru22_lvalue, m22_5055));
  auto ru20_lvalue = rank_update_triangular(tl20_lvalue, m22_2022, 4); EXPECT_TRUE(is_near(ru20_lvalue, m22_5055));
  auto ru02_lvalue = rank_update_triangular(tl02_lvalue, m22_2022, 4); EXPECT_TRUE(is_near(ru02_lvalue, m22_5055));
  auto ru00_lvalue = rank_update_triangular(tl00_lvalue, m22_2022, 4); EXPECT_TRUE(is_near(ru00_lvalue, m22_5055));

  EXPECT_TRUE(is_near(tl22_lvalue, m22_5055));
  EXPECT_TRUE(is_near(tl20_lvalue, m22_5055));
  EXPECT_TRUE(is_near(tl02_lvalue, m22_5055));
  EXPECT_TRUE(is_near(tl00_lvalue, m22_5055));

  EXPECT_TRUE(is_near(m22, m22_5055));
  EXPECT_TRUE(is_near(m20, m22_5055));
  EXPECT_TRUE(is_near(m02, m22_5055));
  EXPECT_TRUE(is_near(m00, m22_5055));

  static_assert(triangular_matrix<decltype(ru22_lvalue)>); static_assert(eigen_triangular_expr<decltype(ru22_lvalue)>);
  static_assert(triangular_matrix<decltype(ru20_lvalue)>); static_assert(eigen_triangular_expr<decltype(ru20_lvalue)>);
  static_assert(triangular_matrix<decltype(ru02_lvalue)>); static_assert(eigen_triangular_expr<decltype(ru02_lvalue)>);
  static_assert(triangular_matrix<decltype(ru00_lvalue)>); static_assert(eigen_triangular_expr<decltype(ru00_lvalue)>);

  static_assert(std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru22_lvalue)>>);
  static_assert(std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru20_lvalue)>>);
  static_assert(std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru02_lvalue)>>);
  static_assert(std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru00_lvalue)>>);

  m22 = m22_3303;
  auto ru22_rvalue = rank_update_triangular(Eigen::TriangularView<M22, Eigen::Upper> {m22}, m22_2022, 4);
  EXPECT_TRUE(is_near(ru22_rvalue, m22_5505));
  static_assert(eigen_triangular_expr<decltype(ru22_rvalue)>);
  static_assert(triangular_matrix<decltype(ru22_rvalue), TriangleType::upper>);
  static_assert(std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru22_rvalue)>>);

  m22 = m22_3033;
  auto t22_lvalue = Eigen::TriangularView<M22, Eigen::Lower> {m22};
  auto ru22_const_lvalue = rank_update_triangular(std::as_const(t22_lvalue), m22_2022, 4);
  EXPECT_TRUE(is_near(ru22_const_lvalue, m22_5055));
  EXPECT_TRUE(is_near(t22_lvalue, m22_3033));
  static_assert(eigen_triangular_expr<decltype(ru22_const_lvalue)>);
  static_assert(triangular_matrix<decltype(ru22_const_lvalue), TriangleType::lower>);
  static_assert(not std::is_lvalue_reference_v<nested_matrix_of_t<decltype(ru22_const_lvalue)>>);
}

