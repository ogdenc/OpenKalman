/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests relating to Eigen3::SelfAdjointMatrix.
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
  using M2x = eigen_matrix_t<double, 2, dynamic_size>;
  using Mx2 = eigen_matrix_t<double, dynamic_size, 2>;
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M1x = eigen_matrix_t<double, 1, dynamic_size>;
  using Mx1 = eigen_matrix_t<double, dynamic_size, 1>;
  using Mxx = eigen_matrix_t<double, dynamic_size, dynamic_size>;

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

TEST(special_matrices, SelfAdjointMatrix_static_checks)
{
  static_assert(writable<SAl<M22>>);
  static_assert(writable<SAl<M22&>>);
  static_assert(not writable<SAl<const M22>>);
  static_assert(not writable<SAl<const M22&>>);
  
  static_assert(writable<SAu<M22>>);
  static_assert(writable<SAu<M22&>>);
  static_assert(not writable<SAu<const M22>>);
  static_assert(not writable<SAu<const M22&>>);

  static_assert(hermitian_matrix<L22>);
  static_assert(hermitian_matrix<L20>);
  static_assert(hermitian_matrix<L02>);
  static_assert(hermitian_matrix<L00>);
  static_assert(hermitian_matrix<U00>);
  static_assert(hermitian_matrix<DM00>);

  static_assert(hermitian_adapter<L22, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<L20, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<L02, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<L00, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<U00, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<DM00, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<DM00, HermitianAdapterType::lower>);

  static_assert(square_matrix<L22>);
  static_assert(square_matrix<L20>);
  static_assert(square_matrix<L02>);
  static_assert(square_matrix<L00>);
  static_assert(square_matrix<U00>);
  static_assert(square_matrix<DM00>);

  static_assert(one_by_one_matrix<SelfAdjointMatrix<M11, TriangleType::upper>>);
  static_assert(one_by_one_matrix<SelfAdjointMatrix<M1x, TriangleType::upper>>);
  static_assert(one_by_one_matrix<SelfAdjointMatrix<Mx1, TriangleType::upper>>);
  static_assert(not one_by_one_matrix<U00>);
  static_assert(one_by_one_matrix<U00, Likelihood::maybe>);

  // \todo fill in other traits

  static_assert(not OpenKalman::internal::has_const<SAl<M22>>::value);
  static_assert(OpenKalman::internal::has_const<SAl<const M22>>::value);
  static_assert(maybe_has_same_shape_as<SAl<M22>, ZeroMatrix<eigen_matrix_t<double, 2, 2>>>);
  static_assert(not maybe_has_same_shape_as<SAl<M22>, ZeroMatrix<eigen_matrix_t<double, 3, 3>>>);

  static_assert(modifiable<SAl<M22>, ConstantAdapter<eigen_matrix_t<double, 2, 2>, 7>>);
  static_assert(modifiable<SAu<M22>, ConstantAdapter<eigen_matrix_t<double, 2, 2>, 7>>);
  
  static_assert(modifiable<SAl<M22>, ZeroMatrix<eigen_matrix_t<double, 2, 2>>>);
  static_assert(modifiable<SAu<M22>, ZeroMatrix<eigen_matrix_t<double, 2, 2>>>);
  static_assert(modifiable<SAl<M22>, Eigen3::IdentityMatrix<M22>>);
  static_assert(modifiable<SAu<M22>, Eigen3::IdentityMatrix<M22>>);
  static_assert(modifiable<SAl<M22>, D<M21>>);
  static_assert(modifiable<SAu<M22>, D<M21>>);
  static_assert(not modifiable<SAl<M22>, M22>);
  static_assert(not modifiable<SAu<M22>, M22>);
  static_assert(modifiable<SAl<M22>, SAl<M22>>);
  static_assert(modifiable<SAu<M22>, SAu<M22>>);
  static_assert(not modifiable<SAl<M22::ConstantReturnType>, SAl<M22::ConstantReturnType>>);
  static_assert(not modifiable<SAu<decltype(M22::Identity() * 2)>, SAu<decltype(M22::Identity() * 2)>>);
  static_assert(modifiable<SAl<M22>, const SAl<M22>>);
  static_assert(modifiable<SAu<M22>, const SAu<M22>>);
  static_assert(modifiable<SAl<M22>, SAl<const M22>>);
  static_assert(modifiable<SAu<M22>, SAu<const M22>>);
  static_assert(not modifiable<SAl<const M22>, SAl<M22>>);
  static_assert(not modifiable<SAu<const M22>, SAu<M22>>);
  static_assert(not modifiable<SAl<decltype(M22::Constant(9))>, M22>);
  static_assert(not modifiable<SAl<M22>, Tl<M22>>);
  static_assert(not modifiable<SAu<M22>, Tu<M22>>);
  static_assert(not modifiable<Tl<M22>, SAl<M22>>);
  static_assert(not modifiable<Tu<M22>, SAu<M22>>);
  static_assert(not modifiable<D<M21>, SAu<M22>>);
  static_assert(modifiable<SAl<M22>&, SAl<M22>>);
  static_assert(modifiable<SAu<M22>&, SAu<M22>>);
  static_assert(modifiable<SAl<M22&>, SAl<M22>>);
  static_assert(modifiable<SAu<M22&>, SAu<M22>>);
  static_assert(not modifiable<SAl<M22&>, M22>);
  static_assert(not modifiable<SAu<M22&>, M22>);
  static_assert(not modifiable<const SAl<M22>&, SAl<M22>>);
  static_assert(not modifiable<const SAu<M22>&, SAu<M22>>);
  static_assert(not modifiable<SAl<const M22&>, SAl<M22>>);
  static_assert(not modifiable<SAu<const M22&>, SAu<M22>>);
  static_assert(not modifiable<SAl<const M22>&, SAl<M22>>);
  static_assert(not modifiable<SAu<const M22>&, SAu<M22>>);
}


TEST(special_matrices, SelfAdjointMatrix_class)
{
  D2 d2 {9, 9};
  //
  L22 l1;
  l1 << 9, 7, 3, 10;
  EXPECT_TRUE(is_near(l1.nested_matrix(), mat22(9, 7, 3, 10)));
  EXPECT_TRUE(is_near(l1, m_93310));
  U22 u1;
  u1 << 9, 3, 7, 10;
  EXPECT_TRUE(is_near(u1.nested_matrix(), mat22(9, 3, 7, 10)));
  EXPECT_TRUE(is_near(u1, m_93310));
  DM22 d1;
  d1 << 9, 10;
  EXPECT_TRUE(is_near(d1.nested_matrix(), mat22(9, 0, 0, 10)));
  EXPECT_TRUE(is_near(d1, mat22(9, 0, 0, 10)));
  DD2 d1b;
  d1b << 9, 10;
  EXPECT_TRUE(is_near(d1b.nested_matrix(), mat22(9, 0, 0, 10)));
  EXPECT_TRUE(is_near(d1b, mat22(9, 0, 0, 10)));
  DL2 d1c;
  d1c << 9, 10;
  EXPECT_TRUE(is_near(d1c.nested_matrix(), mat22(9, 0, 0, 10)));
  EXPECT_TRUE(is_near(d1c, mat22(9, 0, 0, 10)));
  d1c.template triangularView<Eigen::Lower>() = mat22(7, 5, 6, 12);
  EXPECT_TRUE(is_near(d1c, mat22(7, 0, 0, 12)));
  //
  L22 l2 {mat22(9, 3, 3, 10)};
  EXPECT_TRUE(is_near(l1, m_93310));
  U22 u2 {mat22(9, 3, 3, 10)};
  EXPECT_TRUE(is_near(u1, m_93310));
  //
  EXPECT_TRUE(is_near(L22(D2 {3., 4}), mat22(3, 0, 0, 4)));
  EXPECT_TRUE(is_near(U22(D2 {3., 4}), mat22(3, 0, 0, 4)));
  //
  EXPECT_TRUE(is_near(L22(make_zero_matrix_like<M22>()), M22::Zero()));
  EXPECT_TRUE(is_near(U22(make_zero_matrix_like<M22>()), M22::Zero()));
  //
  EXPECT_EQ(l2.rows(), 2);
  EXPECT_EQ(l2.cols(), 2);
  EXPECT_TRUE(is_near(make_zero_matrix_like<L22>(), M22::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like<U22>(), M22::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<L22>(), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<U22>(), M22::Identity()));
  //
  L22 l3(l2); // copy constructor
  EXPECT_TRUE(is_near(l3, m_93310));
  U22 u3(u2); // copy constructor
  EXPECT_TRUE(is_near(u3, m_93310));
  //
  L22 l4 {L22{9, 3, 3, 10}}; // move constructor
  EXPECT_TRUE(is_near(l4, m_93310));
  U22 u4 {U22{9, 3, 3, 10}}; // move constructor
  EXPECT_TRUE(is_near(u4, m_93310));
  //
  L22 l5 = SelfAdjointMatrix<decltype(M22::Zero()), TriangleType::lower>(M22::Zero()); // compatible sa-matrix
  EXPECT_TRUE(is_near(l5, M22::Zero()));
  U22 u5 = SelfAdjointMatrix<decltype(M22::Zero()), TriangleType::upper>(M22::Zero()); // compatible sa-matrix
  EXPECT_TRUE(is_near(u5, M22::Zero()));
  //
  L22 l6 = SelfAdjointMatrix<decltype(M22::Zero()), TriangleType::upper>(M22::Zero()); // opposite sa-matrix
  EXPECT_TRUE(is_near(l6, M22::Zero()));
  U22 u6 = SelfAdjointMatrix<decltype(M22::Zero()), TriangleType::lower>(M22::Zero()); // opposite sa-matrix
  EXPECT_TRUE(is_near(u6, M22::Zero()));
  //
  L22 l7 {m_93310.selfadjointView<Eigen::Lower>()};
  EXPECT_TRUE(is_near(l7, m_93310));
  U22 u7 {m_93310.selfadjointView<Eigen::Upper>()};
  EXPECT_TRUE(is_near(u7, m_93310));
  //
  L22 l8 = {m_93310.selfadjointView<Eigen::Upper>()};
  EXPECT_TRUE(is_near(l8, m_93310));
  U22 u8 = {m_93310.selfadjointView<Eigen::Lower>()};
  EXPECT_TRUE(is_near(u8, m_93310));
  //
  L22 l9 = DM22 {9., 10}; // Construct from a diagonal sa-matrix.
  EXPECT_TRUE(is_near(l9, mat22(9, 0, 0, 10)));
  U22 u9 = DM22 {9., 10};
  EXPECT_TRUE(is_near(u9, mat22(9, 0, 0, 10)));
  DM22 d9 = DM22 {9, 10};
  EXPECT_TRUE(is_near(d9, mat22(9, 0, 0, 10)));
  DM22 d9_b = DD2 {9, 10};
  EXPECT_TRUE(is_near(d9_b, mat22(9, 0, 0, 10)));
  DM22 d9_c = DD2 {9, 10};
  EXPECT_TRUE(is_near(d9_c, mat22(9, 0, 0, 10)));
  DD2 d9b = DM22 {9, 10};
  EXPECT_TRUE(is_near(d9b, mat22(9, 0, 0, 10)));
  DD2 d9b_b = DD2 {9, 10};
  EXPECT_TRUE(is_near(d9b_b, mat22(9, 0, 0, 10)));
  DD2 d9b_c = DL2 {9, 10};
  EXPECT_TRUE(is_near(d9b_c, mat22(9, 0, 0, 10)));
  DL2 d9c = DM22 {9, 10};
  EXPECT_TRUE(is_near(d9c, mat22(9, 0, 0, 10)));
  DL2 d9c_b = DD2 {9, 10};
  EXPECT_TRUE(is_near(d9c_b, mat22(9, 0, 0, 10)));
  DL2 d9c_c = DL2 {9, 10};
  EXPECT_TRUE(is_near(d9c_c, mat22(9, 0, 0, 10)));
  //
  L22 l10 {9, 3, 3, 10}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(l10, m_93310));
  U22 u10 {9, 3, 3, 10}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(u10, m_93310));
  DM22 d10 {9, 10}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d10, mat22(9, 0, 0, 10)));
  DD2 d10b {9, 10}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d10b, mat22(9, 0, 0, 10)));
  DL2 d10c {9, 10}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d10c, mat22(9, 0, 0, 10)));
  //
  l3 = l5; // copy assignment
  EXPECT_TRUE(is_near(l3, M22::Zero()));
  u3 = u5; // copy assignment
  EXPECT_TRUE(is_near(u3, M22::Zero()));
  //
  l5 = L22 {9., 3, 3, 10}; // move assignment
  EXPECT_TRUE(is_near(l5, m_93310));
  u5 = U22 {9., 3, 3, 10}; // move assignment
  EXPECT_TRUE(is_near(u5, m_93310));
  //
  l2 = SelfAdjointMatrix<decltype(M22::Zero()), TriangleType::lower>(M22::Zero()); // copy assignment from compatible sa-matrix
  EXPECT_TRUE(is_near(l2, M22::Zero()));
  u2 = SelfAdjointMatrix<decltype(M22::Zero()), TriangleType::upper>(M22::Zero()); // copy assignment from compatible sa-matrix
  EXPECT_TRUE(is_near(u2, M22::Zero()));
  //
  l2 = U22 {9., 3, 3, 10}; // copy assignment from opposite sa-matrix;
  EXPECT_TRUE(is_near(l2, m_93310));
  u2 = L22 {9., 3, 3, 10}; // copy assignment from opposite sa-matrix;
  EXPECT_TRUE(is_near(u2, m_93310));
  //
  l9 = DM22 {4., 4}; // copy from a diagonal sa-matrix
  EXPECT_TRUE(is_near(l9, mat22(4, 0, 0, 4)));
  l4 = l9;
  EXPECT_TRUE(is_near(l4, mat22(4, 0, 0, 4)));
  l9 = DD2 {5., 5};
  EXPECT_TRUE(is_near(l9, mat22(5, 0, 0, 5)));
  l9 = DL2 {6., 6};
  EXPECT_TRUE(is_near(l9, mat22(6, 0, 0, 6)));
  u9 = DM22 {4., 4};
  EXPECT_TRUE(is_near(u9, mat22(4, 0, 0, 4)));
  u4 = u9;
  EXPECT_TRUE(is_near(u4, mat22(4, 0, 0, 4)));
  u9 = DD2 {5., 5};
  EXPECT_TRUE(is_near(u9, mat22(5, 0, 0, 5)));
  u9 = DL2 {6., 6};
  EXPECT_TRUE(is_near(u9, mat22(6, 0, 0, 6)));
  d9 = DM22 {4., 4};
  EXPECT_TRUE(is_near(d9, mat22(4, 0, 0, 4)));
  d9 = DD2 {5., 5};
  EXPECT_TRUE(is_near(d9, mat22(5, 0, 0, 5)));
  d9 = DL2 {6., 6};
  EXPECT_TRUE(is_near(d9, mat22(6, 0, 0, 6)));
  d9b_b = DM22 {7., 7};
  d9b = d9b_b;
  EXPECT_TRUE(is_near(d9b, mat22(7, 0, 0, 7)));
  l4 = d9b;
  EXPECT_TRUE(is_near(l4, mat22(7, 0, 0, 7)));
  u4 = d9b;
  EXPECT_TRUE(is_near(u4, mat22(7, 0, 0, 7)));
  d9b = DM22 {4., 4};
  EXPECT_TRUE(is_near(d9b, mat22(4, 0, 0, 4)));
  d9b = DD2 {5., 5};
  EXPECT_TRUE(is_near(d9b, mat22(5, 0, 0, 5)));
  d9b = DL2 {6., 6};
  EXPECT_TRUE(is_near(d9b, mat22(6, 0, 0, 6)));
  d9c = d9b_b;
  EXPECT_TRUE(is_near(d9c, mat22(7, 0, 0, 7)));
  l4 = d9c;
  EXPECT_TRUE(is_near(l4, mat22(7, 0, 0, 7)));
  u4 = d9c;
  EXPECT_TRUE(is_near(u4, mat22(7, 0, 0, 7)));
  d9c = DM22 {4., 4};
  EXPECT_TRUE(is_near(d9c, mat22(4, 0, 0, 4)));
  d9c = DD2 {5., 5};
  EXPECT_TRUE(is_near(d9c, mat22(5, 0, 0, 5)));
  d9c = DL2 {6., 6};
  EXPECT_TRUE(is_near(d9c, mat22(6, 0, 0, 6)));
  //
  l3 = make_zero_matrix_like<M22>();
  EXPECT_TRUE(is_near(l3, make_zero_matrix_like<M22>()));
  u3 = make_zero_matrix_like<M22>();
  EXPECT_TRUE(is_near(u3, make_zero_matrix_like<M22>()));
  l3 = make_identity_matrix_like<M22>();
  EXPECT_TRUE(is_near(l3, make_identity_matrix_like<M22>()));
  u3 = make_identity_matrix_like<M22>();
  EXPECT_TRUE(is_near(u3, make_identity_matrix_like<M22>()));
  //
  auto ma = mat22(9, 3, 3, 10);
  auto sa1 = ma.selfadjointView<Eigen::Lower>();
  auto sa2 = ma.selfadjointView<Eigen::Upper>();
  l2 = sa1; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(l2, m_93310));
  u2 = sa1; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(u2, m_93310));
  //
  l3 = sa2; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(l3, m_93310));
  u3 = sa2; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(u3, m_93310));
  //
  l4 = ma.selfadjointView<Eigen::Lower>(); // assign from rvalue of TriangularBase derived object
  EXPECT_TRUE(is_near(l4, m_93310));
  ma = mat22(9, 3, 3, 10);
  u4 = ma.selfadjointView<Eigen::Upper>(); // assign from rvalue of TriangularBase derived object
  EXPECT_TRUE(is_near(u4, m_93310));
  //
  l4 = M22::Zero();
  EXPECT_TRUE(is_near(l4, M22::Zero()));
  u4 = M22::Zero();
  EXPECT_TRUE(is_near(u4, M22::Zero()));
  //
  l4 = M22::Identity();
  EXPECT_TRUE(is_near(l4, M22::Identity()));
  u4 = M22::Identity();
  EXPECT_TRUE(is_near(u4, M22::Identity()));
  //
  l4 = d2;
  EXPECT_TRUE(is_near(l4, d2));
  u4 = d2;
  EXPECT_TRUE(is_near(u4, d2));
  //
  l4 = {9., 3, 3, 10}; // assign from a list of scalars
  EXPECT_TRUE(is_near(l4, m_93310));
  u4 = {9., 3, 3, 10}; // assign from a list of scalars
  EXPECT_TRUE(is_near(u4, m_93310));
  //
  l1 += L22 {9., 3, 3, 10};
  EXPECT_TRUE(is_near(l1, mat22(18., 6, 6, 20)));
  u1 += U22 {9., 3, 3, 10};
  EXPECT_TRUE(is_near(u1, mat22(18., 6, 6, 20)));
  //
  l1 -= U22 {9., 3, 3, 10};
  EXPECT_TRUE(is_near(l1, mat22(9., 3, 3, 10)));
  u1 -= L22 {9., 3, 3, 10};
  EXPECT_TRUE(is_near(u1, mat22(9., 3, 3, 10)));
  //
  l1 += U22 {9., 3, 3, 10};
  EXPECT_TRUE(is_near(l1, mat22(18., 6, 6, 20)));
  u1 += L22 {9., 3, 3, 10};
  EXPECT_TRUE(is_near(u1, mat22(18., 6, 6, 20)));
  //
  l1 -= L22 {9., 3, 3, 10};
  EXPECT_TRUE(is_near(l1, mat22(9., 3, 3, 10)));
  u1 -= U22 {9., 3, 3, 10};
  EXPECT_TRUE(is_near(u1, mat22(9., 3, 3, 10)));
  //
  l1 *= 3;
  EXPECT_TRUE(is_near(l1, mat22(27., 9, 9, 30)));
  u1 *= 3;
  EXPECT_TRUE(is_near(u1, mat22(27., 9, 9, 30)));
  //
  l1 /= 3;
  EXPECT_TRUE(is_near(l1, mat22(9., 3, 3, 10)));
  u1 /= 3;
  EXPECT_TRUE(is_near(u1, mat22(9., 3, 3, 10)));
}


TEST(special_matrices, SelfAdjointMatrix_subscripts)
{
  static_assert(element_gettable<L22, 2>);
  static_assert(not element_gettable<L22, 1>);
  static_assert(element_gettable<U22, 2>);
  static_assert(not element_gettable<U22, 1>);
  static_assert(element_gettable<DM22, 2>);
  static_assert(element_gettable<DM22, 1>);
  static_assert(element_gettable<DD2, 2>);
  static_assert(element_gettable<DD2, 1>);
  static_assert(element_gettable<DL2, 2>);
  static_assert(element_gettable<DL2, 1>);

  static_assert(element_settable<L22&, 2>);
  static_assert(not element_settable<L22&, 1>);
  static_assert(element_settable<U22&, 2>);
  static_assert(not element_settable<U22&, 1>);
  static_assert(element_settable<DM22&, 2>);
  static_assert(element_settable<DM22&, 1>);
  static_assert(element_settable<DD2&, 2>);
  static_assert(element_settable<DD2&, 1>);
  static_assert(element_settable<DL2&, 2>);
  static_assert(element_settable<DL2&, 1>);

  static_assert(not element_settable<const L22&, 2>);
  static_assert(not element_settable<const L22&, 1>);
  static_assert(not element_settable<const U22&, 2>);
  static_assert(not element_settable<const U22&, 1>);
  static_assert(not element_settable<const DM22&, 2>);
  static_assert(not element_settable<const DM22&, 1>);
  static_assert(not element_settable<const DD2&, 2>);
  static_assert(not element_settable<const DD2&, 1>);
  static_assert(not element_settable<const DL2&, 2>);
  static_assert(not element_settable<const DL2&, 1>);

  static_assert(not element_settable<SelfAdjointMatrix<const M22, TriangleType::lower>&, 2>);
  static_assert(not element_settable<SelfAdjointMatrix<const D2, TriangleType::lower>&, 2>);
  static_assert(not element_settable<SelfAdjointMatrix<const D2, TriangleType::lower>&, 1>);
  static_assert(not element_settable<SelfAdjointMatrix<DiagonalMatrix<const eigen_matrix_t<double, 2, 1>>, TriangleType::lower>&, 2>);
  static_assert(not element_settable<SelfAdjointMatrix<DiagonalMatrix<const eigen_matrix_t<double, 2, 1>>, TriangleType::lower>&, 1>);

  auto l1 = L22 {9, 3, 3, 10};
  set_element(l1, 3.1, 1, 0);
  EXPECT_NEAR(get_element(l1, 1, 0), 3.1, 1e-8);
  EXPECT_NEAR(get_element(l1, 0, 1), 3.1, 1e-8);
  set_element(l1, 3.2, 0, 1);
  EXPECT_NEAR(get_element(l1, 1, 0), 3.2, 1e-8);
  EXPECT_NEAR(get_element(l1, 0, 1), 3.2, 1e-8);
  EXPECT_EQ(l1(0, 1), 3.2);
  EXPECT_EQ(l1(1, 1), 10);
  //
  l1(0, 0) = 5;
  EXPECT_EQ(l1(0, 0), 5);
  l1(1, 0) = 6;
  EXPECT_EQ(l1(1, 0), 6);
  l1(0, 1) = 7; // Should overwrite the 6
  EXPECT_EQ(l1(1, 0), 7);
  EXPECT_EQ(l1(0, 1), 7);
  l1(1, 1) = 8;
  EXPECT_EQ(l1(1, 1), 8);
  EXPECT_TRUE(is_near(l1, mat22(5, 7, 7, 8)));

  auto u1 = U22 {9, 3, 3, 10};
  u1(0, 0) = 5;
  EXPECT_EQ(u1(0, 0), 5);
  u1(0, 1) = 6;
  EXPECT_EQ(u1(0, 1), 6);
  u1(1, 0) = 7; // Should overwrite the 6
  EXPECT_EQ(u1(1, 0), 7);
  EXPECT_EQ(u1(0, 1), 7);
  u1(1, 1) = 8;
  EXPECT_EQ(u1(1, 1), 8);
  EXPECT_TRUE(is_near(u1, mat22(5, 7, 7, 8)));
  //
  auto d9 = DM22 {9, 10};
  d9(0, 0) = 7.1;
  EXPECT_NEAR(d9(0), 7.1, 1e-8);
  d9(1) = 8.1;
  EXPECT_NEAR(d9(1, 1), 8.1, 1e-8);
  bool test = false; try { d9(1, 0) = 9.1; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_TRUE(is_near(d9, mat22(7.1, 0, 0, 8.1)));
  //
  auto d9b = DD2 {9, 10};
  d9b(0, 0) = 7.1;
  EXPECT_NEAR(d9b(0, 0), 7.1, 1e-8);
  d9b(1, 1) = 8.1;
  EXPECT_NEAR(d9b(1, 1), 8.1, 1e-8);
  test = false; try { d9b(1, 0) = 9.1; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_TRUE(is_near(d9b, mat22(7.1, 0, 0, 8.1)));
  //
  auto d9c = DL2 {9, 10};
  d9c(0, 0) = 7.1;
  EXPECT_NEAR(d9c(0, 0), 7.1, 1e-8);
  d9c(1, 1) = 8.1;
  EXPECT_NEAR(d9c(1, 1), 8.1, 1e-8);
  test = false; try { d9c(1, 0) = 9.1; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_TRUE(is_near(d9c, mat22(7.1, 0, 0, 8.1)));
  //
  EXPECT_NEAR((SelfAdjointMatrix<eigen_matrix_t<double, 1, 1>, TriangleType::diagonal> {7.})(0), 7., 1e-6);
  EXPECT_NEAR((SelfAdjointMatrix<eigen_matrix_t<double, 1, 1>, TriangleType::lower> {7.})(0), 7., 1e-6);
  EXPECT_NEAR((SelfAdjointMatrix<eigen_matrix_t<double, 1, 1>, TriangleType::upper> {7.})(0), 7., 1e-6);
  EXPECT_NEAR((DM22 {9, 10})(0), 9, 1e-6);
  EXPECT_NEAR((DM22 {9, 10})(1), 10, 1e-6);
  EXPECT_NEAR((DD2 {9, 10})(0), 9, 1e-6);
  EXPECT_NEAR((DD2 {9, 10})(1), 10, 1e-6);
  EXPECT_NEAR((DL2 {9, 10})(0), 9, 1e-6);
  EXPECT_NEAR((DL2 {9, 10})(1), 10, 1e-6);
  EXPECT_NEAR((DM22 {9, 10})[0], 9, 1e-6);
  EXPECT_NEAR((DM22 {9, 10})[1], 10, 1e-6);
  EXPECT_NEAR((DD2 {9, 10})[0], 9, 1e-6);
  EXPECT_NEAR((DD2 {9, 10})[1], 10, 1e-6);
  EXPECT_NEAR((DL2 {9, 10})[0], 9, 1e-6);
  EXPECT_NEAR((DL2 {9, 10})[1], 10, 1e-6);
  //
  EXPECT_NEAR((L22 {9., 3, 3, 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((U22 {9., 3, 3, 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((DM22 {9., 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((DD2 {9., 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((DL2 {9., 10})(0, 0), 9, 1e-6);
  //
  EXPECT_NEAR((L22 {9., 3, 3, 10})(0, 1), 3, 1e-6);
  EXPECT_NEAR((U22 {9., 3, 3, 10})(0, 1), 3, 1e-6);
  EXPECT_NEAR((DM22 {9., 10})(0, 1), 0, 1e-6);
  EXPECT_NEAR((DD2 {9., 10})(0, 1), 0, 1e-6);
  EXPECT_NEAR((DL2 {9., 10})(0, 1), 0, 1e-6);
  //
  EXPECT_NEAR((L22 {9., 3, 3, 10})(1, 0), 3, 1e-6);
  EXPECT_NEAR((U22 {9., 3, 3, 10})(1, 0), 3, 1e-6);
  EXPECT_NEAR((DM22 {9., 10})(1, 0), 0, 1e-6);
  EXPECT_NEAR((DD2 {9., 10})(1, 0), 0, 1e-6);
  EXPECT_NEAR((DL2 {9., 10})(1, 0), 0, 1e-6);
  //
  EXPECT_NEAR((L22 {9., 3, 3, 10})(1, 1), 10, 1e-6);
  EXPECT_NEAR((U22 {9., 3, 3, 10})(1, 1), 10, 1e-6);
  EXPECT_NEAR((DM22 {9., 10})(1, 1), 10, 1e-6);
  EXPECT_NEAR((DD2 {9., 10})(1, 1), 10, 1e-6);
  EXPECT_NEAR((DL2 {9., 10})(1, 1), 10, 1e-6);
  //
  EXPECT_NEAR((L22 {9., 3, 3, 10}).nested_matrix()(0, 0), 9, 1e-6);
  EXPECT_NEAR((L22 {9., 3, 3, 10}).nested_matrix()(1, 0), 3, 1e-6);
  EXPECT_NEAR((L22 {9., 3, 3, 10}).nested_matrix()(1, 1), 10, 1e-6);
}


TEST(special_matrices, make_hermitian_matrix)
{
  auto m22h = make_dense_writable_matrix_from<M22>(3, 1, 1, 3);
  auto m22u = make_dense_writable_matrix_from<M22>(3, 1, 0, 3);
  auto m22l = make_dense_writable_matrix_from<M22>(3, 0, 1, 3);
  auto m22d = make_dense_writable_matrix_from<M22>(3, 0, 0, 3);
  auto m22_uppert = Eigen::TriangularView<M22, Eigen::Upper> {m22h};
  auto m22_lowert = Eigen::TriangularView<M22, Eigen::Lower> {m22h};
  auto m22_upperh = Eigen::SelfAdjointView<M22, Eigen::Upper> {m22u};
  auto m22_lowerh = Eigen::SelfAdjointView<M22, Eigen::Lower> {m22l};

  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::upper>(m22_uppert), m22h));
  static_assert(eigen_SelfAdjointView<decltype(make_hermitian_matrix<HermitianAdapterType::upper>(m22_uppert))>);
  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::lower>(m22_uppert), m22h));
  static_assert(eigen_SelfAdjointView<decltype(make_hermitian_matrix<HermitianAdapterType::lower>(m22_uppert))>);
  static_assert(hermitian_adapter<decltype(make_hermitian_matrix<HermitianAdapterType::lower>(m22_uppert)), HermitianAdapterType::lower>);
  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::lower>(m22_lowert), m22h));
  static_assert(eigen_SelfAdjointView<decltype(make_hermitian_matrix<HermitianAdapterType::lower>(m22_lowert))>);
  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::upper>(m22_lowert), m22h));
  static_assert(eigen_SelfAdjointView<decltype(make_hermitian_matrix<HermitianAdapterType::upper>(m22_lowert))>);
  static_assert(hermitian_adapter<decltype(make_hermitian_matrix<HermitianAdapterType::upper>(m22_lowert)), HermitianAdapterType::upper>);

  auto m20h = M2x{m22h};
  auto m20_upperh = Eigen::SelfAdjointView<M2x, Eigen::Upper> {m20h};
  auto m20_lowerh = Eigen::SelfAdjointView<M2x, Eigen::Lower> {m20h};

  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::upper>(m20_upperh), m22h));
  static_assert(eigen_self_adjoint_expr<decltype(make_hermitian_matrix<HermitianAdapterType::upper>(m20_upperh))>);
  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::lower>(m20_upperh), m22h));
  static_assert(eigen_self_adjoint_expr<decltype(make_hermitian_matrix<HermitianAdapterType::lower>(m20_upperh))>);
  static_assert(hermitian_adapter<decltype(make_hermitian_matrix<HermitianAdapterType::lower>(m20_upperh)), HermitianAdapterType::upper>);
  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::lower>(m20_lowerh), m22h));
  static_assert(eigen_self_adjoint_expr<decltype(make_hermitian_matrix<HermitianAdapterType::lower>(m20_lowerh))>);
  EXPECT_TRUE(is_near(make_hermitian_matrix<HermitianAdapterType::upper>(m20_lowerh), m22h));
  static_assert(eigen_self_adjoint_expr<decltype(make_hermitian_matrix<HermitianAdapterType::upper>(m20_lowerh))>);
  static_assert(hermitian_adapter<decltype(make_hermitian_matrix<HermitianAdapterType::upper>(m20_lowerh)), HermitianAdapterType::lower>);

  static_assert(zero_matrix<decltype(make_EigenSelfAdjointMatrix<TriangleType::upper>(make_zero_matrix_like<M22>()))>);
  static_assert(zero_matrix<decltype(make_EigenSelfAdjointMatrix<TriangleType::lower>(make_zero_matrix_like<M22>()))>);
  static_assert(zero_matrix<decltype(make_EigenSelfAdjointMatrix(make_zero_matrix_like<M22>()))>);
  static_assert(hermitian_adapter<decltype(make_EigenSelfAdjointMatrix<TriangleType::upper>(make_dense_writable_matrix_from<M22>(9, 3, 3, 10))), HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<decltype(make_EigenSelfAdjointMatrix<TriangleType::lower>(make_dense_writable_matrix_from<M22>(9, 3, 3, 10))), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(make_EigenSelfAdjointMatrix(make_dense_writable_matrix_from<M22>(9, 3, 3, 10))), HermitianAdapterType::lower>);
  static_assert(diagonal_matrix<decltype(make_EigenSelfAdjointMatrix<TriangleType::upper>(D2 {3., 4}))>);
  static_assert(diagonal_matrix<decltype(make_EigenSelfAdjointMatrix<TriangleType::lower>(D2 {3., 4}))>);
  static_assert(diagonal_matrix<decltype(make_EigenSelfAdjointMatrix(D2 {3., 4}))>);

  static_assert(hermitian_adapter<decltype(make_EigenSelfAdjointMatrix<TriangleType::upper>(U22 {9, 3, 3, 10})), HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<decltype(make_EigenSelfAdjointMatrix<TriangleType::lower>(L22 {9, 3, 3, 10})), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(make_EigenSelfAdjointMatrix<TriangleType::upper>(L22 {9, 3, 3, 10})), HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<decltype(make_EigenSelfAdjointMatrix<TriangleType::lower>(U22 {9, 3, 3, 10})), HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<decltype(make_EigenSelfAdjointMatrix(U22 {9, 3, 3, 10})), HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<decltype(make_EigenSelfAdjointMatrix(L22 {9, 3, 3, 10})), HermitianAdapterType::lower>);
}


TEST(special_matrices, SelfAdjointMatrix_traits)
{
  using Dl = SelfAdjointMatrix<M22, TriangleType::lower>;
  using Du = SelfAdjointMatrix<M22, TriangleType::upper>;
  //
  EXPECT_TRUE(is_near(make_zero_matrix_like<Dl>(), M22::Zero()));
  EXPECT_TRUE(is_near(make_zero_matrix_like<Du>(), M22::Zero()));
  //
  EXPECT_TRUE(is_near(make_identity_matrix_like<Dl>(), M22::Identity()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<Du>(), M22::Identity()));

  static_assert(hermitian_adapter<L22, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<U22, HermitianAdapterType::upper>);
  static_assert(diagonal_matrix<DM22>);
  static_assert(diagonal_matrix<DD2>);
  static_assert(diagonal_matrix<DL2>);
  static_assert(zero_matrix<decltype(SelfAdjointMatrix<decltype(make_zero_matrix_like<M22>()), TriangleType::lower>(make_zero_matrix_like<M22>()))>);
  static_assert(zero_matrix<decltype(SelfAdjointMatrix<decltype(make_zero_matrix_like<M22>()), TriangleType::upper>(make_zero_matrix_like<M22>()))>);
  static_assert(identity_matrix<decltype(SelfAdjointMatrix<decltype(make_identity_matrix_like<M22>()), TriangleType::lower>(make_identity_matrix_like<M22>()))>);
}


TEST(special_matrices, SelfAdjointMatrix_overloads)
{
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(L22(9., 3, 3, 10)), make_dense_writable_matrix_from<M22>(9, 3, 3, 10)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(U22(9., 3, 3, 10)), make_dense_writable_matrix_from<M22>(9, 3, 3, 10)));
  //
  EXPECT_TRUE(is_near(make_self_contained(L22(M22::Zero())), make_dense_writable_matrix_from<M22>(0, 0, 0, 0)));
  EXPECT_TRUE(is_near(make_self_contained(U22(M22::Zero())), make_dense_writable_matrix_from<M22>(0, 0, 0, 0)));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(L22 {9, 3, 3, 10} * 2))>, L22>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(U22 {9, 3, 3, 10} * 2))>, U22>);
  //
  //
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<decltype(M22::Identity()), TriangleType::lower>(M22::Identity())), M22::Identity()));
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<decltype(M22::Identity()), TriangleType::upper>(M22::Identity())), M22::Identity()));
  static_assert(identity_matrix<decltype(Cholesky_square(SelfAdjointMatrix<decltype(M22::Identity()), TriangleType::lower>(M22::Identity())))>);
  static_assert(identity_matrix<decltype(Cholesky_square(SelfAdjointMatrix<decltype(M22::Identity()), TriangleType::upper>(M22::Identity())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<decltype(make_zero_matrix_like<M22>()), TriangleType::lower>(make_zero_matrix_like<M22>())), M22::Zero()));
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<decltype(make_zero_matrix_like<M22>()), TriangleType::upper>(make_zero_matrix_like<M22>())), M22::Zero()));
  static_assert(zero_matrix<decltype(Cholesky_square(SelfAdjointMatrix<decltype(make_zero_matrix_like<M22>()), TriangleType::lower>(make_zero_matrix_like<M22>())))>);
  static_assert(zero_matrix<decltype(Cholesky_square(SelfAdjointMatrix<decltype(make_zero_matrix_like<M22>()), TriangleType::upper>(make_zero_matrix_like<M22>())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<decltype(DiagonalMatrix{2, 3}), TriangleType::lower>(DiagonalMatrix{2, 3})), DiagonalMatrix{4, 9}));
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<decltype(DiagonalMatrix{2, 3}), TriangleType::upper>(DiagonalMatrix{2, 3})), DiagonalMatrix{4, 9}));
  static_assert(eigen_diagonal_expr<decltype(Cholesky_square(SelfAdjointMatrix<decltype(DiagonalMatrix{2, 3}), TriangleType::lower>(DiagonalMatrix{2, 3})))>);
  static_assert(eigen_diagonal_expr<decltype(Cholesky_square(SelfAdjointMatrix<decltype(DiagonalMatrix{2, 3}), TriangleType::upper>(DiagonalMatrix{2, 3})))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<M22, TriangleType::diagonal>(make_dense_writable_matrix_from<M22>(3, 0, 1, 3))), DiagonalMatrix{9., 9}));
  static_assert(eigen_diagonal_expr<decltype(Cholesky_square(SelfAdjointMatrix<M22, TriangleType::diagonal>(make_dense_writable_matrix_from<M22>(3, 0, 1, 3))))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<eigen_matrix_t<double, 1, 1>, TriangleType::lower>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(81)));
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<eigen_matrix_t<double, 1, 1>, TriangleType::upper>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(81)));
  //
  //
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<decltype(M22::Identity()), TriangleType::lower>(M22::Identity())), M22::Identity()));
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<decltype(M22::Identity()), TriangleType::upper>(M22::Identity())), M22::Identity()));
  static_assert(identity_matrix<decltype(Cholesky_factor(SelfAdjointMatrix<decltype(M22::Identity()), TriangleType::lower>(M22::Identity())))>);
  static_assert(identity_matrix<decltype(Cholesky_factor(SelfAdjointMatrix<decltype(M22::Identity()), TriangleType::upper>(M22::Identity())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<decltype(make_zero_matrix_like<M22>()), TriangleType::lower>(make_zero_matrix_like<M22>())), M22::Zero()));
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<decltype(make_zero_matrix_like<M22>()), TriangleType::upper>(make_zero_matrix_like<M22>())), M22::Zero()));
  static_assert(zero_matrix<decltype(Cholesky_factor(SelfAdjointMatrix<decltype(make_zero_matrix_like<M22>()), TriangleType::lower>(make_zero_matrix_like<M22>())))>);
  static_assert(zero_matrix<decltype(Cholesky_factor(SelfAdjointMatrix<decltype(make_zero_matrix_like<M22>()), TriangleType::upper>(make_zero_matrix_like<M22>())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<decltype(DiagonalMatrix{4, 9}), TriangleType::lower>(DiagonalMatrix{4, 9})), DiagonalMatrix{2, 3}));
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<decltype(DiagonalMatrix{4, 9}), TriangleType::upper>(DiagonalMatrix{4, 9})), DiagonalMatrix{2, 3}));
  static_assert(eigen_diagonal_expr<decltype(Cholesky_factor(SelfAdjointMatrix<decltype(DiagonalMatrix{4, 9}), TriangleType::lower>(DiagonalMatrix{4, 9})))>);
  static_assert(eigen_diagonal_expr<decltype(Cholesky_factor(SelfAdjointMatrix<decltype(DiagonalMatrix{4, 9}), TriangleType::upper>(DiagonalMatrix{4, 9})))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<M22, TriangleType::diagonal>(make_dense_writable_matrix_from<M22>(9, 3, 3, 9))), DiagonalMatrix{3., 3}));
  static_assert(eigen_diagonal_expr<decltype(Cholesky_factor(SelfAdjointMatrix<M22, TriangleType::diagonal>(make_dense_writable_matrix_from<M22>(9, 3, 3, 9))))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<eigen_matrix_t<double, 1, 1>, TriangleType::lower>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(3)));
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<eigen_matrix_t<double, 1, 1>, TriangleType::upper>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(3)));
  //
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(U22 {9., 3, 3, 10}), mat22(3., 0, 1, 3)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(L22 {9., 3, 3, 10}), mat22(3., 1, 0, 3)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(L22 {9., 3, 3, 10}), mat22(3., 0, 1, 3)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(U22 {9., 3, 3, 10}), mat22(3., 1, 0, 3)));
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(U22 {9., 3, 3, 10})), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(L22 {9., 3, 3, 10})), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(L22 {9., 3, 3, 10})), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(U22 {9., 3, 3, 10})), TriangleType::upper>);
  //
  // Semidefinite case:
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(U22 {9., 3, 3, 1}), mat22(3., 0, 1, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(L22 {9., 3, 3, 1}), mat22(3., 1, 0, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(L22 {9., 3, 3, 1}), mat22(3., 0, 1, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(U22 {9., 3, 3, 1}), mat22(3., 1, 0, 0)));
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(U22 {9., 3, 3, 1})), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(L22 {9., 3, 3, 1})), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(L22 {9., 3, 3, 1})), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(U22 {9., 3, 3, 1})), TriangleType::upper>);

  // Constant semidefinite case:
  using Const922 = ConstantAdapter<eigen_matrix_t<double, 2, 2>, 9>;
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<Const922, TriangleType::lower> {Const922 {}}), mat22(3., 0, 3, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<Const922, TriangleType::upper> {Const922 {}}), mat22(3., 3, 0, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<Const922, TriangleType::lower> {Const922 {}}), mat22(3., 0, 3, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<Const922, TriangleType::upper> {Const922 {}}), mat22(3., 3, 0, 0)));
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<Const922, TriangleType::lower> {Const922 {}})), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<Const922, TriangleType::upper> {Const922 {}})), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<Const922, TriangleType::upper> {Const922 {}})), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<Const922, TriangleType::lower> {Const922 {}})), TriangleType::upper>);

  using M2Const = typename M22::ConstantReturnType;
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<M2Const, TriangleType::lower>(M22::Constant(9))), mat22(3., 0, 3, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<M2Const, TriangleType::upper>(M22::Constant(9))), mat22(3., 3, 0, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<M2Const, TriangleType::lower>(M22::Constant(9))), mat22(3., 0, 3, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<M2Const, TriangleType::upper>(M22::Constant(9))), mat22(3., 3, 0, 0)));
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<M2Const, TriangleType::lower>(M22::Constant(9)))), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<M2Const, TriangleType::upper>(M22::Constant(9)))), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<M2Const, TriangleType::upper>(M22::Constant(9)))), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<M2Const, TriangleType::lower>(M22::Constant(9)))), TriangleType::upper>);

  // Zero (positive and negative semidefinite) case:
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(U22 {0., 0, 0, 0}), mat22(0., 0, 0, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(L22 {0., 0, 0, 0}), mat22(0., 0, 0, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(L22 {0., 0, 0, 0}), mat22(0., 0, 0, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(U22 {0., 0, 0, 0}), mat22(0., 0, 0, 0)));
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(U22 {0., 0, 0, 0})), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(L22 {0., 0, 0, 0})), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(L22 {0., 0, 0, 0})), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(U22 {0., 0, 0, 0})), TriangleType::upper>);
  //
  //
  EXPECT_TRUE(is_near(diagonal_of(L22 {9., 3, 3, 10}), make_eigen_matrix<double, 2, 1>(9., 10)));
  EXPECT_TRUE(is_near(diagonal_of(U22 {9., 3, 3, 10}), make_eigen_matrix<double, 2, 1>(9., 10)));
  //
  EXPECT_TRUE(is_near(transpose(L22 {9., 3, 3, 10}), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(transpose(U22 {9., 3, 3, 10}), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(transpose(CL22 {9., cdouble(3,-1), cdouble(3,1), 10}), make_dense_writable_matrix_from<C22>(9., cdouble(3,1), cdouble(3,-1), 10)));
  EXPECT_TRUE(is_near(transpose(CU22 {9., cdouble(3,-1), cdouble(3,1), 10}), make_dense_writable_matrix_from<C22>(9., cdouble(3,1), cdouble(3,-1), 10)));
  //
  EXPECT_TRUE(is_near(adjoint(L22 {9., 3, 3, 10}), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(adjoint(U22 {9., 3, 3, 10}), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(adjoint(CL22 {9., cdouble(3,-1), cdouble(3,1), 10}), make_dense_writable_matrix_from<C22>(9., cdouble(3,-1), cdouble(3,1), 10)));
  EXPECT_TRUE(is_near(adjoint(CU22 {9., cdouble(3,-1), cdouble(3,1), 10}), make_dense_writable_matrix_from<C22>(9., cdouble(3,-1), cdouble(3,1), 10)));
  //
  EXPECT_NEAR(determinant(L22 {9., 3, 3, 10}), 81, 1e-6);
  EXPECT_NEAR(determinant(U22 {9., 3, 3, 10}), 81, 1e-6);
  //
  EXPECT_NEAR(trace(L22 {9., 3, 3, 10}), 19, 1e-6);
  EXPECT_NEAR(trace(U22 {9., 3, 3, 10}), 19, 1e-6);
  //
  EXPECT_TRUE(is_near(average_reduce<1>(L22 {9., 3, 3, 10}), make_eigen_matrix(6., 6.5)));
  EXPECT_TRUE(is_near(average_reduce<1>(U22 {9., 3, 3, 10}), make_eigen_matrix(6., 6.5)));
  //
  EXPECT_TRUE(is_near(average_reduce<0>(L22 {9., 3, 3, 10}), make_eigen_matrix<double, 1, 2>(6., 6.5)));
  EXPECT_TRUE(is_near(average_reduce<0>(U22 {9., 3, 3, 10}), make_eigen_matrix<double, 1, 2>(6., 6.5)));
  //
}


TEST(special_matrices, SelfAdjointMatrix_solve)
{
  EXPECT_TRUE(is_near(solve(L22 {9., 3, 3, 10}, make_eigen_matrix<double, 2, 1>(15, 23)), make_eigen_matrix(1., 2)));
  EXPECT_TRUE(is_near(solve(U22 {9., 3, 3, 10}, make_eigen_matrix<double, 2, 1>(15, 23)), make_eigen_matrix(1., 2)));
  //
  EXPECT_TRUE(is_near(solve(L22 {9., 3, 3, 10}, make_eigen_matrix<double, 2, 1>(15, 23)), make_eigen_matrix<double, 2, 1>(1, 2)));
  EXPECT_TRUE(is_near(solve(U22 {9., 3, 3, 10}, make_eigen_matrix<double, 2, 1>(15, 23)), make_eigen_matrix<double, 2, 1>(1, 2)));
}


TEST(special_matrices, SelfAdjointMatrix_decompositions)
{
  EXPECT_TRUE(is_near(Cholesky_square(LQ_decomposition(L22 {9., 3, 3, 10})), mat22(90., 57, 57, 109)));
  EXPECT_TRUE(is_near(Cholesky_square(LQ_decomposition(U22 {9., 3, 3, 10})), mat22(90., 57, 57, 109)));
  //
  EXPECT_TRUE(is_near(Cholesky_square(QR_decomposition(L22 {9., 3, 3, 10})), mat22(90., 57, 57, 109)));
  EXPECT_TRUE(is_near(Cholesky_square(QR_decomposition(U22 {9., 3, 3, 10})), mat22(90., 57, 57, 109)));
}


TEST(special_matrices, SelfAdjointMatrix_blocks_lower)
{
  auto ma = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::lower> {1, 2, 3,
                                                                                  2, 4, 5,
                                                                                  3, 5, 6};
  auto mb = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::lower> {4, 5, 6,
                                                                                  5, 7, 8,
                                                                                  6, 8, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}, mb),
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1., 2, 0, 0, 0,
                                                                          2, 3, 0, 0, 0,
                                                                          0, 0, 4, 5, 6,
                                                                          0, 0, 5, 7, 8,
                                                                          0, 0, 6, 8, 9}));
  static_assert(hermitian_adapter<decltype(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}, mb)), HermitianAdapterType::lower>);

  EXPECT_TRUE(is_near(concatenate_vertical(ma, mb),
    make_eigen_matrix<6,3>(1., 2, 3,
                            2, 4, 5,
                            3, 5, 6,
                            4, 5, 6,
                            5, 7, 8,
                            6, 8, 9)));
  EXPECT_TRUE(is_near(concatenate_horizontal(ma, mb),
    make_eigen_matrix<3,6>(1., 2, 3, 4, 5, 6,
                            2, 4, 5, 5, 7, 8,
                            3, 5, 6, 6, 8, 9)));
  EXPECT_TRUE(is_near(split_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1., 2, 0, 0, 0,
                                                                          2, 3, 0, 0, 0,
                                                                          0, 0, 4, 5, 6,
                                                                          0, 0, 5, 7, 8,
                                                                          0, 0, 6, 8, 9}),
      std::tuple {SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}, mb}));
  const auto a1 = SelfAdjointMatrix<const eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1., 2, 0, 0, 0,
                                                                                              2, 3, 0, 0, 0,
                                                                                              0, 0, 4, 5, 6,
                                                                                              0, 0, 5, 7, 8,
                                                                                              0, 0, 6, 8, 9};
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(a1), std::tuple {SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}, mb}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1., 2, 0, 0, 0,
                                                                          2, 3, 0, 0, 0,
                                                                          0, 0, 4, 5, 6,
                                                                          0, 0, 5, 7, 8,
                                                                          0, 0, 6, 8, 9}),
    std::tuple {SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3},
               SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {4., 5, 5, 7}}));
  EXPECT_TRUE(is_near(split_vertical(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<2, 3>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1, 2, 0, 0, 0,
                                                                          2, 3, 0, 0, 0,
                                                                          0, 0, 4, 5, 6,
                                                                          0, 0, 5, 7, 8,
                                                                          0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<2,5>(1., 2, 0, 0, 0,
                                        2, 3, 0, 0, 0),
               make_eigen_matrix<3,5>(0., 0, 4, 5, 6,
                                       0, 0, 5, 7, 8,
                                       0, 0, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1, 2, 0, 0, 0,
                                                                          2, 3, 0, 0, 0,
                                                                          0, 0, 4, 5, 6,
                                                                          0, 0, 5, 7, 8,
                                                                          0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<2,5>(1., 2, 0, 0, 0,
                                        2, 3, 0, 0, 0),
               make_eigen_matrix<2,5>(0., 0, 4, 5, 6,
                                       0, 0, 5, 7, 8)}));
  EXPECT_TRUE(is_near(split_horizontal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<2, 3>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1, 2, 0, 0, 0,
                                                                          2, 3, 0, 0, 0,
                                                                          0, 0, 4, 5, 6,
                                                                          0, 0, 5, 7, 8,
                                                                          0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<5,2>(1., 2, 2, 3, 0, 0, 0, 0, 0, 0),
               make_eigen_matrix<5,3>(0., 0, 0, 0, 0, 0, 4, 5, 6, 5, 7, 8, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1, 2, 0, 0, 0,
                                                                          2, 3, 0, 0, 0,
                                                                          0, 0, 4, 5, 6,
                                                                          0, 0, 5, 7, 8,
                                                                          0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<5,2>(1., 2, 2, 3, 0, 0, 0, 0, 0, 0),
               make_eigen_matrix<5,2>(0., 0, 0, 0, 4, 5, 5, 7, 6, 8)}));

  EXPECT_TRUE(is_near(column(mb, 2), make_eigen_matrix(6., 8, 9)));
  EXPECT_TRUE(is_near(column<1>(mb), make_eigen_matrix(5., 7, 8)));
  EXPECT_TRUE(is_near(row(mb, 2), make_eigen_matrix<double, 1, 3>(6., 8, 9)));
  EXPECT_TRUE(is_near(row<1>(mb), make_eigen_matrix<double, 1, 3>(5., 7, 8)));

  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 6, 7,
      6, 8, 9,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 6, 8,
      5, 8, 10,
      6, 9, 11)));

  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row){ return make_self_contained(row + row.Constant(1)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 6, 7,
      6, 8, 9,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_rowwise([](const auto& row, std::size_t i){ return make_self_contained(row + row.Constant(i)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 5, 6,
      6, 8, 9,
      8, 10, 11)));

  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 6, 7,
      6, 8, 9,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 6, 8,
      6, 9, 11,
      8, 11, 13)));
}


TEST(special_matrices, SelfAdjointMatrix_blocks_upper)
{
  auto ma = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::upper> {1, 2, 3,
                                                                                      2, 4, 5,
                                                                                      3, 5, 6};
  auto mb = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::upper> {4., 5, 6,
                                                                                      5, 7, 8,
                                                                                      6, 8, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}, mb),
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}));
  static_assert(hermitian_adapter<decltype(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}, mb)), HermitianAdapterType::upper>);

  EXPECT_TRUE(is_near(concatenate_vertical(ma, mb),
    make_eigen_matrix<6,3>(1., 2, 3,
                                    2, 4, 5,
                                    3, 5, 6,
                                    4, 5, 6,
                                    5, 7, 8,
                                    6, 8, 9)));
  EXPECT_TRUE(is_near(concatenate_horizontal(ma, mb),
    make_eigen_matrix<3,6>(1., 2, 3, 4, 5, 6,
                                    2, 4, 5, 5, 7, 8,
                                    3, 5, 6, 6, 8, 9)));
  EXPECT_TRUE(is_near(split_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}, mb}));
  const auto a1 = SelfAdjointMatrix<const eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                                                  2, 3, 0, 0, 0,
                                                                                                  0, 0, 4, 5, 6,
                                                                                                  0, 0, 5, 7, 8,
                                                                                                  0, 0, 6, 8, 9};
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(a1), std::tuple {SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}, mb}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3},
               SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {4., 5, 5, 7}}));
  EXPECT_TRUE(is_near(split_vertical(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_vertical<2, 3>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<2,5>(1., 2, 0, 0, 0,
                                               2, 3, 0, 0, 0),
               make_eigen_matrix<3,5>(0., 0, 4, 5, 6,
                                               0, 0, 5, 7, 8,
                                               0, 0, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<2,5>(1., 2, 0, 0, 0,
                                               2, 3, 0, 0, 0),
               make_eigen_matrix<2,5>(0., 0, 4, 5, 6,
                                               0, 0, 5, 7, 8)}));
  EXPECT_TRUE(is_near(split_horizontal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<2, 3>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<5,2>(1., 2, 2, 3, 0, 0, 0, 0, 0, 0),
               make_eigen_matrix<5,3>(0., 0, 0, 0, 0, 0, 4, 5, 6, 5, 7, 8, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {make_eigen_matrix<5,2>(1., 2, 2, 3, 0, 0, 0, 0, 0, 0),
               make_eigen_matrix<5,2>(0., 0, 0, 0, 4, 5, 5, 7, 6, 8)}));
  EXPECT_TRUE(is_near(column(mb, 2), make_eigen_matrix(6., 8, 9)));
  EXPECT_TRUE(is_near(column<1>(mb), make_eigen_matrix(5., 7, 8)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col){ return make_self_contained(col + col.Constant(1)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 6, 7,
      6, 8, 9,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_columnwise([](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 6, 8,
      5, 8, 10,
      6, 9, 11)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x){ return x + 1; }, mb),
    make_eigen_matrix<double, 3, 3>(
      5, 6, 7,
      6, 8, 9,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_coefficientwise([](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }, mb),
    make_eigen_matrix<double, 3, 3>(
      4, 6, 8,
      6, 9, 11,
      8, 11, 13)));
}


TEST(special_matrices, SelfAdjointMatrix_blocks_mixed)
{
  auto ma = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::upper> {1, 2, 3,
                                                                                      2, 4, 5,
                                                                                      3, 5, 6};
  auto mb = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::lower> {4., 5, 6,
                                                                                      5, 7, 8,
                                                                                      6, 8, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}, mb),
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}));
  static_assert(hermitian_adapter<decltype(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}, mb)), HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<decltype(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}, ma)), HermitianAdapterType::lower>);

  EXPECT_TRUE(is_near(concatenate_vertical(ma, mb),
    make_eigen_matrix<6,3>(1., 2, 3,
                                    2, 4, 5,
                                    3, 5, 6,
                                    4, 5, 6,
                                    5, 7, 8,
                                    6, 8, 9)));
  EXPECT_TRUE(is_near(concatenate_horizontal(ma, mb),
    make_eigen_matrix<3,6>(1., 2, 3, 4, 5, 6,
                                    2, 4, 5, 5, 7, 8,
                                    3, 5, 6, 6, 8, 9)));
  auto mc = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::upper> {4., 5, 6,
                                                                                      5, 7, 8,
                                                                                      6, 8, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}, mc),
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1., 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}));
}


TEST(special_matrices, SelfAdjointMatrix_arithmetic_lower)
{
  auto ma = L22 {4., 5, 5, 6};
  auto mb = L22 {1., 2, 2, 3};
  auto d = DiagonalMatrix<eigen_matrix_t<double, 2, 1>> {1, 3};
  auto i = M22::Identity();
  auto z = ZeroMatrix<eigen_matrix_t<double, 2, 2>> {};

  EXPECT_TRUE(is_near(ma + mb, mat22(5, 7, 7, 9))); static_assert(hermitian_matrix<decltype(ma + mb)>);
  EXPECT_TRUE(is_near(ma + d, mat22(5, 5, 5, 9))); static_assert(hermitian_matrix<decltype(ma + d)>);
  EXPECT_TRUE(is_near(d + ma, mat22(5, 5, 5, 9))); static_assert(hermitian_matrix<decltype(d + ma)>);
  EXPECT_TRUE(is_near(ma + i, mat22(5, 5, 5, 7))); static_assert(hermitian_matrix<decltype(ma + i)>);
  EXPECT_TRUE(is_near(i + ma, mat22(5, 5, 5, 7))); static_assert(hermitian_matrix<decltype(i + ma)>);
  EXPECT_TRUE(is_near(ma + z, mat22(4, 5, 5, 6))); static_assert(hermitian_matrix<decltype(ma + z)>);
  EXPECT_TRUE(is_near(z + ma, mat22(4, 5, 5, 6))); static_assert(hermitian_matrix<decltype(z + ma)>);

  EXPECT_TRUE(is_near(ma - mb, mat22(3, 3, 3, 3))); static_assert(hermitian_matrix<decltype(ma - mb)>);
  EXPECT_TRUE(is_near(ma - d, mat22(3, 5, 5, 3))); static_assert(hermitian_matrix<decltype(ma - d)>);
  EXPECT_TRUE(is_near(d - ma, mat22(-3, -5, -5, -3))); static_assert(hermitian_matrix<decltype(d - ma)>);
  EXPECT_TRUE(is_near(ma - i, mat22(3, 5, 5, 5))); static_assert(hermitian_matrix<decltype(ma - i)>);
  EXPECT_TRUE(is_near(i - ma, mat22(-3, -5, -5, -5))); static_assert(hermitian_matrix<decltype(i - ma)>);
  EXPECT_TRUE(is_near(ma - z, mat22(4, 5, 5, 6))); static_assert(hermitian_matrix<decltype(ma - z)>);
  EXPECT_TRUE(is_near(z - ma, mat22(-4, -5, -5, -6))); static_assert(hermitian_matrix<decltype(z - ma)>);

  EXPECT_TRUE(is_near(ma * 2, mat22(8, 10, 10, 12))); static_assert(hermitian_matrix<decltype(ma * 2)>);
  EXPECT_TRUE(is_near(2 * ma, mat22(8, 10, 10, 12))); static_assert(hermitian_matrix<decltype(2 * ma)>);
  EXPECT_TRUE(is_near(ma / 2, mat22(2, 2.5, 2.5, 3))); static_assert(hermitian_matrix<decltype(ma / 2)>);
  static_assert(hermitian_matrix<decltype(ma / 0)>);
  EXPECT_TRUE(is_near(-ma, mat22(-4, -5, -5, -6)));  static_assert(hermitian_matrix<decltype(-ma)>);

  EXPECT_TRUE(is_near(SelfAdjointMatrix<decltype(i), TriangleType::diagonal> {i} * 2, mat22(2, 0, 0, 2)));
  static_assert(diagonal_matrix<decltype(SelfAdjointMatrix<decltype(i), TriangleType::diagonal> {i} * 2)>);
  EXPECT_TRUE(is_near(2 * SelfAdjointMatrix<decltype(i), TriangleType::diagonal> {i}, mat22(2, 0, 0, 2)));
  static_assert(diagonal_matrix<decltype(2 * SelfAdjointMatrix<decltype(i), TriangleType::diagonal> {i})>);
  EXPECT_TRUE(is_near(SelfAdjointMatrix<decltype(i), TriangleType::diagonal> {i} / 0.5, mat22(2, 0, 0, 2)));
  static_assert(diagonal_matrix<decltype(SelfAdjointMatrix<decltype(i), TriangleType::diagonal> {i} / 2)>);

  EXPECT_TRUE(is_near(ma * mb, mat22(14, 23, 17, 28)));
  EXPECT_TRUE(is_near(ma * d, mat22(4, 15, 5, 18)));
  EXPECT_TRUE(is_near(d * ma, mat22(4, 5, 15, 18)));
  EXPECT_TRUE(is_near(ma * i, ma));  static_assert(hermitian_matrix<decltype(ma * i)>);
  EXPECT_TRUE(is_near(i * ma, ma));  static_assert(hermitian_matrix<decltype(i * ma)>);
  EXPECT_TRUE(is_near(ma * z, z));  static_assert(zero_matrix<decltype(ma * z)>);
  EXPECT_TRUE(is_near(z * ma, z));  static_assert(zero_matrix<decltype(z * ma)>);
  EXPECT_TRUE(is_near(L22{make_dense_writable_matrix_from<L22>(mat22(1, 2, 3, 4)} * (ma * mat22(1, 3, 2, 4))), mat22(48, 110, 110, 252)));

  auto tl1 = TriangularMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {2, 0, 1, 2};
  auto tu1 = TriangularMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {2, 1, 0, 2};
  EXPECT_TRUE(is_near(ma * tl1, mat22(13, 10, 16, 12)));
  EXPECT_TRUE(is_near(ma * tu1, mat22(8, 14, 10, 17)));
  EXPECT_TRUE(is_near(tl1 * ma, mat22(8, 10, 14, 17)));
  EXPECT_TRUE(is_near(tu1 * ma, mat22(13, 16, 10, 12)));
  EXPECT_TRUE(is_near(mat22(4, 5, 5, 6) * mb, mat22(14., 23, 17, 28)));
  EXPECT_TRUE(is_near(ma * mat22(1, 2, 2, 3), mat22(14., 23, 17, 28)));
}


TEST(special_matrices, SelfAdjointMatrix_arithmetic_upper)
{
  auto ma = U22 {4., 5, 5, 6};
  auto mb = U22 {1., 2, 2, 3};
  auto d = DiagonalMatrix<eigen_matrix_t<double, 2, 1>> {1, 3};
  auto i = M22::Identity();
  auto z = ZeroMatrix<eigen_matrix_t<double, 2, 2>> {};
  EXPECT_TRUE(is_near(ma + mb, mat22(5, 7, 7, 9))); static_assert(hermitian_matrix<decltype(ma + mb)>);
  EXPECT_TRUE(is_near(ma + d, mat22(5, 5, 5, 9))); static_assert(hermitian_matrix<decltype(ma + d)>);
  EXPECT_TRUE(is_near(d + ma, mat22(5, 5, 5, 9))); static_assert(hermitian_matrix<decltype(d + ma)>);
  EXPECT_TRUE(is_near(ma + i, mat22(5, 5, 5, 7))); static_assert(hermitian_matrix<decltype(ma + i)>);
  EXPECT_TRUE(is_near(i + ma, mat22(5, 5, 5, 7))); static_assert(hermitian_matrix<decltype(i + ma)>);
  EXPECT_TRUE(is_near(ma + z, mat22(4, 5, 5, 6))); static_assert(hermitian_matrix<decltype(ma + z)>);
  EXPECT_TRUE(is_near(z + ma, mat22(4, 5, 5, 6))); static_assert(hermitian_matrix<decltype(z + ma)>);

  EXPECT_TRUE(is_near(ma - mb, mat22(3, 3, 3, 3))); static_assert(hermitian_matrix<decltype(ma - mb)>);
  EXPECT_TRUE(is_near(ma - d, mat22(3, 5, 5, 3))); static_assert(hermitian_matrix<decltype(ma - d)>);
  EXPECT_TRUE(is_near(d - ma, mat22(-3, -5, -5, -3))); static_assert(hermitian_matrix<decltype(d - ma)>);
  EXPECT_TRUE(is_near(ma - i, mat22(3, 5, 5, 5))); static_assert(hermitian_matrix<decltype(ma - i)>);
  EXPECT_TRUE(is_near(i - ma, mat22(-3, -5, -5, -5))); static_assert(hermitian_matrix<decltype(i - ma)>);
  EXPECT_TRUE(is_near(ma - z, mat22(4, 5, 5, 6))); static_assert(hermitian_matrix<decltype(ma - z)>);
  EXPECT_TRUE(is_near(z - ma, mat22(-4, -5, -5, -6))); static_assert(hermitian_matrix<decltype(z - ma)>);

  EXPECT_TRUE(is_near(ma * 2, mat22(8, 10, 10, 12))); static_assert(hermitian_matrix<decltype(ma * 2)>);
  EXPECT_TRUE(is_near(2 * ma, mat22(8, 10, 10, 12))); static_assert(hermitian_matrix<decltype(2 * ma)>);
  EXPECT_TRUE(is_near(ma / 2, mat22(2, 2.5, 2.5, 3))); static_assert(hermitian_matrix<decltype(ma / 2)>);
  static_assert(hermitian_matrix<decltype(ma / 0)>);
  EXPECT_TRUE(is_near(-ma, mat22(-4, -5, -5, -6)));  static_assert(hermitian_matrix<decltype(-ma)>);

  EXPECT_TRUE(is_near(ma * mb, mat22(14, 23, 17, 28)));
  EXPECT_TRUE(is_near(ma * d, mat22(4, 15, 5, 18)));
  EXPECT_TRUE(is_near(d * ma, mat22(4, 5, 15, 18)));
  EXPECT_TRUE(is_near(ma * i, ma));  static_assert(hermitian_matrix<decltype(ma * i)>);
  EXPECT_TRUE(is_near(i * ma, ma));  static_assert(hermitian_matrix<decltype(i * ma)>);
  EXPECT_TRUE(is_near(ma * z, z));  static_assert(zero_matrix<decltype(ma * z)>);
  EXPECT_TRUE(is_near(z * ma, z));  static_assert(zero_matrix<decltype(z * ma)>);

  auto tl1 = TriangularMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {2, 0, 1, 2};
  auto tu1 = TriangularMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {2, 1, 0, 2};
  EXPECT_TRUE(is_near(ma * tl1, mat22(13, 10, 16, 12)));
  EXPECT_TRUE(is_near(ma * tu1, mat22(8, 14, 10, 17)));
  EXPECT_TRUE(is_near(tl1 * ma, mat22(8, 10, 14, 17)));
  EXPECT_TRUE(is_near(tu1 * ma, mat22(13, 16, 10, 12)));
  EXPECT_TRUE(is_near(mat22(4, 5, 5, 6) * mb, mat22(14., 23, 17, 28)));
  EXPECT_TRUE(is_near(ma * mat22(1, 2, 2, 3), mat22(14., 23, 17, 28)));
}


TEST(special_matrices, SelfAdjointMatrix_arithmetic_mixed)
{
  auto ma = U22 {4., 5, 5, 6};
  auto mb = L22 {1., 2, 2, 3};
  auto tl1 = TriangularMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {2, 0, 1, 2};
  auto tu1 = TriangularMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {2, 1, 0, 2};
  EXPECT_TRUE(is_near(ma + mb, mat22(5., 7, 7, 9))); static_assert(hermitian_adapter<decltype(ma + mb), HermitianAdapterType::upper>);
  EXPECT_TRUE(is_near(mb + ma, mat22(5., 7, 7, 9))); static_assert(hermitian_adapter<decltype(mb + ma), HermitianAdapterType::lower>);
  EXPECT_TRUE(is_near(ma - mb, mat22(3, 3, 3, 3))); static_assert(hermitian_adapter<decltype(ma - mb), HermitianAdapterType::upper>);
  EXPECT_TRUE(is_near(mb - ma, mat22(-3, -3, -3, -3))); static_assert(hermitian_adapter<decltype(mb - ma), HermitianAdapterType::lower>);
  EXPECT_TRUE(is_near(ma * mb, mat22(14, 23, 17, 28)));
  EXPECT_TRUE(is_near(mb * ma, mat22(14, 17, 23, 28)));

  EXPECT_TRUE(is_near(ma * tl1, mat22(13, 10, 16, 12)));
  EXPECT_TRUE(is_near(ma * tu1, mat22(8, 14, 10, 17)));
  EXPECT_TRUE(is_near(mb * tl1, mat22(4, 4, 7, 6)));
  EXPECT_TRUE(is_near(mb * tu1, mat22(2, 5, 4, 8)));
  EXPECT_TRUE(is_near(tl1 * ma, mat22(8, 10, 14, 17)));
  EXPECT_TRUE(is_near(tu1 * ma, mat22(13, 16, 10, 12)));
  EXPECT_TRUE(is_near(tl1 * mb, mat22(2, 4, 5, 8)));
  EXPECT_TRUE(is_near(tu1 * mb, mat22(4, 7, 4, 6)));
}


TEST(special_matrices, SelfAdjointMatrix_references)
{
  SelfAdjointMatrix<M22, TriangleType::lower> x {m_4225};
  SelfAdjointMatrix<M22&, TriangleType::lower> x_lvalue = x;
  EXPECT_TRUE(is_near(x_lvalue, m_4225));
  x = SelfAdjointMatrix<M22, TriangleType::lower> {m_93310};
  EXPECT_TRUE(is_near(x_lvalue, m_93310));
  x_lvalue = SelfAdjointMatrix<M22, TriangleType::lower> {m_4225};
  EXPECT_TRUE(is_near(x, m_4225));
  EXPECT_TRUE(is_near(SelfAdjointMatrix<M22&, TriangleType::lower> {m_4225}.nested_matrix(), mat22(4, 2, 2, 5)));
  //
  using V = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>>;
  V v1 {1., 2, 3,
        2, 4, -6,
        3, -6, -3};
  SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  EXPECT_EQ(v1(0,1), 4.1);
  v2(0, 2) = 5.2;
  EXPECT_EQ(v1(0,2), 5.2);
  EXPECT_EQ(v1(2,0), 5.2);
  SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 4.1);
  SelfAdjointMatrix<const eigen_matrix_t<double, 3, 3>&> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4);
}

