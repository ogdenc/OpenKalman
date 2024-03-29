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

#include "eigen3.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


using M2 = eigen_matrix_t<double, 2, 2>;
using D2 = DiagonalMatrix<eigen_matrix_t<double, 2, 1>>;
using Lower = SelfAdjointMatrix<M2, TriangleType::lower>;
using Upper = SelfAdjointMatrix<M2, TriangleType::upper>;
using Diagonal = SelfAdjointMatrix<M2, TriangleType::diagonal>;
using Diagonal2 = SelfAdjointMatrix<D2, TriangleType::diagonal>;
using Diagonal3 = SelfAdjointMatrix<D2, TriangleType::lower>;

template<typename...Args>
inline auto mat22(Args...args) { return MatrixTraits<M2>::make(args...); }


TEST(eigen3, SelfAdjointMatrix_class)
{
  auto m = mat22(9, 3, 3, 10);
  D2 d2 {9, 9};
  //
  Lower l1;
  l1 << 9, 7, 3, 10;
  EXPECT_TRUE(is_near(l1.nested_matrix(), mat22(9, 7, 3, 10)));
  EXPECT_TRUE(is_near(l1, m));
  Upper u1;
  u1 << 9, 3, 7, 10;
  EXPECT_TRUE(is_near(u1.nested_matrix(), mat22(9, 3, 7, 10)));
  EXPECT_TRUE(is_near(u1, m));
  Diagonal d1;
  d1 << 9, 10;
  EXPECT_TRUE(is_near(d1.nested_matrix(), mat22(9, 0, 0, 10)));
  EXPECT_TRUE(is_near(d1, mat22(9, 0, 0, 10)));
  d1.template triangularView<Eigen::Lower>() = mat22(7, 5, 6, 12);
  EXPECT_TRUE(is_near(d1, mat22(7, 0, 0, 12)));
  Diagonal2 d1b;
  d1b << 9, 10;
  EXPECT_TRUE(is_near(d1b.nested_matrix(), mat22(9, 0, 0, 10)));
  EXPECT_TRUE(is_near(d1b, mat22(9, 0, 0, 10)));
  d1b.template triangularView<Eigen::Lower>() = mat22(7, 5, 6, 12);
  EXPECT_TRUE(is_near(d1b, mat22(7, 0, 0, 12)));
  Diagonal3 d1c;
  d1c << 9, 10;
  EXPECT_TRUE(is_near(d1c.nested_matrix(), mat22(9, 0, 0, 10)));
  EXPECT_TRUE(is_near(d1c, mat22(9, 0, 0, 10)));
  d1c.template triangularView<Eigen::Lower>() = mat22(7, 5, 6, 12);
  EXPECT_TRUE(is_near(d1c, mat22(7, 0, 0, 12)));
  //
  Lower l2 {mat22(9, 3, 3, 10)};
  EXPECT_TRUE(is_near(l1, m));
  Upper u2 {mat22(9, 3, 3, 10)};
  EXPECT_TRUE(is_near(u1, m));
  //
  EXPECT_TRUE(is_near(Lower(DiagonalMatrix {3., 4}), mat22(3, 0, 0, 4)));
  EXPECT_TRUE(is_near(Upper(DiagonalMatrix {3., 4}), mat22(3, 0, 0, 4)));
  //
  EXPECT_TRUE(is_near(Lower(MatrixTraits<M2>::zero()), M2::Zero()));
  EXPECT_TRUE(is_near(Upper(MatrixTraits<M2>::zero()), M2::Zero()));
  //
  EXPECT_EQ(Lower::rows(), 2);
  EXPECT_EQ(Lower::cols(), 2);
  EXPECT_TRUE(is_near(Lower::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(Upper::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(Lower::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(Upper::identity(), M2::Identity()));
  //
  Lower l3(l2); // copy constructor
  EXPECT_TRUE(is_near(l3, m));
  Upper u3(u2); // copy constructor
  EXPECT_TRUE(is_near(u3, m));
  //
  Lower l4 = Lower{9, 3, 3, 10}; // move constructor
  EXPECT_TRUE(is_near(l4, m));
  Upper u4 = Upper{9, 3, 3, 10}; // move constructor
  EXPECT_TRUE(is_near(u4, m));
  //
  Lower l5 = SelfAdjointMatrix<decltype(M2::Zero()), TriangleType::lower>(M2::Zero()); // compatible sa-matrix
  EXPECT_TRUE(is_near(l5, M2::Zero()));
  Upper u5 = SelfAdjointMatrix<decltype(M2::Zero()), TriangleType::upper>(M2::Zero()); // compatible sa-matrix
  EXPECT_TRUE(is_near(u5, M2::Zero()));
  //
  Lower l6 = SelfAdjointMatrix<decltype(M2::Zero()), TriangleType::upper>(M2::Zero()); // opposite sa-matrix
  EXPECT_TRUE(is_near(l6, M2::Zero()));
  Upper u6 = SelfAdjointMatrix<decltype(M2::Zero()), TriangleType::lower>(M2::Zero()); // opposite sa-matrix
  EXPECT_TRUE(is_near(u6, M2::Zero()));
  //
  Lower l7 {m.selfadjointView<Eigen::Lower>()};
  EXPECT_TRUE(is_near(l7, m));
  Upper u7 {m.selfadjointView<Eigen::Upper>()};
  EXPECT_TRUE(is_near(u7, m));
  //
  Lower l8 = {m.selfadjointView<Eigen::Upper>()};
  EXPECT_TRUE(is_near(l8, m));
  Upper u8 = {m.selfadjointView<Eigen::Lower>()};
  EXPECT_TRUE(is_near(u8, m));
  //
  Lower l9 = Diagonal {9., 10}; // Construct from a diagonal sa-matrix.
  EXPECT_TRUE(is_near(l9, mat22(9, 0, 0, 10)));
  Upper u9 = Diagonal {9., 10};
  EXPECT_TRUE(is_near(u9, mat22(9, 0, 0, 10)));
  Diagonal d9 = Diagonal {9, 10};
  EXPECT_TRUE(is_near(d9, mat22(9, 0, 0, 10)));
  Diagonal d9_b = Diagonal2 {9, 10};
  EXPECT_TRUE(is_near(d9_b, mat22(9, 0, 0, 10)));
  Diagonal d9_c = Diagonal2 {9, 10};
  EXPECT_TRUE(is_near(d9_c, mat22(9, 0, 0, 10)));
  Diagonal2 d9b = Diagonal {9, 10};
  EXPECT_TRUE(is_near(d9b, mat22(9, 0, 0, 10)));
  Diagonal2 d9b_b = Diagonal2 {9, 10};
  EXPECT_TRUE(is_near(d9b_b, mat22(9, 0, 0, 10)));
  Diagonal2 d9b_c = Diagonal3 {9, 10};
  EXPECT_TRUE(is_near(d9b_c, mat22(9, 0, 0, 10)));
  Diagonal3 d9c = Diagonal {9, 10};
  EXPECT_TRUE(is_near(d9c, mat22(9, 0, 0, 10)));
  Diagonal3 d9c_b = Diagonal2 {9, 10};
  EXPECT_TRUE(is_near(d9c_b, mat22(9, 0, 0, 10)));
  Diagonal3 d9c_c = Diagonal3 {9, 10};
  EXPECT_TRUE(is_near(d9c_c, mat22(9, 0, 0, 10)));
  //
  Lower l10 {9, 3, 3, 10}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(l10, m));
  Upper u10 {9, 3, 3, 10}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(u10, m));
  Diagonal d10 {9, 10}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d10, mat22(9, 0, 0, 10)));
  Diagonal2 d10b {9, 10}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d10b, mat22(9, 0, 0, 10)));
  Diagonal3 d10c {9, 10}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d10c, mat22(9, 0, 0, 10)));
  //
  l3 = l5; // copy assignment
  EXPECT_TRUE(is_near(l3, M2::Zero()));
  u3 = u5; // copy assignment
  EXPECT_TRUE(is_near(u3, M2::Zero()));
  //
  l5 = Lower {9., 3, 3, 10}; // move assignment
  EXPECT_TRUE(is_near(l5, m));
  u5 = Upper {9., 3, 3, 10}; // move assignment
  EXPECT_TRUE(is_near(u5, m));
  //
  l2 = SelfAdjointMatrix<decltype(M2::Zero()), TriangleType::lower>(M2::Zero()); // copy assignment from compatible sa-matrix
  EXPECT_TRUE(is_near(l2, M2::Zero()));
  u2 = SelfAdjointMatrix<decltype(M2::Zero()), TriangleType::upper>(M2::Zero()); // copy assignment from compatible sa-matrix
  EXPECT_TRUE(is_near(u2, M2::Zero()));
  //
  l2 = Upper {9., 3, 3, 10}; // copy assignment from opposite sa-matrix;
  EXPECT_TRUE(is_near(l2, m));
  u2 = Lower {9., 3, 3, 10}; // copy assignment from opposite sa-matrix;
  EXPECT_TRUE(is_near(u2, m));
  //
  l9 = Diagonal {4., 4}; // copy from a diagonal sa-matrix
  EXPECT_TRUE(is_near(l9, mat22(4, 0, 0, 4)));
  l4 = l9;
  EXPECT_TRUE(is_near(l4, mat22(4, 0, 0, 4)));
  l9 = Diagonal2 {5., 5};
  EXPECT_TRUE(is_near(l9, mat22(5, 0, 0, 5)));
  l9 = Diagonal3 {6., 6};
  EXPECT_TRUE(is_near(l9, mat22(6, 0, 0, 6)));
  u9 = Diagonal {4., 4};
  EXPECT_TRUE(is_near(u9, mat22(4, 0, 0, 4)));
  u4 = u9;
  EXPECT_TRUE(is_near(u4, mat22(4, 0, 0, 4)));
  u9 = Diagonal2 {5., 5};
  EXPECT_TRUE(is_near(u9, mat22(5, 0, 0, 5)));
  u9 = Diagonal3 {6., 6};
  EXPECT_TRUE(is_near(u9, mat22(6, 0, 0, 6)));
  d9 = Diagonal {4., 4};
  EXPECT_TRUE(is_near(d9, mat22(4, 0, 0, 4)));
  d9 = Diagonal2 {5., 5};
  EXPECT_TRUE(is_near(d9, mat22(5, 0, 0, 5)));
  d9 = Diagonal3 {6., 6};
  EXPECT_TRUE(is_near(d9, mat22(6, 0, 0, 6)));
  d9b_b = Diagonal {7., 7};
  d9b = d9b_b;
  EXPECT_TRUE(is_near(d9b, mat22(7, 0, 0, 7)));
  l4 = d9b;
  EXPECT_TRUE(is_near(l4, mat22(7, 0, 0, 7)));
  u4 = d9b;
  EXPECT_TRUE(is_near(u4, mat22(7, 0, 0, 7)));
  d9b = Diagonal {4., 4};
  EXPECT_TRUE(is_near(d9b, mat22(4, 0, 0, 4)));
  d9b = Diagonal2 {5., 5};
  EXPECT_TRUE(is_near(d9b, mat22(5, 0, 0, 5)));
  d9b = Diagonal3 {6., 6};
  EXPECT_TRUE(is_near(d9b, mat22(6, 0, 0, 6)));
  d9c = d9b_b;
  EXPECT_TRUE(is_near(d9c, mat22(7, 0, 0, 7)));
  l4 = d9c;
  EXPECT_TRUE(is_near(l4, mat22(7, 0, 0, 7)));
  u4 = d9c;
  EXPECT_TRUE(is_near(u4, mat22(7, 0, 0, 7)));
  d9c = Diagonal {4., 4};
  EXPECT_TRUE(is_near(d9c, mat22(4, 0, 0, 4)));
  d9c = Diagonal2 {5., 5};
  EXPECT_TRUE(is_near(d9c, mat22(5, 0, 0, 5)));
  d9c = Diagonal3 {6., 6};
  EXPECT_TRUE(is_near(d9c, mat22(6, 0, 0, 6)));
  //
  l3 = MatrixTraits<M2>::zero();
  EXPECT_TRUE(is_near(l3, MatrixTraits<M2>::zero()));
  u3 = MatrixTraits<M2>::zero();
  EXPECT_TRUE(is_near(u3, MatrixTraits<M2>::zero()));
  l3 = MatrixTraits<M2>::identity();
  EXPECT_TRUE(is_near(l3, MatrixTraits<M2>::identity()));
  u3 = MatrixTraits<M2>::identity();
  EXPECT_TRUE(is_near(u3, MatrixTraits<M2>::identity()));
  //
  auto m1 = mat22(9, 3, 3, 10);
  auto sa1 = m1.selfadjointView<Eigen::Lower>();
  auto sa2 = m1.selfadjointView<Eigen::Upper>();
  l2 = sa1; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(l2, m));
  u2 = sa1; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(u2, m));
  //
  l3 = sa2; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(l3, m));
  u3 = sa2; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(u3, m));
  //
  l4 = m1.selfadjointView<Eigen::Lower>(); // assign from rvalue of TriangularBase derived object
  EXPECT_TRUE(is_near(l4, m));
  m1 = mat22(9, 3, 3, 10);
  u4 = m1.selfadjointView<Eigen::Upper>(); // assign from rvalue of TriangularBase derived object
  EXPECT_TRUE(is_near(u4, m));
  //
  l4 = M2::Zero();
  EXPECT_TRUE(is_near(l4, M2::Zero()));
  u4 = M2::Zero();
  EXPECT_TRUE(is_near(u4, M2::Zero()));
  //
  l4 = M2::Identity();
  EXPECT_TRUE(is_near(l4, M2::Identity()));
  u4 = M2::Identity();
  EXPECT_TRUE(is_near(u4, M2::Identity()));
  //
  l4 = d2;
  EXPECT_TRUE(is_near(l4, d2));
  u4 = d2;
  EXPECT_TRUE(is_near(u4, d2));
  //
  l4 = {9., 3, 3, 10}; // assign from a list of scalars
  EXPECT_TRUE(is_near(l4, m));
  u4 = {9., 3, 3, 10}; // assign from a list of scalars
  EXPECT_TRUE(is_near(u4, m));
  //
  l1 += Lower {9., 3, 3, 10};
  EXPECT_TRUE(is_near(l1, mat22(18., 6, 6, 20)));
  u1 += Upper {9., 3, 3, 10};
  EXPECT_TRUE(is_near(u1, mat22(18., 6, 6, 20)));
  //
  l1 -= Upper {9., 3, 3, 10};
  EXPECT_TRUE(is_near(l1, mat22(9., 3, 3, 10)));
  u1 -= Lower {9., 3, 3, 10};
  EXPECT_TRUE(is_near(u1, mat22(9., 3, 3, 10)));
  //
  l1 += Upper {9., 3, 3, 10};
  EXPECT_TRUE(is_near(l1, mat22(18., 6, 6, 20)));
  u1 += Lower {9., 3, 3, 10};
  EXPECT_TRUE(is_near(u1, mat22(18., 6, 6, 20)));
  //
  l1 -= Lower {9., 3, 3, 10};
  EXPECT_TRUE(is_near(l1, mat22(9., 3, 3, 10)));
  u1 -= Upper {9., 3, 3, 10};
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


TEST(eigen3, SelfAdjointMatrix_subscripts)
{
  static_assert(element_gettable<Lower, 2>);
  static_assert(not element_gettable<Lower, 1>);
  static_assert(element_gettable<Upper, 2>);
  static_assert(not element_gettable<Upper, 1>);
  static_assert(element_gettable<Diagonal, 2>);
  static_assert(element_gettable<Diagonal, 1>);
  static_assert(element_gettable<Diagonal2, 2>);
  static_assert(element_gettable<Diagonal2, 1>);
  static_assert(element_gettable<Diagonal3, 2>);
  static_assert(element_gettable<Diagonal3, 1>);

  static_assert(element_settable<Lower, 2>);
  static_assert(not element_settable<Lower, 1>);
  static_assert(element_settable<Upper, 2>);
  static_assert(not element_settable<Upper, 1>);
  static_assert(element_settable<Diagonal, 2>);
  static_assert(element_settable<Diagonal, 1>);
  static_assert(element_settable<Diagonal2, 2>);
  static_assert(element_settable<Diagonal2, 1>);
  static_assert(element_settable<Diagonal3, 2>);
  static_assert(element_settable<Diagonal3, 1>);

  static_assert(not element_settable<const Lower, 2>);
  static_assert(not element_settable<const Lower, 1>);
  static_assert(not element_settable<const Upper, 2>);
  static_assert(not element_settable<const Upper, 1>);
  static_assert(not element_settable<const Diagonal, 2>);
  static_assert(not element_settable<const Diagonal, 1>);
  static_assert(not element_settable<const Diagonal2, 2>);
  static_assert(not element_settable<const Diagonal2, 1>);
  static_assert(not element_settable<const Diagonal3, 2>);
  static_assert(not element_settable<const Diagonal3, 1>);

  static_assert(not element_settable<SelfAdjointMatrix<const M2, TriangleType::lower>, 2>);
  static_assert(not element_settable<SelfAdjointMatrix<const D2, TriangleType::lower>, 2>);
  static_assert(not element_settable<SelfAdjointMatrix<const D2, TriangleType::lower>, 1>);
  static_assert(not element_settable<SelfAdjointMatrix<DiagonalMatrix<const eigen_matrix_t<double, 2, 1>>, TriangleType::lower>, 2>);
  static_assert(not element_settable<SelfAdjointMatrix<DiagonalMatrix<const eigen_matrix_t<double, 2, 1>>, TriangleType::lower>, 1>);

  auto l1 = Lower {9, 3, 3, 10};
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

  auto u1 = Upper {9, 3, 3, 10};
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
  auto d9 = Diagonal {9, 10};
  d9(0, 0) = 7.1;
  EXPECT_NEAR(d9(0), 7.1, 1e-8);
  d9(1) = 8.1;
  EXPECT_NEAR(d9(1, 1), 8.1, 1e-8);
  bool test = false; try { d9(1, 0) = 9.1; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_TRUE(is_near(d9, mat22(7.1, 0, 0, 8.1)));
  //
  auto d9b = Diagonal2 {9, 10};
  d9b(0, 0) = 7.1;
  EXPECT_NEAR(d9b(0, 0), 7.1, 1e-8);
  d9b(1, 1) = 8.1;
  EXPECT_NEAR(d9b(1, 1), 8.1, 1e-8);
  test = false; try { d9b(1, 0) = 9.1; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_TRUE(is_near(d9b, mat22(7.1, 0, 0, 8.1)));
  //
  auto d9c = Diagonal3 {9, 10};
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
  EXPECT_NEAR((Diagonal {9, 10})(0), 9, 1e-6);
  EXPECT_NEAR((Diagonal {9, 10})(1), 10, 1e-6);
  EXPECT_NEAR((Diagonal2 {9, 10})(0), 9, 1e-6);
  EXPECT_NEAR((Diagonal2 {9, 10})(1), 10, 1e-6);
  EXPECT_NEAR((Diagonal3 {9, 10})(0), 9, 1e-6);
  EXPECT_NEAR((Diagonal3 {9, 10})(1), 10, 1e-6);
  EXPECT_NEAR((Diagonal {9, 10})[0], 9, 1e-6);
  EXPECT_NEAR((Diagonal {9, 10})[1], 10, 1e-6);
  EXPECT_NEAR((Diagonal2 {9, 10})[0], 9, 1e-6);
  EXPECT_NEAR((Diagonal2 {9, 10})[1], 10, 1e-6);
  EXPECT_NEAR((Diagonal3 {9, 10})[0], 9, 1e-6);
  EXPECT_NEAR((Diagonal3 {9, 10})[1], 10, 1e-6);
  //
  EXPECT_NEAR((Lower {9., 3, 3, 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((Upper {9., 3, 3, 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((Diagonal {9., 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((Diagonal2 {9., 10})(0, 0), 9, 1e-6);
  EXPECT_NEAR((Diagonal3 {9., 10})(0, 0), 9, 1e-6);
  //
  EXPECT_NEAR((Lower {9., 3, 3, 10})(0, 1), 3, 1e-6);
  EXPECT_NEAR((Upper {9., 3, 3, 10})(0, 1), 3, 1e-6);
  EXPECT_NEAR((Diagonal {9., 10})(0, 1), 0, 1e-6);
  EXPECT_NEAR((Diagonal2 {9., 10})(0, 1), 0, 1e-6);
  EXPECT_NEAR((Diagonal3 {9., 10})(0, 1), 0, 1e-6);
  //
  EXPECT_NEAR((Lower {9., 3, 3, 10})(1, 0), 3, 1e-6);
  EXPECT_NEAR((Upper {9., 3, 3, 10})(1, 0), 3, 1e-6);
  EXPECT_NEAR((Diagonal {9., 10})(1, 0), 0, 1e-6);
  EXPECT_NEAR((Diagonal2 {9., 10})(1, 0), 0, 1e-6);
  EXPECT_NEAR((Diagonal3 {9., 10})(1, 0), 0, 1e-6);
  //
  EXPECT_NEAR((Lower {9., 3, 3, 10})(1, 1), 10, 1e-6);
  EXPECT_NEAR((Upper {9., 3, 3, 10})(1, 1), 10, 1e-6);
  EXPECT_NEAR((Diagonal {9., 10})(1, 1), 10, 1e-6);
  EXPECT_NEAR((Diagonal2 {9., 10})(1, 1), 10, 1e-6);
  EXPECT_NEAR((Diagonal3 {9., 10})(1, 1), 10, 1e-6);
  //
  EXPECT_NEAR((Lower {9., 3, 3, 10}).nested_matrix()(0, 0), 9, 1e-6);
  EXPECT_NEAR((Lower {9., 3, 3, 10}).nested_matrix()(1, 0), 3, 1e-6);
  EXPECT_NEAR((Lower {9., 3, 3, 10}).nested_matrix()(1, 1), 10, 1e-6);
}


TEST(eigen3, SelfAdjointMatrix_make)
{
  static_assert(zero_matrix<decltype(make_EigenSelfAdjointMatrix<TriangleType::upper>(MatrixTraits<M2>::zero()))>);
  static_assert(zero_matrix<decltype(make_EigenSelfAdjointMatrix<TriangleType::lower>(MatrixTraits<M2>::zero()))>);
  static_assert(zero_matrix<decltype(make_EigenSelfAdjointMatrix(MatrixTraits<M2>::zero()))>);
  static_assert(upper_triangular_storage<decltype(make_EigenSelfAdjointMatrix<TriangleType::upper>(make_native_matrix<M2>(9, 3, 3, 10)))>);
  static_assert(lower_triangular_storage<decltype(make_EigenSelfAdjointMatrix<TriangleType::lower>(make_native_matrix<M2>(9, 3, 3, 10)))>);
  static_assert(lower_triangular_storage<decltype(make_EigenSelfAdjointMatrix(make_native_matrix<M2>(9, 3, 3, 10)))>);
  static_assert(diagonal_matrix<decltype(make_EigenSelfAdjointMatrix<TriangleType::upper>(DiagonalMatrix {3., 4}))>);
  static_assert(diagonal_matrix<decltype(make_EigenSelfAdjointMatrix<TriangleType::lower>(DiagonalMatrix {3., 4}))>);
  static_assert(diagonal_matrix<decltype(make_EigenSelfAdjointMatrix(DiagonalMatrix {3., 4}))>);

  static_assert(upper_triangular_storage<decltype(make_EigenSelfAdjointMatrix<TriangleType::upper>(Upper {9, 3, 3, 10}))>);
  static_assert(lower_triangular_storage<decltype(make_EigenSelfAdjointMatrix<TriangleType::lower>(Lower {9, 3, 3, 10}))>);
  static_assert(upper_triangular_storage<decltype(make_EigenSelfAdjointMatrix<TriangleType::upper>(Lower {9, 3, 3, 10}))>);
  static_assert(lower_triangular_storage<decltype(make_EigenSelfAdjointMatrix<TriangleType::lower>(Upper {9, 3, 3, 10}))>);
  static_assert(upper_triangular_storage<decltype(make_EigenSelfAdjointMatrix(Upper {9, 3, 3, 10}))>);
  static_assert(lower_triangular_storage<decltype(make_EigenSelfAdjointMatrix(Lower {9, 3, 3, 10}))>);
}


TEST(eigen3, SelfAdjointMatrix_traits)
{
  M2 m;
  m << 9, 3, 3, 10;
  using Dl = SelfAdjointMatrix<M2, TriangleType::lower>;
  using Du = SelfAdjointMatrix<M2, TriangleType::upper>;
  EXPECT_TRUE(is_near(MatrixTraits<Dl>::make(m), m));
  EXPECT_TRUE(is_near(MatrixTraits<Du>::make(m), m));
  //
  EXPECT_TRUE(is_near(MatrixTraits<Dl>::make(9, 3, 3, 10), m));
  EXPECT_TRUE(is_near(MatrixTraits<Du>::make(9, 3, 3, 10), m));
  //
  EXPECT_TRUE(is_near(MatrixTraits<Dl>::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<Du>::zero(), M2::Zero()));
  //
  EXPECT_TRUE(is_near(MatrixTraits<Dl>::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(MatrixTraits<Du>::identity(), M2::Identity()));

  static_assert(lower_triangular_storage<Lower>);
  static_assert(upper_triangular_storage<Upper>);
  static_assert(diagonal_matrix<Diagonal>);
  static_assert(diagonal_matrix<Diagonal2>);
  static_assert(diagonal_matrix<Diagonal3>);
  static_assert(zero_matrix<decltype(SelfAdjointMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero()))>);
  static_assert(zero_matrix<decltype(SelfAdjointMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero()))>);
  static_assert(identity_matrix<decltype(SelfAdjointMatrix<decltype(MatrixTraits<M2>::identity()), TriangleType::lower>(MatrixTraits<M2>::identity()))>);
}


TEST(eigen3, SelfAdjointMatrix_overloads)
{
  EXPECT_TRUE(is_near(make_native_matrix(Lower(9., 3, 3, 10)), make_native_matrix<M2>(9, 3, 3, 10)));
  EXPECT_TRUE(is_near(make_native_matrix(Upper(9., 3, 3, 10)), make_native_matrix<M2>(9, 3, 3, 10)));
  //
  EXPECT_TRUE(is_near(make_self_contained(Lower(M2::Zero())), make_native_matrix<M2>(0, 0, 0, 0)));
  EXPECT_TRUE(is_near(make_self_contained(Upper(M2::Zero())), make_native_matrix<M2>(0, 0, 0, 0)));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(Lower {9, 3, 3, 10} * 2))>, Lower>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(Upper {9, 3, 3, 10} * 2))>, Upper>);
  //
  //
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<decltype(M2::Identity()), TriangleType::lower>(M2::Identity())), M2::Identity()));
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<decltype(M2::Identity()), TriangleType::upper>(M2::Identity())), M2::Identity()));
  static_assert(identity_matrix<decltype(Cholesky_square(SelfAdjointMatrix<decltype(M2::Identity()), TriangleType::lower>(M2::Identity())))>);
  static_assert(identity_matrix<decltype(Cholesky_square(SelfAdjointMatrix<decltype(M2::Identity()), TriangleType::upper>(M2::Identity())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero())), M2::Zero()));
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero())), M2::Zero()));
  static_assert(zero_matrix<decltype(Cholesky_square(SelfAdjointMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero())))>);
  static_assert(zero_matrix<decltype(Cholesky_square(SelfAdjointMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<decltype(DiagonalMatrix{2, 3}), TriangleType::lower>(DiagonalMatrix{2, 3})), DiagonalMatrix{4, 9}));
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<decltype(DiagonalMatrix{2, 3}), TriangleType::upper>(DiagonalMatrix{2, 3})), DiagonalMatrix{4, 9}));
  static_assert(eigen_diagonal_expr<decltype(Cholesky_square(SelfAdjointMatrix<decltype(DiagonalMatrix{2, 3}), TriangleType::lower>(DiagonalMatrix{2, 3})))>);
  static_assert(eigen_diagonal_expr<decltype(Cholesky_square(SelfAdjointMatrix<decltype(DiagonalMatrix{2, 3}), TriangleType::upper>(DiagonalMatrix{2, 3})))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<M2, TriangleType::diagonal>(make_native_matrix<M2>(3, 0, 1, 3))), DiagonalMatrix{9., 9}));
  static_assert(eigen_diagonal_expr<decltype(Cholesky_square(SelfAdjointMatrix<M2, TriangleType::diagonal>(make_native_matrix<M2>(3, 0, 1, 3))))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<eigen_matrix_t<double, 1, 1>, TriangleType::lower>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(81)));
  EXPECT_TRUE(is_near(Cholesky_square(SelfAdjointMatrix<eigen_matrix_t<double, 1, 1>, TriangleType::upper>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(81)));
  //
  //
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<decltype(M2::Identity()), TriangleType::lower>(M2::Identity())), M2::Identity()));
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<decltype(M2::Identity()), TriangleType::upper>(M2::Identity())), M2::Identity()));
  static_assert(identity_matrix<decltype(Cholesky_factor(SelfAdjointMatrix<decltype(M2::Identity()), TriangleType::lower>(M2::Identity())))>);
  static_assert(identity_matrix<decltype(Cholesky_factor(SelfAdjointMatrix<decltype(M2::Identity()), TriangleType::upper>(M2::Identity())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero())), M2::Zero()));
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero())), M2::Zero()));
  static_assert(zero_matrix<decltype(Cholesky_factor(SelfAdjointMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero())))>);
  static_assert(zero_matrix<decltype(Cholesky_factor(SelfAdjointMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<decltype(DiagonalMatrix{4, 9}), TriangleType::lower>(DiagonalMatrix{4, 9})), DiagonalMatrix{2, 3}));
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<decltype(DiagonalMatrix{4, 9}), TriangleType::upper>(DiagonalMatrix{4, 9})), DiagonalMatrix{2, 3}));
  static_assert(eigen_diagonal_expr<decltype(Cholesky_factor(SelfAdjointMatrix<decltype(DiagonalMatrix{4, 9}), TriangleType::lower>(DiagonalMatrix{4, 9})))>);
  static_assert(eigen_diagonal_expr<decltype(Cholesky_factor(SelfAdjointMatrix<decltype(DiagonalMatrix{4, 9}), TriangleType::upper>(DiagonalMatrix{4, 9})))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<M2, TriangleType::diagonal>(make_native_matrix<M2>(9, 3, 3, 9))), DiagonalMatrix{3., 3}));
  static_assert(eigen_diagonal_expr<decltype(Cholesky_factor(SelfAdjointMatrix<M2, TriangleType::diagonal>(make_native_matrix<M2>(9, 3, 3, 9))))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<eigen_matrix_t<double, 1, 1>, TriangleType::lower>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(3)));
  EXPECT_TRUE(is_near(Cholesky_factor(SelfAdjointMatrix<eigen_matrix_t<double, 1, 1>, TriangleType::upper>(eigen_matrix_t<double, 1, 1>(9))), eigen_matrix_t<double, 1, 1>(3)));
  //
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(Upper {9., 3, 3, 10}), mat22(3., 0, 1, 3)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(Lower {9., 3, 3, 10}), mat22(3., 1, 0, 3)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(Lower {9., 3, 3, 10}), mat22(3., 0, 1, 3)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(Upper {9., 3, 3, 10}), mat22(3., 1, 0, 3)));
  static_assert(lower_triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(Upper {9., 3, 3, 10}))>);
  static_assert(upper_triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(Lower {9., 3, 3, 10}))>);
  static_assert(lower_triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(Lower {9., 3, 3, 10}))>);
  static_assert(upper_triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(Upper {9., 3, 3, 10}))>);
  //
  // Semidefinite case:
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(Upper {9., 3, 3, 1}), mat22(3., 0, 1, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(Lower {9., 3, 3, 1}), mat22(3., 1, 0, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(Lower {9., 3, 3, 1}), mat22(3., 0, 1, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(Upper {9., 3, 3, 1}), mat22(3., 1, 0, 0)));
  static_assert(lower_triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(Upper {9., 3, 3, 1}))>);
  static_assert(upper_triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(Lower {9., 3, 3, 1}))>);
  static_assert(lower_triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(Lower {9., 3, 3, 1}))>);
  static_assert(upper_triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(Upper {9., 3, 3, 1}))>);

  // Constant semidefinite case:
  using Const922 = ConstantMatrix<double, 9, 2, 2>;
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<Const922, TriangleType::lower> {Const922 {}}), mat22(3., 0, 3, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<Const922, TriangleType::upper> {Const922 {}}), mat22(3., 3, 0, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<Const922, TriangleType::lower> {Const922 {}}), mat22(3., 0, 3, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<Const922, TriangleType::upper> {Const922 {}}), mat22(3., 3, 0, 0)));
  static_assert(lower_triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<Const922, TriangleType::lower> {Const922 {}}))>);
  static_assert(upper_triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<Const922, TriangleType::upper> {Const922 {}}))>);
  static_assert(lower_triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<Const922, TriangleType::upper> {Const922 {}}))>);
  static_assert(upper_triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<Const922, TriangleType::lower> {Const922 {}}))>);

  using M2Const = typename M2::ConstantReturnType;
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<M2Const, TriangleType::lower>(M2::Constant(9))), mat22(3., 0, 3, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<M2Const, TriangleType::upper>(M2::Constant(9))), mat22(3., 3, 0, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<M2Const, TriangleType::lower>(M2::Constant(9))), mat22(3., 0, 3, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<M2Const, TriangleType::upper>(M2::Constant(9))), mat22(3., 3, 0, 0)));
  static_assert(lower_triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<M2Const, TriangleType::lower>(M2::Constant(9))))>);
  static_assert(upper_triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<M2Const, TriangleType::upper>(M2::Constant(9))))>);
  static_assert(lower_triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(SelfAdjointMatrix<M2Const, TriangleType::upper>(M2::Constant(9))))>);
  static_assert(upper_triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(SelfAdjointMatrix<M2Const, TriangleType::lower>(M2::Constant(9))))>);

  // Zero (positive and negative semidefinite) case:
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(Upper {0., 0, 0, 0}), mat22(0., 0, 0, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(Lower {0., 0, 0, 0}), mat22(0., 0, 0, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::lower>(Lower {0., 0, 0, 0}), mat22(0., 0, 0, 0)));
  EXPECT_TRUE(is_near(Cholesky_factor<TriangleType::upper>(Upper {0., 0, 0, 0}), mat22(0., 0, 0, 0)));
  static_assert(lower_triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(Upper {0., 0, 0, 0}))>);
  static_assert(upper_triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(Lower {0., 0, 0, 0}))>);
  static_assert(lower_triangular_matrix<decltype(Cholesky_factor<TriangleType::lower>(Lower {0., 0, 0, 0}))>);
  static_assert(upper_triangular_matrix<decltype(Cholesky_factor<TriangleType::upper>(Upper {0., 0, 0, 0}))>);
  //
  //
  EXPECT_TRUE(is_near(diagonal_of(Lower {9., 3, 3, 10}), make_native_matrix<double, 2, 1>(9., 10)));
  EXPECT_TRUE(is_near(diagonal_of(Upper {9., 3, 3, 10}), make_native_matrix<double, 2, 1>(9., 10)));
  //
  EXPECT_TRUE(is_near(transpose(Lower {9., 3, 3, 10}), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(transpose(Upper {9., 3, 3, 10}), mat22(9., 3, 3, 10)));
  //
  EXPECT_TRUE(is_near(adjoint(Lower {9., 3, 3, 10}), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(adjoint(Upper {9., 3, 3, 10}), mat22(9., 3, 3, 10)));
  //
  EXPECT_NEAR(determinant(Lower {9., 3, 3, 10}), 81, 1e-6);
  EXPECT_NEAR(determinant(Upper {9., 3, 3, 10}), 81, 1e-6);
  //
  EXPECT_NEAR(trace(Lower {9., 3, 3, 10}), 19, 1e-6);
  EXPECT_NEAR(trace(Upper {9., 3, 3, 10}), 19, 1e-6);
  //
  EXPECT_TRUE(is_near(solve(Lower {9., 3, 3, 10}, make_native_matrix<double, 2, 1>(15, 23)), make_native_matrix(1., 2)));
  EXPECT_TRUE(is_near(solve(Upper {9., 3, 3, 10}, make_native_matrix<double, 2, 1>(15, 23)), make_native_matrix(1., 2)));
  //
  EXPECT_TRUE(is_near(reduce_columns(Lower {9., 3, 3, 10}), make_native_matrix(6., 6.5)));
  EXPECT_TRUE(is_near(reduce_columns(Upper {9., 3, 3, 10}), make_native_matrix(6., 6.5)));
  //
  auto sl1 = Lower {9., 3, 3, 10};
  rank_update(sl1, make_native_matrix<M2>(2, 0, 1, 2), 4);
  EXPECT_TRUE(is_near(sl1, mat22(25., 11, 11, 30)));
  auto su1 = Upper {9., 3, 3, 10};
  rank_update(su1, make_native_matrix<M2>(2, 1, 0, 2), 4);
  EXPECT_TRUE(is_near(su1, mat22(29., 11, 11, 26)));
  //
  const auto sl2 = Lower {9., 3, 3, 10};
  EXPECT_TRUE(is_near(rank_update(sl2, make_native_matrix<M2>(2, 0, 1, 2), 4), mat22(25., 11, 11, 30)));
  const auto su2 = Upper {9., 3, 3, 10};
  EXPECT_TRUE(is_near(rank_update(su2, make_native_matrix<M2>(2, 1, 0, 2), 4), mat22(29., 11, 11, 26)));
  //
  EXPECT_TRUE(is_near(rank_update(Lower {9., 3, 3, 10}, make_native_matrix<M2>(2, 0, 1, 2), 4), mat22(25., 11, 11, 30)));
  EXPECT_TRUE(is_near(rank_update(Upper {9., 3, 3, 10}, make_native_matrix<M2>(2, 1, 0, 2), 4), mat22(29., 11, 11, 26)));
  EXPECT_TRUE(is_near(rank_update(Diagonal {9., 10}, make_native_matrix<M2>(2, 0, 1, 2), 4), mat22(25., 8, 8, 30)));
  EXPECT_TRUE(is_near(rank_update(Diagonal2 {9., 10}, make_native_matrix<M2>(2, 0, 1, 2), 4), mat22(25., 8, 8, 30)));
  EXPECT_TRUE(is_near(rank_update(Diagonal3 {9., 10}, make_native_matrix<M2>(2, 0, 1, 2), 4), mat22(25., 8, 8, 30)));
  //
  EXPECT_TRUE(is_near(solve(Lower {9., 3, 3, 10}, make_native_matrix<double, 2, 1>(15, 23)), make_native_matrix<double, 2, 1>(1, 2)));
  EXPECT_TRUE(is_near(solve(Upper {9., 3, 3, 10}, make_native_matrix<double, 2, 1>(15, 23)), make_native_matrix<double, 2, 1>(1, 2)));
  //
  EXPECT_TRUE(is_near(Cholesky_square(LQ_decomposition(Lower {9., 3, 3, 10})), mat22(90., 57, 57, 109)));
  EXPECT_TRUE(is_near(Cholesky_square(LQ_decomposition(Upper {9., 3, 3, 10})), mat22(90., 57, 57, 109)));
  //
  EXPECT_TRUE(is_near(Cholesky_square(QR_decomposition(Lower {9., 3, 3, 10})), mat22(90., 57, 57, 109)));
  EXPECT_TRUE(is_near(Cholesky_square(QR_decomposition(Upper {9., 3, 3, 10})), mat22(90., 57, 57, 109)));
}


TEST(eigen3, SelfAdjointMatrix_blocks_lower)
{
  auto m0 = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::lower> {1, 2, 3,
                                                                                      2, 4, 5,
                                                                                      3, 5, 6};
  auto m1 = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::lower> {4, 5, 6,
                                                                                      5, 7, 8,
                                                                                      6, 8, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}, m1),
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1., 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}));
  static_assert(lower_triangular_storage<decltype(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}, m1))>);

  EXPECT_TRUE(is_near(concatenate_vertical(m0, m1),
    make_native_matrix<6,3>(1., 2, 3,
                                   2, 4, 5,
                                   3, 5, 6,
                                   4, 5, 6,
                                   5, 7, 8,
                                   6, 8, 9)));
  EXPECT_TRUE(is_near(concatenate_horizontal(m0, m1),
    make_native_matrix<3,6>(1., 2, 3, 4, 5, 6,
                                   2, 4, 5, 5, 7, 8,
                                   3, 5, 6, 6, 8, 9)));
  EXPECT_TRUE(is_near(split_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1., 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
      std::tuple {SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}, m1}));
  const auto a1 = SelfAdjointMatrix<const eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1., 2, 0, 0, 0,
                                                                                                  2, 3, 0, 0, 0,
                                                                                                  0, 0, 4, 5, 6,
                                                                                                  0, 0, 5, 7, 8,
                                                                                                  0, 0, 6, 8, 9};
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(a1), std::tuple {SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}, m1}));
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
    std::tuple {make_native_matrix<2,5>(1., 2, 0, 0, 0,
                                               2, 3, 0, 0, 0),
               make_native_matrix<3,5>(0., 0, 4, 5, 6,
                                               0, 0, 5, 7, 8,
                                               0, 0, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1, 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {make_native_matrix<2,5>(1., 2, 0, 0, 0,
                                               2, 3, 0, 0, 0),
               make_native_matrix<2,5>(0., 0, 4, 5, 6,
                                               0, 0, 5, 7, 8)}));
  EXPECT_TRUE(is_near(split_horizontal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<2, 3>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1, 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {make_native_matrix<5,2>(1., 2, 2, 3, 0, 0, 0, 0, 0, 0),
               make_native_matrix<5,3>(0., 0, 0, 0, 0, 0, 4, 5, 6, 5, 7, 8, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1, 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {make_native_matrix<5,2>(1., 2, 2, 3, 0, 0, 0, 0, 0, 0),
               make_native_matrix<5,2>(0., 0, 0, 0, 4, 5, 5, 7, 6, 8)}));

  EXPECT_TRUE(is_near(column(m1, 2), make_native_matrix(6., 8, 9)));
  EXPECT_TRUE(is_near(column<1>(m1), make_native_matrix(5., 7, 8)));
  EXPECT_TRUE(is_near(apply_columnwise(m1, [](const auto& col){ return make_self_contained(col + col.Constant(1)); }),
    make_native_matrix<double, 3, 3>(
      5, 6, 7,
      6, 8, 9,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_columnwise(m1, [](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }),
    make_native_matrix<double, 3, 3>(
      4, 6, 8,
      5, 8, 10,
      6, 9, 11)));
  EXPECT_TRUE(is_near(apply_coefficientwise(m1, [](const auto& x){ return x + 1; }),
    make_native_matrix<double, 3, 3>(
      5, 6, 7,
      6, 8, 9,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_coefficientwise(m1, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }),
    make_native_matrix<double, 3, 3>(
      4, 6, 8,
      6, 9, 11,
      8, 11, 13)));
}


TEST(eigen3, SelfAdjointMatrix_blocks_upper)
{
  auto m0 = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::upper> {1, 2, 3,
                                                                                      2, 4, 5,
                                                                                      3, 5, 6};
  auto m1 = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::upper> {4., 5, 6,
                                                                                      5, 7, 8,
                                                                                      6, 8, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}, m1),
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}));
  static_assert(upper_triangular_storage<decltype(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}, m1))>);

  EXPECT_TRUE(is_near(concatenate_vertical(m0, m1),
    make_native_matrix<6,3>(1., 2, 3,
                                    2, 4, 5,
                                    3, 5, 6,
                                    4, 5, 6,
                                    5, 7, 8,
                                    6, 8, 9)));
  EXPECT_TRUE(is_near(concatenate_horizontal(m0, m1),
    make_native_matrix<3,6>(1., 2, 3, 4, 5, 6,
                                    2, 4, 5, 5, 7, 8,
                                    3, 5, 6, 6, 8, 9)));
  EXPECT_TRUE(is_near(split_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}, m1}));
  const auto a1 = SelfAdjointMatrix<const eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                                                  2, 3, 0, 0, 0,
                                                                                                  0, 0, 4, 5, 6,
                                                                                                  0, 0, 5, 7, 8,
                                                                                                  0, 0, 6, 8, 9};
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(a1), std::tuple {SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}, m1}));
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
    std::tuple {make_native_matrix<2,5>(1., 2, 0, 0, 0,
                                               2, 3, 0, 0, 0),
               make_native_matrix<3,5>(0., 0, 4, 5, 6,
                                               0, 0, 5, 7, 8,
                                               0, 0, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {make_native_matrix<2,5>(1., 2, 0, 0, 0,
                                               2, 3, 0, 0, 0),
               make_native_matrix<2,5>(0., 0, 4, 5, 6,
                                               0, 0, 5, 7, 8)}));
  EXPECT_TRUE(is_near(split_horizontal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple {}));
  EXPECT_TRUE(is_near(split_horizontal<2, 3>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {make_native_matrix<5,2>(1., 2, 2, 3, 0, 0, 0, 0, 0, 0),
               make_native_matrix<5,3>(0., 0, 0, 0, 0, 0, 4, 5, 6, 5, 7, 8, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}),
    std::tuple {make_native_matrix<5,2>(1., 2, 2, 3, 0, 0, 0, 0, 0, 0),
               make_native_matrix<5,2>(0., 0, 0, 0, 4, 5, 5, 7, 6, 8)}));
  EXPECT_TRUE(is_near(column(m1, 2), make_native_matrix(6., 8, 9)));
  EXPECT_TRUE(is_near(column<1>(m1), make_native_matrix(5., 7, 8)));
  EXPECT_TRUE(is_near(apply_columnwise(m1, [](const auto& col){ return make_self_contained(col + col.Constant(1)); }),
    make_native_matrix<double, 3, 3>(
      5, 6, 7,
      6, 8, 9,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_columnwise(m1, [](const auto& col, std::size_t i){ return make_self_contained(col + col.Constant(i)); }),
    make_native_matrix<double, 3, 3>(
      4, 6, 8,
      5, 8, 10,
      6, 9, 11)));
  EXPECT_TRUE(is_near(apply_coefficientwise(m1, [](const auto& x){ return x + 1; }),
    make_native_matrix<double, 3, 3>(
      5, 6, 7,
      6, 8, 9,
      7, 9, 10)));
  EXPECT_TRUE(is_near(apply_coefficientwise(m1, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }),
    make_native_matrix<double, 3, 3>(
      4, 6, 8,
      6, 9, 11,
      8, 11, 13)));
}


TEST(eigen3, SelfAdjointMatrix_blocks_mixed)
{
  auto m0 = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::upper> {1, 2, 3,
                                                                                      2, 4, 5,
                                                                                      3, 5, 6};
  auto m1 = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::lower> {4., 5, 6,
                                                                                      5, 7, 8,
                                                                                      6, 8, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}, m1),
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}));
  static_assert(upper_triangular_storage<decltype(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}, m1))>);
  static_assert(lower_triangular_storage<decltype(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}, m0))>);

  EXPECT_TRUE(is_near(concatenate_vertical(m0, m1),
    make_native_matrix<6,3>(1., 2, 3,
                                    2, 4, 5,
                                    3, 5, 6,
                                    4, 5, 6,
                                    5, 7, 8,
                                    6, 8, 9)));
  EXPECT_TRUE(is_near(concatenate_horizontal(m0, m1),
    make_native_matrix<3,6>(1., 2, 3, 4, 5, 6,
                                    2, 4, 5, 5, 7, 8,
                                    3, 5, 6, 6, 8, 9)));
  auto m2 = SelfAdjointMatrix<eigen_matrix_t<double, 3, 3>, TriangleType::upper> {4., 5, 6,
                                                                                      5, 7, 8,
                                                                                      6, 8, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}, m2),
    SelfAdjointMatrix<eigen_matrix_t<double, 5, 5>, TriangleType::lower> {1., 2, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 5, 6,
                                                                              0, 0, 5, 7, 8,
                                                                              0, 0, 6, 8, 9}));
}


TEST(eigen3, SelfAdjointMatrix_arithmetic_lower)
{
  auto m1 = Lower {4., 5, 5, 6};
  auto m2 = Lower {1., 2, 2, 3};
  auto d = DiagonalMatrix<eigen_matrix_t<double, 2, 1>> {1, 3};
  auto i = M2::Identity();
  auto z = ZeroMatrix<double, 2, 2> {};

  EXPECT_TRUE(is_near(m1 + m2, mat22(5, 7, 7, 9))); static_assert(self_adjoint_matrix<decltype(m1 + m2)>);
  EXPECT_TRUE(is_near(m1 + d, mat22(5, 5, 5, 9))); static_assert(self_adjoint_matrix<decltype(m1 + d)>);
  EXPECT_TRUE(is_near(d + m1, mat22(5, 5, 5, 9))); static_assert(self_adjoint_matrix<decltype(d + m1)>);
  EXPECT_TRUE(is_near(m1 + i, mat22(5, 5, 5, 7))); static_assert(self_adjoint_matrix<decltype(m1 + i)>);
  EXPECT_TRUE(is_near(i + m1, mat22(5, 5, 5, 7))); static_assert(self_adjoint_matrix<decltype(i + m1)>);
  EXPECT_TRUE(is_near(m1 + z, mat22(4, 5, 5, 6))); static_assert(self_adjoint_matrix<decltype(m1 + z)>);
  EXPECT_TRUE(is_near(z + m1, mat22(4, 5, 5, 6))); static_assert(self_adjoint_matrix<decltype(z + m1)>);

  EXPECT_TRUE(is_near(m1 - m2, mat22(3, 3, 3, 3))); static_assert(self_adjoint_matrix<decltype(m1 - m2)>);
  EXPECT_TRUE(is_near(m1 - d, mat22(3, 5, 5, 3))); static_assert(self_adjoint_matrix<decltype(m1 - d)>);
  EXPECT_TRUE(is_near(d - m1, mat22(-3, -5, -5, -3))); static_assert(self_adjoint_matrix<decltype(d - m1)>);
  EXPECT_TRUE(is_near(m1 - i, mat22(3, 5, 5, 5))); static_assert(self_adjoint_matrix<decltype(m1 - i)>);
  EXPECT_TRUE(is_near(i - m1, mat22(-3, -5, -5, -5))); static_assert(self_adjoint_matrix<decltype(i - m1)>);
  EXPECT_TRUE(is_near(m1 - z, mat22(4, 5, 5, 6))); static_assert(self_adjoint_matrix<decltype(m1 - z)>);
  EXPECT_TRUE(is_near(z - m1, mat22(-4, -5, -5, -6))); static_assert(self_adjoint_matrix<decltype(z - m1)>);

  EXPECT_TRUE(is_near(m1 * 2, mat22(8, 10, 10, 12))); static_assert(self_adjoint_matrix<decltype(m1 * 2)>);
  EXPECT_TRUE(is_near(2 * m1, mat22(8, 10, 10, 12))); static_assert(self_adjoint_matrix<decltype(2 * m1)>);
  EXPECT_TRUE(is_near(m1 / 2, mat22(2, 2.5, 2.5, 3))); static_assert(self_adjoint_matrix<decltype(m1 / 2)>);
  static_assert(self_adjoint_matrix<decltype(m1 / 0)>);
  EXPECT_TRUE(is_near(-m1, mat22(-4, -5, -5, -6)));  static_assert(self_adjoint_matrix<decltype(-m1)>);

  EXPECT_TRUE(is_near(SelfAdjointMatrix<decltype(i), TriangleType::diagonal> {i} * 2, mat22(2, 0, 0, 2)));
  static_assert(diagonal_matrix<decltype(SelfAdjointMatrix<decltype(i), TriangleType::diagonal> {i} * 2)>);
  EXPECT_TRUE(is_near(2 * SelfAdjointMatrix<decltype(i), TriangleType::diagonal> {i}, mat22(2, 0, 0, 2)));
  static_assert(diagonal_matrix<decltype(2 * SelfAdjointMatrix<decltype(i), TriangleType::diagonal> {i})>);
  EXPECT_TRUE(is_near(SelfAdjointMatrix<decltype(i), TriangleType::diagonal> {i} / 0.5, mat22(2, 0, 0, 2)));
  static_assert(diagonal_matrix<decltype(SelfAdjointMatrix<decltype(i), TriangleType::diagonal> {i} / 2)>);

  EXPECT_TRUE(is_near(m1 * m2, mat22(14, 23, 17, 28)));
  EXPECT_TRUE(is_near(m1 * d, mat22(4, 15, 5, 18)));
  EXPECT_TRUE(is_near(d * m1, mat22(4, 5, 15, 18)));
  EXPECT_TRUE(is_near(m1 * i, m1));  static_assert(self_adjoint_matrix<decltype(m1 * i)>);
  EXPECT_TRUE(is_near(i * m1, m1));  static_assert(self_adjoint_matrix<decltype(i * m1)>);
  EXPECT_TRUE(is_near(m1 * z, z));  static_assert(zero_matrix<decltype(m1 * z)>);
  EXPECT_TRUE(is_near(z * m1, z));  static_assert(zero_matrix<decltype(z * m1)>);
  EXPECT_TRUE(is_near(MatrixTraits<Lower>::make(mat22(1, 2, 3, 4) * (m1 * mat22(1, 3, 2, 4))), mat22(48, 110, 110, 252)));

  auto tl1 = TriangularMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {2, 0, 1, 2};
  auto tu1 = TriangularMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {2, 1, 0, 2};
  EXPECT_TRUE(is_near(m1 * tl1, mat22(13, 10, 16, 12)));
  EXPECT_TRUE(is_near(m1 * tu1, mat22(8, 14, 10, 17)));
  EXPECT_TRUE(is_near(tl1 * m1, mat22(8, 10, 14, 17)));
  EXPECT_TRUE(is_near(tu1 * m1, mat22(13, 16, 10, 12)));
  EXPECT_TRUE(is_near(mat22(4, 5, 5, 6) * m2, mat22(14., 23, 17, 28)));
  EXPECT_TRUE(is_near(m1 * mat22(1, 2, 2, 3), mat22(14., 23, 17, 28)));
}


TEST(eigen3, SelfAdjointMatrix_arithmetic_upper)
{
  auto m1 = Upper {4., 5, 5, 6};
  auto m2 = Upper {1., 2, 2, 3};
  auto d = DiagonalMatrix<eigen_matrix_t<double, 2, 1>> {1, 3};
  auto i = M2::Identity();
  auto z = ZeroMatrix<double, 2, 2> {};
  EXPECT_TRUE(is_near(m1 + m2, mat22(5, 7, 7, 9))); static_assert(self_adjoint_matrix<decltype(m1 + m2)>);
  EXPECT_TRUE(is_near(m1 + d, mat22(5, 5, 5, 9))); static_assert(self_adjoint_matrix<decltype(m1 + d)>);
  EXPECT_TRUE(is_near(d + m1, mat22(5, 5, 5, 9))); static_assert(self_adjoint_matrix<decltype(d + m1)>);
  EXPECT_TRUE(is_near(m1 + i, mat22(5, 5, 5, 7))); static_assert(self_adjoint_matrix<decltype(m1 + i)>);
  EXPECT_TRUE(is_near(i + m1, mat22(5, 5, 5, 7))); static_assert(self_adjoint_matrix<decltype(i + m1)>);
  EXPECT_TRUE(is_near(m1 + z, mat22(4, 5, 5, 6))); static_assert(self_adjoint_matrix<decltype(m1 + z)>);
  EXPECT_TRUE(is_near(z + m1, mat22(4, 5, 5, 6))); static_assert(self_adjoint_matrix<decltype(z + m1)>);

  EXPECT_TRUE(is_near(m1 - m2, mat22(3, 3, 3, 3))); static_assert(self_adjoint_matrix<decltype(m1 - m2)>);
  EXPECT_TRUE(is_near(m1 - d, mat22(3, 5, 5, 3))); static_assert(self_adjoint_matrix<decltype(m1 - d)>);
  EXPECT_TRUE(is_near(d - m1, mat22(-3, -5, -5, -3))); static_assert(self_adjoint_matrix<decltype(d - m1)>);
  EXPECT_TRUE(is_near(m1 - i, mat22(3, 5, 5, 5))); static_assert(self_adjoint_matrix<decltype(m1 - i)>);
  EXPECT_TRUE(is_near(i - m1, mat22(-3, -5, -5, -5))); static_assert(self_adjoint_matrix<decltype(i - m1)>);
  EXPECT_TRUE(is_near(m1 - z, mat22(4, 5, 5, 6))); static_assert(self_adjoint_matrix<decltype(m1 - z)>);
  EXPECT_TRUE(is_near(z - m1, mat22(-4, -5, -5, -6))); static_assert(self_adjoint_matrix<decltype(z - m1)>);

  EXPECT_TRUE(is_near(m1 * 2, mat22(8, 10, 10, 12))); static_assert(self_adjoint_matrix<decltype(m1 * 2)>);
  EXPECT_TRUE(is_near(2 * m1, mat22(8, 10, 10, 12))); static_assert(self_adjoint_matrix<decltype(2 * m1)>);
  EXPECT_TRUE(is_near(m1 / 2, mat22(2, 2.5, 2.5, 3))); static_assert(self_adjoint_matrix<decltype(m1 / 2)>);
  static_assert(self_adjoint_matrix<decltype(m1 / 0)>);
  EXPECT_TRUE(is_near(-m1, mat22(-4, -5, -5, -6)));  static_assert(self_adjoint_matrix<decltype(-m1)>);

  EXPECT_TRUE(is_near(m1 * m2, mat22(14, 23, 17, 28)));
  EXPECT_TRUE(is_near(m1 * d, mat22(4, 15, 5, 18)));
  EXPECT_TRUE(is_near(d * m1, mat22(4, 5, 15, 18)));
  EXPECT_TRUE(is_near(m1 * i, m1));  static_assert(self_adjoint_matrix<decltype(m1 * i)>);
  EXPECT_TRUE(is_near(i * m1, m1));  static_assert(self_adjoint_matrix<decltype(i * m1)>);
  EXPECT_TRUE(is_near(m1 * z, z));  static_assert(zero_matrix<decltype(m1 * z)>);
  EXPECT_TRUE(is_near(z * m1, z));  static_assert(zero_matrix<decltype(z * m1)>);

  auto tl1 = TriangularMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {2, 0, 1, 2};
  auto tu1 = TriangularMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {2, 1, 0, 2};
  EXPECT_TRUE(is_near(m1 * tl1, mat22(13, 10, 16, 12)));
  EXPECT_TRUE(is_near(m1 * tu1, mat22(8, 14, 10, 17)));
  EXPECT_TRUE(is_near(tl1 * m1, mat22(8, 10, 14, 17)));
  EXPECT_TRUE(is_near(tu1 * m1, mat22(13, 16, 10, 12)));
  EXPECT_TRUE(is_near(mat22(4, 5, 5, 6) * m2, mat22(14., 23, 17, 28)));
  EXPECT_TRUE(is_near(m1 * mat22(1, 2, 2, 3), mat22(14., 23, 17, 28)));
}


TEST(eigen3, SelfAdjointMatrix_arithmetic_mixed)
{
  auto m1 = Upper {4., 5, 5, 6};
  auto m2 = Lower {1., 2, 2, 3};
  auto tl1 = TriangularMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::lower> {2, 0, 1, 2};
  auto tu1 = TriangularMatrix<eigen_matrix_t<double, 2, 2>, TriangleType::upper> {2, 1, 0, 2};
  EXPECT_TRUE(is_near(m1 + m2, mat22(5., 7, 7, 9))); static_assert(upper_triangular_storage<decltype(m1 + m2)>);
  EXPECT_TRUE(is_near(m2 + m1, mat22(5., 7, 7, 9))); static_assert(lower_triangular_storage<decltype(m2 + m1)>);
  EXPECT_TRUE(is_near(m1 - m2, mat22(3, 3, 3, 3))); static_assert(upper_triangular_storage<decltype(m1 - m2)>);
  EXPECT_TRUE(is_near(m2 - m1, mat22(-3, -3, -3, -3))); static_assert(lower_triangular_storage<decltype(m2 - m1)>);
  EXPECT_TRUE(is_near(m1 * m2, mat22(14, 23, 17, 28)));
  EXPECT_TRUE(is_near(m2 * m1, mat22(14, 17, 23, 28)));

  EXPECT_TRUE(is_near(m1 * tl1, mat22(13, 10, 16, 12)));
  EXPECT_TRUE(is_near(m1 * tu1, mat22(8, 14, 10, 17)));
  EXPECT_TRUE(is_near(m2 * tl1, mat22(4, 4, 7, 6)));
  EXPECT_TRUE(is_near(m2 * tu1, mat22(2, 5, 4, 8)));
  EXPECT_TRUE(is_near(tl1 * m1, mat22(8, 10, 14, 17)));
  EXPECT_TRUE(is_near(tu1 * m1, mat22(13, 16, 10, 12)));
  EXPECT_TRUE(is_near(tl1 * m2, mat22(2, 4, 5, 8)));
  EXPECT_TRUE(is_near(tu1 * m2, mat22(4, 7, 4, 6)));
}


TEST(eigen3, SelfAdjointMatrix_references)
{
  M2 m, n;
  m << 4, 2, 2, 5;
  n << 9, 3, 3, 10;
  SelfAdjointMatrix<M2, TriangleType::lower> x {m};
  SelfAdjointMatrix<M2&, TriangleType::lower> x_lvalue = x;
  EXPECT_TRUE(is_near(x_lvalue, m));
  x = SelfAdjointMatrix<M2, TriangleType::lower> {n};
  EXPECT_TRUE(is_near(x_lvalue, n));
  x_lvalue = SelfAdjointMatrix<M2, TriangleType::lower> {m};
  EXPECT_TRUE(is_near(x, m));
  EXPECT_TRUE(is_near(SelfAdjointMatrix<M2&, TriangleType::lower> {m}.nested_matrix(), mat22(4, 2, 2, 5)));
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

