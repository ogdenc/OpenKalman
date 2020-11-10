/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen3.gtest.hpp"

using namespace OpenKalman;

using M2 = Eigen::Matrix<double, 2, 2>;
using D2 = DiagonalMatrix<Eigen::Matrix<double, 2, 1>>;
using Lower = TriangularMatrix<M2, TriangleType::lower>;
using Upper = TriangularMatrix<M2, TriangleType::upper>;
using Diagonal = TriangularMatrix<M2, TriangleType::diagonal>;
using Diagonal2 = TriangularMatrix<D2, TriangleType::diagonal>;
using Diagonal3 = TriangularMatrix<D2, TriangleType::lower>;

template<typename...Args>
inline auto mat22(Args...args) { return MatrixTraits<M2>::make(args...); }

TEST_F(eigen3, TriangularMatrix_class)
{
  M2 ml, mu;
  ml << 3, 0, 1, 3;
  mu << 3, 1, 0, 3;
  //
  Lower l1;
  l1 << 3, 0, 1, 3;
  EXPECT_TRUE(is_near(l1.base_matrix(), ml));
  Upper u1;
  u1 << 3, 1, 0, 3;
  EXPECT_TRUE(is_near(u1.base_matrix(), mu));
  Diagonal d1;
  d1 << 3, 3;
  EXPECT_TRUE(is_near(d1.base_matrix(), mat22(3, 0, 0, 3)));
  EXPECT_TRUE(is_near(d1, mat22(3, 0, 0, 3)));
  d1.template triangularView<Eigen::Lower>() = (M2() << 2, 5, 6, 2).finished();
  EXPECT_TRUE(is_near(d1, mat22(2, 0, 0, 2)));
  Diagonal2 d1b;
  d1b << 3, 3;
  EXPECT_TRUE(is_near(d1b.base_matrix(), mat22(3, 0, 0, 3)));
  EXPECT_TRUE(is_near(d1b, mat22(3, 0, 0, 3)));
  d1b.template triangularView<Eigen::Lower>() = (M2() << 2, 5, 6, 2).finished();
  EXPECT_TRUE(is_near(d1b, mat22(2, 0, 0, 2)));
  Diagonal3 d1c;
  d1c << 3, 3;
  EXPECT_TRUE(is_near(d1c.base_matrix(), mat22(3, 0, 0, 3)));
  EXPECT_TRUE(is_near(d1c, mat22(3, 0, 0, 3)));
  d1c.template triangularView<Eigen::Lower>() = (M2() << 2, 5, 6, 2).finished();
  EXPECT_TRUE(is_near(d1c, mat22(2, 0, 0, 2)));
  //
  Lower l2 = (M2() << 3, 0, 1, 3).finished();
  EXPECT_TRUE(is_near(l1, l2));
  Upper u2 = (M2() << 3, 1, 0, 3).finished();
  EXPECT_TRUE(is_near(u1, u2));
  //
  EXPECT_TRUE(is_near(Lower(DiagonalMatrix {3., 4}), (M2() << 3, 0, 0, 4).finished()));
  EXPECT_TRUE(is_near(Upper(DiagonalMatrix {3., 4}), (M2() << 3, 0, 0, 4).finished()));
  //
  EXPECT_TRUE(is_near(Lower(MatrixTraits<M2>::zero()), M2::Zero()));
  EXPECT_TRUE(is_near(Upper(MatrixTraits<M2>::zero()), M2::Zero()));
  //
  Lower l3(l2); // copy constructor
  EXPECT_TRUE(is_near(l3, ml));
  Upper u3(u2); // copy constructor
  EXPECT_TRUE(is_near(u3, mu));
  //
  Lower l4 = Lower {3, 0, 1, 3}; // move constructor
  EXPECT_TRUE(is_near(l4, ml));
  Upper u4 = Upper{3, 1, 0, 3}; // move constructor
  EXPECT_TRUE(is_near(u4, mu));
  //
  Lower l5 = TriangularMatrix<decltype(M2::Zero()), TriangleType::lower>(M2::Zero()); // compatible triangular matrix
  EXPECT_TRUE(is_near(l5, M2::Zero()));
  Upper u5 = TriangularMatrix<decltype(M2::Zero()), TriangleType::upper>(M2::Zero()); // compatible triangular matrix
  EXPECT_TRUE(is_near(u5, M2::Zero()));
  //
  Lower l7 = SelfAdjointMatrix<M2, TriangleType::lower>{9, 3, 3, 10}; // compatible self-adjoint matrix
  EXPECT_TRUE(is_near(l7, ml));
  Upper u7 = SelfAdjointMatrix<M2, TriangleType::upper>{9, 3, 3, 10}; // compatible self-adjoint matrix
  EXPECT_TRUE(is_near(u7, mu));
  //
  Lower l8 = SelfAdjointMatrix<M2, TriangleType::upper>{9, 3, 3, 10}; // compatible opposite self-adjoint matrix
  EXPECT_TRUE(is_near(l8, ml));
  Upper u8 = SelfAdjointMatrix<M2, TriangleType::lower>{9, 3, 3, 10}; // compatible opposite self-adjoint matrix
  EXPECT_TRUE(is_near(u8, mu));
  Diagonal d9 {3, 3}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d9, mat22(3, 0, 0, 3)));
  Diagonal2 d9b {3, 3}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d9b, mat22(3, 0, 0, 3)));
  Diagonal3 d9c {3, 3}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d9c, mat22(3, 0, 0, 3)));
  //
  l3 = l5; // copy assignment
  EXPECT_TRUE(is_near(l3, M2::Zero()));
  u3 = u5; // copy assignment
  EXPECT_TRUE(is_near(u3, M2::Zero()));
  //
  l5 = Lower {3., 0, 1, 3}; // move assignment
  EXPECT_TRUE(is_near(l5, ml));
  u5 = Upper {3., 1, 0, 3}; // move assignment
  EXPECT_TRUE(is_near(u5, mu));
  //
  l2 = TriangularMatrix<decltype(M2::Zero()), TriangleType::lower>(M2::Zero()); // copy assignment from compatible triangular matrix
  EXPECT_TRUE(is_near(l2, M2::Zero()));
  u2 = TriangularMatrix<decltype(M2::Zero()), TriangleType::upper>(M2::Zero()); // copy assignment from compatible triangular matrix
  EXPECT_TRUE(is_near(u2, M2::Zero()));
  //
  l2 = SelfAdjointMatrix<M2, TriangleType::lower>{9, 3, 3, 10}; // compatible self-adjoint matrix
  EXPECT_TRUE(is_near(l2, ml));
  u2 = SelfAdjointMatrix<M2, TriangleType::upper>{9, 3, 3, 10}; // compatible self-adjoint matrix
  EXPECT_TRUE(is_near(u2, mu));
  //
  l2 = SelfAdjointMatrix<M2, TriangleType::upper>{9, 3, 3, 10}; // compatible opposite self-adjoint matrix
  EXPECT_TRUE(is_near(l2, ml));
  u2 = SelfAdjointMatrix<M2, TriangleType::lower>{9, 3, 3, 10}; // compatible opposite self-adjoint matrix
  EXPECT_TRUE(is_near(u2, mu));
  //
  l3 = (M2() << 3, 0, 1, 3).finished(); // assign from regular matrix
  EXPECT_TRUE(is_near(l3, ml));
  u3 = (M2() << 3, 1, 0, 3).finished(); // assign from regular matrix
  EXPECT_TRUE(is_near(u3, mu));
  //
  auto tl = ml.triangularView<Eigen::Lower>();
  auto tu = mu.triangularView<Eigen::Upper>();
  l2 = tl; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(l2, ml));
  u2 = tl; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(u2, mu));
  //
  l3 = tu; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(l3, ml));
  u3 = tu; // copy from TriangularBase derived object
  EXPECT_TRUE(is_near(u3, mu));
  //
  l4 = ml.triangularView<Eigen::Lower>(); // assign from rvalue reference to TriangularBase derived object
  EXPECT_TRUE(is_near(l4, ml));
  u4 = ml.triangularView<Eigen::Lower>(); // assign from rvalue reference to TriangularBase derived object
  EXPECT_TRUE(is_near(u4, mu));
  //
  l5 = mu.triangularView<Eigen::Upper>(); // assign from rvalue reference to TriangularBase derived object
  EXPECT_TRUE(is_near(l5, ml));
  u5 = mu.triangularView<Eigen::Upper>(); // assign from rvalue reference to TriangularBase derived object
  EXPECT_TRUE(is_near(u5, mu));
  //
  l4 = {3, 0, 1, 3}; // assign from a list of scalars
  EXPECT_TRUE(is_near(l4, ml));
  u4 = {3, 1, 0, 3}; // assign from a list of scalars
  EXPECT_TRUE(is_near(u4, mu));
  //
  l1 += l2;
  EXPECT_TRUE(is_near(l1, mat22(6., 0, 2, 6)));
  u1 += u2;
  EXPECT_TRUE(is_near(u1, mat22(6., 2, 0, 6)));
  //
  l1 -= TriangularMatrix {3., 0, 1, 3};
  EXPECT_TRUE(is_near(l1, mat22(3., 0, 1, 3)));
  u1 -= Upper {3., 1, 0, 3};
  EXPECT_TRUE(is_near(u1, mat22(3., 1, 0, 3)));
  //
  l1 *= 3;
  EXPECT_TRUE(is_near(l1, mat22(9., 0, 3, 9)));
  u1 *= 3;
  EXPECT_TRUE(is_near(u1, mat22(9., 3, 0, 9)));
  //
  l1 /= 3;
  EXPECT_TRUE(is_near(l1, mat22(3., 0, 1, 3)));
  u1 /= 3;
  EXPECT_TRUE(is_near(u1, mat22(3., 1, 0, 3)));
  //
  l2 *= l1;
  EXPECT_TRUE(is_near(l2, mat22(9., 0, 6, 9)));
  u2 *= u1;
  EXPECT_TRUE(is_near(u2, mat22(9., 6, 0, 9)));
  //
  EXPECT_TRUE(is_near(l1, mat22(3., 0, 1, 3)));
  EXPECT_TRUE(is_near(u1, mat22(3., 1, 0, 3)));
  //
  EXPECT_TRUE(is_near(l1.solve((Eigen::Matrix<double, 2, 1>() << 3, 7).finished()), (Eigen::Matrix<double, 2, 1>() << 1, 2).finished()));
  EXPECT_TRUE(is_near(u1.solve((Eigen::Matrix<double, 2, 1>() << 3, 9).finished()), (Eigen::Matrix<double, 2, 1>() << 0, 3).finished()));
}

TEST_F(eigen3, TriangularMatrix_subscripts)
{
  static_assert(is_element_gettable_v<Lower, 2>);
  static_assert(not is_element_gettable_v<Lower, 1>);
  static_assert(is_element_gettable_v<Upper, 2>);
  static_assert(not is_element_gettable_v<Upper, 1>);
  static_assert(is_element_gettable_v<Diagonal, 2>);
  static_assert(is_element_gettable_v<Diagonal, 1>);
  static_assert(is_element_gettable_v<Diagonal2, 2>);
  static_assert(is_element_gettable_v<Diagonal2, 1>);
  static_assert(is_element_gettable_v<Diagonal3, 2>);
  static_assert(is_element_gettable_v<Diagonal3, 1>);

  static_assert(is_element_settable_v<Lower, 2>);
  static_assert(not is_element_settable_v<Lower, 1>);
  static_assert(is_element_settable_v<Upper, 2>);
  static_assert(not is_element_settable_v<Upper, 1>);
  static_assert(is_element_settable_v<Diagonal, 2>);
  static_assert(is_element_settable_v<Diagonal, 1>);
  static_assert(is_element_settable_v<Diagonal2, 2>);
  static_assert(is_element_settable_v<Diagonal2, 1>);
  static_assert(is_element_settable_v<Diagonal3, 2>);
  static_assert(is_element_settable_v<Diagonal3, 1>);

  static_assert(not is_element_settable_v<const Lower, 2>);
  static_assert(not is_element_settable_v<const Lower, 1>);
  static_assert(not is_element_settable_v<const Upper, 2>);
  static_assert(not is_element_settable_v<const Upper, 1>);
  static_assert(not is_element_settable_v<const Diagonal, 2>);
  static_assert(not is_element_settable_v<const Diagonal, 1>);
  static_assert(not is_element_settable_v<const Diagonal2, 2>);
  static_assert(not is_element_settable_v<const Diagonal2, 1>);
  static_assert(not is_element_settable_v<const Diagonal3, 2>);
  static_assert(not is_element_settable_v<const Diagonal3, 1>);

  static_assert(not is_element_settable_v<TriangularMatrix<const M2, TriangleType::lower>, 2>);
  static_assert(not is_element_settable_v<TriangularMatrix<const D2, TriangleType::lower>, 2>);
  static_assert(not is_element_settable_v<TriangularMatrix<const D2, TriangleType::lower>, 1>);
  static_assert(not is_element_settable_v<TriangularMatrix<DiagonalMatrix<const Eigen::Matrix<double, 2, 1>>, TriangleType::lower>, 2>);
  static_assert(not is_element_settable_v<TriangularMatrix<DiagonalMatrix<const Eigen::Matrix<double, 2, 1>>, TriangleType::lower>, 1>);

  auto l1 = Lower {3, 0, 1, 3};
  set_element(l1, 1.1, 1, 0);
  EXPECT_NEAR(get_element(l1, 1, 0), 1.1, 1e-8);
  bool test = false; try { set_element(l1, 2.1, 0, 1); } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_NEAR(get_element(l1, 0, 1), 0, 1e-8);

  auto u1 = Upper {3, 1, 0, 3};
  set_element(u1, 1.1, 0, 1);
  EXPECT_NEAR(get_element(u1, 0, 1), 1.1, 1e-8);
  test = false;
  try { set_element(u1, 2.1, 1, 0); } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_NEAR(get_element(u1, 1, 0), 0, 1e-8);

  EXPECT_EQ(l1(0, 1), 0);
  EXPECT_EQ(u1(1, 0), 0);
  //
  EXPECT_EQ(l1(1, 1), 3);
  EXPECT_EQ(u1(1, 1), 3);
  //
  l1(0, 0) = 5;
  EXPECT_NEAR(l1(0, 0), 5, 1e-8);
  l1(1, 0) = 6;
  EXPECT_NEAR(l1(1, 0), 6, 1e-8);
  l1(1, 1) = 8;
  test = false; try { l1(0, 1) = 7; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_TRUE(is_near(l1, mat22(5, 0, 6, 8)));

  u1(0, 0) = 5;
  EXPECT_NEAR(u1(0, 0), 5, 1e-8);
  u1(0, 1) = 6;
  EXPECT_NEAR(u1(0, 1), 6, 1e-8);
  u1(1, 1) = 8;
  test = false; try { u1(1, 0) = 7; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_TRUE(is_near(u1, mat22(5, 6, 0, 8)));
  //
  auto d9 = Diagonal {9, 10};
  d9(0, 0) = 7.1;
  EXPECT_NEAR(d9(0), 7.1, 1e-8);
  d9(1) = 8.1;
  EXPECT_NEAR(d9(1, 1), 8.1, 1e-8);
  test = false; try { d9(1, 0) = 9.1; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_TRUE(is_near(d9, mat22(7.1, 0, 0, 8.1)));

  auto d9b = Diagonal2 {9, 10};
  d9b(0, 0) = 7.1;
  EXPECT_NEAR(d9b(0, 0), 7.1, 1e-8);
  d9b(1, 1) = 8.1;
  EXPECT_NEAR(d9b(1, 1), 8.1, 1e-8);
  test = false; try { d9b(1, 0) = 9.1; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_TRUE(is_near(d9b, mat22(7.1, 0, 0, 8.1)));

  auto d9c = Diagonal3 {9, 10};
  d9c(0, 0) = 7.1;
  EXPECT_NEAR(d9c(0, 0), 7.1, 1e-8);
  d9c(1, 1) = 8.1;
  EXPECT_NEAR(d9c(1, 1), 8.1, 1e-8);
  test = false; try { d9c(1, 0) = 9.1; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_TRUE(is_near(d9c, mat22(7.1, 0, 0, 8.1)));
  //
  EXPECT_NEAR((TriangularMatrix<Eigen::Matrix<double, 1, 1>, TriangleType::lower> {7.})(0), 7., 1e-6);
  EXPECT_NEAR((TriangularMatrix<Eigen::Matrix<double, 1, 1>, TriangleType::upper> {7.})(0), 7., 1e-6);
  EXPECT_NEAR((Diagonal {2, 3})(0), 2, 1e-6);
  EXPECT_NEAR((Diagonal {2, 3})(1), 3, 1e-6);
  EXPECT_NEAR((Diagonal2 {2, 3})(0), 2, 1e-6);
  EXPECT_NEAR((Diagonal2 {2, 3})(1), 3, 1e-6);
  EXPECT_NEAR((Diagonal3 {2, 3})(0), 2, 1e-6);
  EXPECT_NEAR((Diagonal3 {2, 3})(1), 3, 1e-6);
  EXPECT_NEAR((Diagonal {2, 3})[0], 2, 1e-6);
  EXPECT_NEAR((Diagonal {2, 3})[1], 3, 1e-6);
  EXPECT_NEAR((Diagonal2 {2, 3})[0], 2, 1e-6);
  EXPECT_NEAR((Diagonal2 {2, 3})[1], 3, 1e-6);
  EXPECT_NEAR((Diagonal3 {2, 3})[0], 2, 1e-6);
  EXPECT_NEAR((Diagonal3 {2, 3})[1], 3, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((Upper {3., 1, 0, 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((Diagonal {3., 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((Diagonal2 {3., 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((Diagonal3 {3., 3})(0, 0), 3, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 3})(0, 1), 0, 1e-6);
  EXPECT_NEAR((Upper {3., 1, 0, 3})(0, 1), 1, 1e-6);
  EXPECT_NEAR((Diagonal {3., 3})(0, 1), 0, 1e-6);
  EXPECT_NEAR((Diagonal2 {3., 3})(0, 1), 0, 1e-6);
  EXPECT_NEAR((Diagonal3 {3., 3})(0, 1), 0, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 3})(1, 0), 1, 1e-6);
  EXPECT_NEAR((Upper {3., 1, 0, 3})(1, 0), 0, 1e-6);
  EXPECT_NEAR((Diagonal {3., 3})(1, 0), 0, 1e-6);
  EXPECT_NEAR((Diagonal2 {3., 3})(1, 0), 0, 1e-6);
  EXPECT_NEAR((Diagonal3 {3., 3})(1, 0), 0, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((Upper {3., 1, 0, 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((Diagonal {3., 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((Diagonal2 {3., 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((Diagonal3 {3., 3})(1, 1), 3, 1e-6);
}

TEST_F(eigen3, TriangularMatrix_make)
{
  static_assert(zero_matrix<decltype(make_EigenTriangularMatrix<TriangleType::upper>(MatrixTraits<M2>::zero()))>);
  static_assert(zero_matrix<decltype(make_EigenTriangularMatrix<TriangleType::lower>(MatrixTraits<M2>::zero()))>);
  static_assert(zero_matrix<decltype(make_EigenTriangularMatrix(MatrixTraits<M2>::zero()))>);
  static_assert(upper_triangular_matrix<decltype(make_EigenTriangularMatrix<TriangleType::upper>((M2() << 3, 1, 0, 3).finished()))>);
  static_assert(lower_triangular_matrix<decltype(make_EigenTriangularMatrix<TriangleType::lower>((M2() << 3, 0, 1, 3).finished()))>);
  static_assert(lower_triangular_matrix<decltype(make_EigenTriangularMatrix((M2() << 3, 0, 1, 3).finished()))>);
  static_assert(diagonal_matrix<decltype(make_EigenTriangularMatrix<TriangleType::upper>(DiagonalMatrix {3., 4}))>);
  static_assert(diagonal_matrix<decltype(make_EigenTriangularMatrix<TriangleType::lower>(DiagonalMatrix {3., 4}))>);
  static_assert(diagonal_matrix<decltype(make_EigenTriangularMatrix(DiagonalMatrix {3., 4}))>);

  static_assert(upper_triangular_matrix<decltype(make_EigenTriangularMatrix<TriangleType::upper>(Upper {3, 1, 0, 3}))>);
  static_assert(lower_triangular_matrix<decltype(make_EigenTriangularMatrix<TriangleType::lower>(Lower {3, 0, 1, 3}))>);
  static_assert(upper_triangular_matrix<decltype(make_EigenTriangularMatrix(Upper {3, 1, 0, 3}))>);
  static_assert(lower_triangular_matrix<decltype(make_EigenTriangularMatrix(Lower {3, 0, 1, 3}))>);
}

TEST_F(eigen3, TriangularMatrix_traits)
{
  M2 ml, mu;
  ml << 3, 0, 1, 3;
  mu << 3, 1, 0, 3;
  using Dl = TriangularMatrix<M2, TriangleType::lower>;
  using Du = TriangularMatrix<M2, TriangleType::upper>;
  EXPECT_TRUE(is_near(MatrixTraits<Dl>::make(ml), ml));
  EXPECT_TRUE(is_near(MatrixTraits<Du>::make(mu), mu));
  //
  EXPECT_TRUE(is_near(MatrixTraits<Dl>::make(3, 0, 1, 3), ml));
  EXPECT_TRUE(is_near(MatrixTraits<Du>::make(3, 1, 0, 3), mu));
  //
  EXPECT_TRUE(is_near(MatrixTraits<Dl>::zero(), M2::Zero()));
  EXPECT_TRUE(is_near(MatrixTraits<Du>::zero(), M2::Zero()));
  //
  EXPECT_TRUE(is_near(MatrixTraits<Dl>::identity(), M2::Identity()));
  EXPECT_TRUE(is_near(MatrixTraits<Du>::identity(), M2::Identity()));

  static_assert(lower_triangular_matrix<Lower>);
  static_assert(upper_triangular_matrix<Upper>);
  static_assert(diagonal_matrix<Diagonal>);
  static_assert(diagonal_matrix<Diagonal2>);
  static_assert(diagonal_matrix<Diagonal3>);
  static_assert(zero_matrix<decltype(TriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero()))>);
  static_assert(zero_matrix<decltype(TriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero()))>);
  static_assert(identity_matrix<decltype(TriangularMatrix<decltype(MatrixTraits<M2>::identity()), TriangleType::lower>(MatrixTraits<M2>::identity()))>);
}

TEST_F(eigen3, TriangularMatrix_overloads)
{
  M2 ml, mu;
  ml << 3, 0, 1, 3;
  mu << 3, 1, 0, 3;
  //
  EXPECT_TRUE(is_near(make_native_matrix(Lower(3., 0, 1, 3)), ml));
  EXPECT_TRUE(is_near(make_native_matrix(Upper(3., 1, 0, 3)), mu));
  //
  EXPECT_TRUE(is_near(make_self_contained(Lower(M2::Zero())), (M2() << 0, 0, 0, 0).finished()));
  EXPECT_TRUE(is_near(make_self_contained(Upper(M2::Zero())), (M2() << 0, 0, 0, 0).finished()));
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(Lower {9, 3, 3, 10} * 2))>, Lower>);
  static_assert(std::is_same_v<std::decay_t<decltype(make_self_contained(Upper {9, 3, 3, 10} * 2))>, Upper>);
  //
  //
  EXPECT_TRUE(is_near(Cholesky_square(TriangularMatrix<Eigen::Matrix<double, 1, 1>, TriangleType::lower>(Eigen::Matrix<double, 1, 1>(4))), Eigen::Matrix<double, 1, 1>(16)));
  static_assert(one_by_one_matrix<decltype(Cholesky_square(TriangularMatrix<Eigen::Matrix<double, 1, 1>, TriangleType::lower>(Eigen::Matrix<double, 1, 1>(4))))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(TriangularMatrix<decltype(M2::Identity()), TriangleType::lower>(M2::Identity())), M2::Identity()));
  EXPECT_TRUE(is_near(Cholesky_square(TriangularMatrix<decltype(M2::Identity()), TriangleType::upper>(M2::Identity())), M2::Identity()));
  static_assert(identity_matrix<decltype(Cholesky_square(TriangularMatrix<decltype(M2::Identity()), TriangleType::lower>(M2::Identity())))>);
  static_assert(identity_matrix<decltype(Cholesky_square(TriangularMatrix<decltype(M2::Identity()), TriangleType::upper>(M2::Identity())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(TriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero())), M2::Zero()));
  EXPECT_TRUE(is_near(Cholesky_square(TriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero())), M2::Zero()));
  static_assert(zero_matrix<decltype(Cholesky_square(TriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero())))>);
  static_assert(zero_matrix<decltype(Cholesky_square(TriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(TriangularMatrix<decltype(DiagonalMatrix{2., 3}), TriangleType::lower>(DiagonalMatrix{2., 3})), DiagonalMatrix{4., 9}));
  EXPECT_TRUE(is_near(Cholesky_square(TriangularMatrix<decltype(DiagonalMatrix{2., 3}), TriangleType::upper>(DiagonalMatrix{2., 3})), DiagonalMatrix{4., 9}));
  static_assert(eigen_diagonal_expr<decltype(Cholesky_square(TriangularMatrix<decltype(DiagonalMatrix{2., 3}), TriangleType::lower>(DiagonalMatrix{2., 3})))>);
  static_assert(eigen_diagonal_expr<decltype(Cholesky_square(TriangularMatrix<decltype(DiagonalMatrix{2., 3}), TriangleType::upper>(DiagonalMatrix{2., 3})))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(TriangularMatrix<M2, TriangleType::diagonal>(ml)), DiagonalMatrix{9., 9}));
  static_assert(eigen_diagonal_expr<decltype(Cholesky_square(TriangularMatrix<M2, TriangleType::diagonal>(ml)))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(Lower {3., 0, 1, 3}), mat22(9., 3, 3, 10)));
  EXPECT_TRUE(is_near(Cholesky_square(Upper {3., 1, 0, 3}), mat22(9., 3, 3, 10)));
  static_assert(is_lower_storage_triangle_v<decltype(Cholesky_square(Lower {3, 0, 1, 3}))>);
  static_assert(is_upper_storage_triangle_v<decltype(Cholesky_square(Upper {3, 1, 0, 3}))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(TriangularMatrix<Eigen::Matrix<double, 1, 1>, TriangleType::lower>(Eigen::Matrix<double, 1, 1>(9))), Eigen::Matrix<double, 1, 1>(81)));
  EXPECT_TRUE(is_near(Cholesky_square(TriangularMatrix<Eigen::Matrix<double, 1, 1>, TriangleType::upper>(Eigen::Matrix<double, 1, 1>(9))), Eigen::Matrix<double, 1, 1>(81)));
  //
  //
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularMatrix<Eigen::Matrix<double, 1, 1>, TriangleType::lower>(Eigen::Matrix<double, 1, 1>(4))), Eigen::Matrix<double, 1, 1>(2)));
  static_assert(one_by_one_matrix<decltype(Cholesky_factor(TriangularMatrix<Eigen::Matrix<double, 1, 1>, TriangleType::lower>(Eigen::Matrix<double, 1, 1>(4))))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularMatrix<decltype(M2::Identity()), TriangleType::lower>(M2::Identity())), M2::Identity()));
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularMatrix<decltype(M2::Identity()), TriangleType::upper>(M2::Identity())), M2::Identity()));
  static_assert(identity_matrix<decltype(Cholesky_factor(TriangularMatrix<decltype(M2::Identity()), TriangleType::lower>(M2::Identity())))>);
  static_assert(identity_matrix<decltype(Cholesky_factor(TriangularMatrix<decltype(M2::Identity()), TriangleType::upper>(M2::Identity())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero())), M2::Zero()));
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero())), M2::Zero()));
  static_assert(zero_matrix<decltype(Cholesky_factor(TriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero())))>);
  static_assert(zero_matrix<decltype(Cholesky_factor(TriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularMatrix<decltype(DiagonalMatrix{4., 9}), TriangleType::lower>(DiagonalMatrix{4., 9})), DiagonalMatrix{2., 3}));
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularMatrix<decltype(DiagonalMatrix{4., 9}), TriangleType::upper>(DiagonalMatrix{4., 9})), DiagonalMatrix{2., 3}));
  static_assert(eigen_diagonal_expr<decltype(Cholesky_factor(TriangularMatrix<decltype(DiagonalMatrix{4., 9}), TriangleType::lower>(DiagonalMatrix{4., 9})))>);
  static_assert(eigen_diagonal_expr<decltype(Cholesky_factor(TriangularMatrix<decltype(DiagonalMatrix{4., 9}), TriangleType::upper>(DiagonalMatrix{4., 9})))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularMatrix<M2, TriangleType::diagonal>(ml)), DiagonalMatrix{std::sqrt(3.), std::sqrt(3.)}));
  static_assert(eigen_diagonal_expr<decltype(Cholesky_factor(TriangularMatrix<M2, TriangleType::diagonal>(ml)))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularMatrix<Eigen::Matrix<double, 1, 1>, TriangleType::lower>(Eigen::Matrix<double, 1, 1>(9))), Eigen::Matrix<double, 1, 1>(3)));
  EXPECT_TRUE(is_near(Cholesky_factor(TriangularMatrix<Eigen::Matrix<double, 1, 1>, TriangleType::upper>(Eigen::Matrix<double, 1, 1>(9))), Eigen::Matrix<double, 1, 1>(3)));
  //
  //
  EXPECT_TRUE(is_near(transpose(Lower {3., 0, 1, 3}), mu));
  EXPECT_TRUE(is_near(transpose(Upper {3., 1, 0, 3}), ml));
  //
  EXPECT_TRUE(is_near(adjoint(Lower {3., 0, 1, 3}), mu));
  EXPECT_TRUE(is_near(adjoint(Upper {3., 1, 0, 3}), ml));
  //
  EXPECT_NEAR(determinant(Lower {3., 0, 1, 3}), 9, 1e-6);
  EXPECT_NEAR(determinant(Upper {3., 1, 0, 3}), 9, 1e-6);
  //
  EXPECT_NEAR(trace(Lower {3., 0, 1, 3}), 6, 1e-6);
  EXPECT_NEAR(trace(Upper {3., 1, 0, 3}), 6, 1e-6);
  //
  EXPECT_TRUE(is_near(solve(Lower {3., 0, 1, 3}, (Eigen::Matrix<double, 2, 1>() << 3, 7).finished()), make_native_matrix(1., 2)));
  EXPECT_TRUE(is_near(solve(Upper {3., 1, 0, 3}, (Eigen::Matrix<double, 2, 1>() << 3, 9).finished()), make_native_matrix(0., 3)));
  //
  EXPECT_TRUE(is_near(reduce_columns(Lower {3., 0, 1, 3}), make_native_matrix(1.5, 2)));
  EXPECT_TRUE(is_near(reduce_columns(Upper {3., 1, 0, 3}), make_native_matrix(2, 1.5)));
  //
  auto sl1 = Lower {3., 0, 1, 3};
  rank_update(sl1, (M2() << 2, 0, 1, 2).finished(), 4);
  EXPECT_TRUE(is_near(sl1, mat22(5., 0, 2.2, std::sqrt(25.16))));
  auto su1 = Upper {3., 1, 0, 3};
  rank_update(su1, (M2() << 2, 0, 1, 2).finished(), 4);
  EXPECT_TRUE(is_near(su1, mat22(5., 2.2, 0, std::sqrt(25.16))));
  //
  const auto sl2 = Lower {3., 0, 1, 3};
  EXPECT_TRUE(is_near(rank_update(sl2, (M2() << 2, 0, 1, 2).finished(), 4), mat22(5., 0, 2.2, std::sqrt(25.16))));
  const auto su2 = Upper {3., 1, 0, 3};
  EXPECT_TRUE(is_near(rank_update(su2, (M2() << 2, 0, 1, 2).finished(), 4), mat22(5., 2.2, 0, std::sqrt(25.16))));
  //
  EXPECT_TRUE(is_near(rank_update(Lower {3., 0, 1, 3}, (M2() << 2, 0, 1, 2).finished(), 4), mat22(5., 0, 2.2, std::sqrt(25.16))));
  EXPECT_TRUE(is_near(rank_update(Upper {3., 1, 0, 3}, (M2() << 2, 0, 1, 2).finished(), 4), mat22(5., 2.2, 0, std::sqrt(25.16))));
  //
  EXPECT_TRUE(is_near(solve(Lower {3., 0, 1, 3}, (Eigen::Matrix<double, 2, 1>() << 3, 7).finished()), (Eigen::Matrix<double, 2, 1>() << 1, 2).finished()));
  EXPECT_TRUE(is_near(solve(Upper {3., 1, 0, 3}, (Eigen::Matrix<double, 2, 1>() << 3, 9).finished()), (Eigen::Matrix<double, 2, 1>() << 0, 3).finished()));
  //
  EXPECT_TRUE(is_near(LQ_decomposition(Lower {3., 0, 1, 3}), mat22(3., 0, 1, 3)));
  EXPECT_TRUE(is_near(QR_decomposition(Upper {3., 1, 0, 3}), mat22(3., 1, 0, 3)));
  EXPECT_TRUE(is_near(Cholesky_square(LQ_decomposition(Upper {3., 1, 0, 3})), mat22(10, 3, 3, 9)));
  EXPECT_TRUE(is_near(Cholesky_square(QR_decomposition(Lower {3., 0, 1, 3})), mat22(10, 3, 3, 9)));
}

TEST_F(eigen3, TriangularMatrix_blocks_lower)
{
  auto m0 = TriangularMatrix<Eigen::Matrix<double, 3, 3>, TriangleType::lower> {1, 0, 0,
                                                                                     2, 4, 0,
                                                                                     3, 5, 6};
  auto m1 = TriangularMatrix<Eigen::Matrix<double, 3, 3>, TriangleType::lower> {4, 0, 0,
                                                                                     5, 7, 0,
                                                                                     6, 8, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 0, 2, 3}, m1),
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1., 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}));
  static_assert(lower_triangular_matrix<decltype(concatenate_diagonal(TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 0, 2, 3}, m1))>);

  EXPECT_TRUE(is_near(concatenate_vertical(m0, m1),
    make_native_matrix<6,3>(1., 0, 0,
                                    2, 4, 0,
                                    3, 5, 6,
                                    4, 0, 0,
                                    5, 7, 0,
                                    6, 8, 9)));
  EXPECT_TRUE(is_near(concatenate_horizontal(m0, m1),
    make_native_matrix<3, 6>(1., 0, 0, 4, 0, 0,
                                     2, 4, 0, 5, 7, 0,
                                     3, 5, 6, 6, 8, 9)));
  EXPECT_TRUE(is_near(split_diagonal(TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1., 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple{TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 0, 2, 3}, m1}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1., 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple{TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 0, 2, 3},
               TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {4., 0, 5, 7}}));
  EXPECT_TRUE(is_near(split_vertical(TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_vertical<2, 3>(
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1, 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple{make_native_matrix<2,5>(1., 0, 0, 0, 0,
                                               2, 3, 0, 0, 0),
               make_native_matrix<3,5>(0., 0, 4, 0, 0,
                                               0, 0, 5, 7, 0,
                                               0, 0, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1, 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple{make_native_matrix<2,5>(1., 0, 0, 0, 0,
                                               2, 3, 0, 0, 0),
               make_native_matrix<2,5>(0., 0, 4, 0, 0,
                                               0, 0, 5, 7, 0)}));
  EXPECT_TRUE(is_near(split_horizontal(TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_horizontal<2, 3>(
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1, 0, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 0, 0,
                                                                              0, 0, 5, 7, 0,
                                                                              0, 0, 6, 8, 9}),
    std::tuple{make_native_matrix<5,2>(1., 0, 2, 3, 0, 0, 0, 0, 0, 0),
               make_native_matrix<5,3>(0., 0, 0, 0, 0, 0, 4, 0, 0, 5, 7, 0, 6, 8, 9)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1, 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple{make_native_matrix<5,2>(1., 0, 2, 3, 0, 0, 0, 0, 0, 0),
               make_native_matrix<5,2>(0., 0, 0, 0, 4, 0, 5, 7, 6, 8)}));

  EXPECT_TRUE(is_near(column(m1, 2), make_native_matrix(0., 0, 9)));
  EXPECT_TRUE(is_near(column<1>(m1), make_native_matrix(0., 7, 8)));
  EXPECT_TRUE(is_near(apply_columnwise(m1, [](const auto& col){ return col + col.Constant(1); }),
    (Eigen::Matrix<double, 3, 3>() <<
      5, 1, 1,
      6, 8, 1,
      7, 9, 10).finished()));
  EXPECT_TRUE(is_near(apply_columnwise(m1, [](const auto& col, std::size_t i){ return col + col.Constant(i); }),
    (Eigen::Matrix<double, 3, 3>() <<
      4, 1, 2,
      5, 8, 2,
      6, 9, 11).finished()));
  EXPECT_TRUE(is_near(apply_coefficientwise(m1, [](const auto& x){ return x + 1; }),
    (Eigen::Matrix<double, 3, 3>() <<
      5, 1, 1,
      6, 8, 1,
      7, 9, 10).finished()));
  EXPECT_TRUE(is_near(apply_coefficientwise(m1, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }),
    (Eigen::Matrix<double, 3, 3>() <<
      4, 1, 2,
      6, 9, 3,
      8, 11, 13).finished()));
}

TEST_F(eigen3, TriangularMatrix_blocks_upper)
{
  auto m0 = TriangularMatrix<Eigen::Matrix<double, 3, 3>, TriangleType::upper> {1, 2, 3,
                                                                                     0, 4, 5,
                                                                                     0, 0, 6};
  auto m1 = TriangularMatrix<Eigen::Matrix<double, 3, 3>, TriangleType::upper> {4., 5, 6,
                                                                                     0, 7, 8,
                                                                                     0, 0, 9};
  EXPECT_TRUE(is_near(concatenate_diagonal(TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 0, 3}, m1),
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}));
  static_assert(upper_triangular_matrix<decltype(concatenate_diagonal(TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 0, 3}, m1))>);

  EXPECT_TRUE(is_near(concatenate_vertical(m0, m1),
    make_native_matrix<6,3>(1., 2, 3,
                                    0, 4, 5,
                                    0, 0, 6,
                                    4, 5, 6,
                                    0, 7, 8,
                                    0, 0, 9)));
  EXPECT_TRUE(is_near(concatenate_horizontal(m0, m1),
    make_native_matrix<3,6>(1., 2, 3, 4, 5, 6,
                                    0, 4, 5, 0, 7, 8,
                                    0, 0, 6, 0, 0, 9)));
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple{TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 0, 3}, m1}));
  EXPECT_TRUE(is_near(split_diagonal(TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple{TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 0, 3},
               TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {4., 5, 0, 7}}));
  EXPECT_TRUE(is_near(split_vertical(TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_vertical<2, 3>(
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple{make_native_matrix<2,5>(1., 2, 0, 0, 0,
                                               0, 3, 0, 0, 0),
               make_native_matrix<3,5>(0., 0, 4, 5, 6,
                                               0, 0, 0, 7, 8,
                                               0, 0, 0, 0, 9)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple{make_native_matrix<2,5>(1., 2, 0, 0, 0,
                                               0, 3, 0, 0, 0),
               make_native_matrix<2,5>(0., 0, 4, 5, 6,
                                               0, 0, 0, 7, 8)}));
  EXPECT_TRUE(is_near(split_horizontal(TriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_horizontal<2, 3>(
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple{make_native_matrix<5,2>(1., 2, 0, 3, 0, 0, 0, 0, 0, 0),
               make_native_matrix<5,3>(0., 0, 0, 0, 0, 0, 4, 5, 6, 0, 7, 8, 0, 0, 9)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(
    TriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple{make_native_matrix<5,2>(1., 2, 0, 3, 0, 0, 0, 0, 0, 0),
               make_native_matrix<5,2>(0., 0, 0, 0, 4, 5, 0, 7, 0, 0)}));
  EXPECT_TRUE(is_near(column(m1, 2), make_native_matrix(6., 8, 9)));
  EXPECT_TRUE(is_near(column<1>(m1), make_native_matrix(5., 7, 0)));
  EXPECT_TRUE(is_near(apply_columnwise(m1, [](const auto& col){ return col + col.Constant(1); }),
    (Eigen::Matrix<double, 3, 3>() <<
      5, 6, 7,
      1, 8, 9,
      1, 1, 10).finished()));
  EXPECT_TRUE(is_near(apply_columnwise(m1, [](const auto& col, std::size_t i){ return col + col.Constant(i); }),
    (Eigen::Matrix<double, 3, 3>() <<
      4, 6, 8,
      0, 8, 10,
      0, 1, 11).finished()));
  EXPECT_TRUE(is_near(apply_coefficientwise(m1, [](const auto& x){ return x + 1; }),
    (Eigen::Matrix<double, 3, 3>() <<
      5, 6, 7,
      1, 8, 9,
      1, 1, 10).finished()));
  EXPECT_TRUE(is_near(apply_coefficientwise(m1, [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }),
    (Eigen::Matrix<double, 3, 3>() <<
      4, 6, 8,
      1, 9, 11,
      2, 3, 13).finished()));
}

TEST_F(eigen3, TriangularMatrix_arithmetic_lower)
{
  auto m1 = Lower {4., 0, 5, 6};
  auto m2 = Lower {1., 0, 2, 3};
  auto d = DiagonalMatrix<Eigen::Matrix<double, 2, 1>> {1, 3};
  auto i = M2::Identity();
  auto z = ZeroMatrix<M2> {};
  EXPECT_TRUE(is_near(m1 + m2, mat22(5, 0, 7, 9))); static_assert(lower_triangular_matrix<decltype(m1 + m2)>);
  EXPECT_TRUE(is_near(m1 + d, mat22(5, 0, 5, 9))); static_assert(lower_triangular_matrix<decltype(m1 + d)>);
  EXPECT_TRUE(is_near(d + m1, mat22(5, 0, 5, 9))); static_assert(lower_triangular_matrix<decltype(d + m1)>);
  EXPECT_TRUE(is_near(m1 + i, mat22(5, 0, 5, 7))); static_assert(lower_triangular_matrix<decltype(m1 + i)>);
  EXPECT_TRUE(is_near(i + m1, mat22(5, 0, 5, 7))); static_assert(lower_triangular_matrix<decltype(i + m1)>);
  EXPECT_TRUE(is_near(m1 + z, mat22(4, 0, 5, 6))); static_assert(lower_triangular_matrix<decltype(m1 + z)>);
  EXPECT_TRUE(is_near(z + m1, mat22(4, 0, 5, 6))); static_assert(lower_triangular_matrix<decltype(z + m1)>);

  EXPECT_TRUE(is_near(m1 - m2, mat22(3, 0, 3, 3))); static_assert(lower_triangular_matrix<decltype(m1 - m2)>);
  EXPECT_TRUE(is_near(m1 - d, mat22(3, 0, 5, 3))); static_assert(lower_triangular_matrix<decltype(m1 - d)>);
  EXPECT_TRUE(is_near(d - m1, mat22(-3, 0, -5, -3))); static_assert(lower_triangular_matrix<decltype(d - m1)>);
  EXPECT_TRUE(is_near(m1 - i, mat22(3, 0, 5, 5))); static_assert(lower_triangular_matrix<decltype(m1 - i)>);
  EXPECT_TRUE(is_near(i - m1, mat22(-3, 0, -5, -5))); static_assert(lower_triangular_matrix<decltype(i - m1)>);
  EXPECT_TRUE(is_near(m1 - z, mat22(4, 0, 5, 6))); static_assert(lower_triangular_matrix<decltype(m1 - z)>);
  EXPECT_TRUE(is_near(z - m1, mat22(-4, 0, -5, -6))); static_assert(lower_triangular_matrix<decltype(z - m1)>);

  EXPECT_TRUE(is_near(m1 * 2, mat22(8, 0, 10, 12))); static_assert(lower_triangular_matrix<decltype(m1 * 2)>);
  EXPECT_TRUE(is_near(2 * m1, mat22(8, 0, 10, 12))); static_assert(lower_triangular_matrix<decltype(2 * m1)>);
  EXPECT_TRUE(is_near(m1 / 2, mat22(2, 0, 2.5, 3))); static_assert(lower_triangular_matrix<decltype(m1 / 2)>);
  static_assert(lower_triangular_matrix<decltype(m1 / 0)>);
  EXPECT_TRUE(is_near(-m1, mat22(-4, 0, -5, -6)));  static_assert(lower_triangular_matrix<decltype(-m1)>);

  EXPECT_TRUE(is_near(m1 * m2, mat22(4, 0, 17, 18))); static_assert(lower_triangular_matrix<decltype(m1 * m2)>);
  EXPECT_TRUE(is_near(m1 * d, mat22(4, 0, 5, 18))); static_assert(lower_triangular_matrix<decltype(m1 * d)>);
  EXPECT_TRUE(is_near(d * m1, mat22(4, 0, 15, 18))); static_assert(lower_triangular_matrix<decltype(d * m2)>);
  EXPECT_TRUE(is_near(m1 * i, m1));  static_assert(lower_triangular_matrix<decltype(m1 * i)>);
  EXPECT_TRUE(is_near(i * m1, m1));  static_assert(lower_triangular_matrix<decltype(i * m1)>);
  EXPECT_TRUE(is_near(m1 * z, z));  static_assert(zero_matrix<decltype(m1 * z)>);
  EXPECT_TRUE(is_near(z * m1, z));  static_assert(zero_matrix<decltype(z * m1)>);

  EXPECT_TRUE(is_near(mat22(4, 0, 5, 6) * m2, mat22(4, 0, 17, 18)));
  EXPECT_TRUE(is_near(m1 * mat22(1, 0, 2, 3), mat22(4, 0, 17, 18)));
}

TEST_F(eigen3, TriangularMatrix_arithmetic_upper)
{
  auto m1 = Upper {4., 5, 0, 6};
  auto m2 = Upper {1., 2, 0, 3};
  auto d = DiagonalMatrix<Eigen::Matrix<double, 2, 1>> {1, 3};
  auto i = M2::Identity();
  auto z = ZeroMatrix<M2> {};
  EXPECT_TRUE(is_near(m1 + m2, mat22(5, 7, 0, 9))); static_assert(upper_triangular_matrix<decltype(m1 + m2)>);
  EXPECT_TRUE(is_near(m1 + d, mat22(5, 5, 0, 9))); static_assert(upper_triangular_matrix<decltype(m1 + d)>);
  EXPECT_TRUE(is_near(d + m1, mat22(5, 5, 0, 9))); static_assert(upper_triangular_matrix<decltype(d + m1)>);
  EXPECT_TRUE(is_near(m1 + i, mat22(5, 5, 0, 7))); static_assert(upper_triangular_matrix<decltype(m1 + i)>);
  EXPECT_TRUE(is_near(i + m1, mat22(5, 5, 0, 7))); static_assert(upper_triangular_matrix<decltype(i + m1)>);
  EXPECT_TRUE(is_near(m1 + z, mat22(4, 5, 0, 6))); static_assert(upper_triangular_matrix<decltype(m1 + z)>);
  EXPECT_TRUE(is_near(z + m1, mat22(4, 5, 0, 6))); static_assert(upper_triangular_matrix<decltype(z + m1)>);

  EXPECT_TRUE(is_near(m1 - m2, mat22(3, 3, 0, 3))); static_assert(upper_triangular_matrix<decltype(m1 - m2)>);
  EXPECT_TRUE(is_near(m1 - d, mat22(3, 5, 0, 3))); static_assert(upper_triangular_matrix<decltype(m1 - d)>);
  EXPECT_TRUE(is_near(d - m1, mat22(-3, -5, 0, -3))); static_assert(upper_triangular_matrix<decltype(d - m1)>);
  EXPECT_TRUE(is_near(m1 - i, mat22(3, 5, 0, 5))); static_assert(upper_triangular_matrix<decltype(m1 - i)>);
  EXPECT_TRUE(is_near(i - m1, mat22(-3, -5, 0, -5))); static_assert(upper_triangular_matrix<decltype(i - m1)>);
  EXPECT_TRUE(is_near(m1 - z, mat22(4, 5, 0, 6))); static_assert(upper_triangular_matrix<decltype(m1 - z)>);
  EXPECT_TRUE(is_near(z - m1, mat22(-4, -5, 0, -6))); static_assert(upper_triangular_matrix<decltype(z - m1)>);

  EXPECT_TRUE(is_near(m1 * 2, mat22(8, 10, 0, 12))); static_assert(upper_triangular_matrix<decltype(m1 * 2)>);
  EXPECT_TRUE(is_near(2 * m1, mat22(8, 10, 0, 12))); static_assert(upper_triangular_matrix<decltype(2 * m1)>);
  EXPECT_TRUE(is_near(m1 / 2, mat22(2, 2.5, 0, 3))); static_assert(upper_triangular_matrix<decltype(m1 / 2)>);
  static_assert(upper_triangular_matrix<decltype(m1 / 0)>);
  EXPECT_TRUE(is_near(-m1, mat22(-4, -5, 0, -6)));  static_assert(upper_triangular_matrix<decltype(-m1)>);

  EXPECT_TRUE(is_near(m1 * m2, mat22(4, 23, 0, 18))); static_assert(upper_triangular_matrix<decltype(m1 * m2)>);
  EXPECT_TRUE(is_near(m1 * d, mat22(4, 15, 0, 18))); static_assert(upper_triangular_matrix<decltype(m1 * d)>);
  EXPECT_TRUE(is_near(d * m1, mat22(4, 5, 0, 18))); static_assert(upper_triangular_matrix<decltype(d * m1)>);
  EXPECT_TRUE(is_near(m1 * i, m1));  static_assert(upper_triangular_matrix<decltype(m1 * i)>);
  EXPECT_TRUE(is_near(i * m1, m1));  static_assert(upper_triangular_matrix<decltype(i * m1)>);
  EXPECT_TRUE(is_near(m1 * z, z));  static_assert(zero_matrix<decltype(m1 * z)>);
  EXPECT_TRUE(is_near(z * m1, z));  static_assert(zero_matrix<decltype(z * m1)>);

  EXPECT_TRUE(is_near(mat22(4, 5, 0, 6) * m2, mat22(4, 23, 0, 18)));
  EXPECT_TRUE(is_near(m1 * mat22(1, 2, 0, 3), mat22(4, 23, 0, 18)));
}

TEST_F(eigen3, TriangularMatrix_arithmetic_mixed)
{
  auto m_upper = Upper {4., 5, 0, 6};
  auto m_lower = Lower {1., 0, 2, 3};
  EXPECT_TRUE(is_near(m_upper + m_lower, mat22(5, 5, 2, 9)));
  EXPECT_TRUE(is_near(m_lower + m_upper, mat22(5, 5, 2, 9)));
  EXPECT_TRUE(is_near(m_upper - m_lower, mat22(3, 5, -2, 3)));
  EXPECT_TRUE(is_near(m_lower - m_upper, mat22(-3, -5, 2, -3)));
  EXPECT_TRUE(is_near(m_upper * m_lower, mat22(14, 15, 12, 18)));
  EXPECT_TRUE(is_near(m_lower * m_upper, mat22(4, 5, 8, 28)));
}

TEST_F(eigen3, TriangularMatrix_references)
{
  M2 m, n;
  m << 2, 0, 1, 2;
  n << 3, 0, 1, 3;
  TriangularMatrix<M2, TriangleType::lower> x = m;
  TriangularMatrix<M2&, TriangleType::lower> x_lvalue = x;
  EXPECT_TRUE(is_near(x_lvalue, m));
  x = n;
  EXPECT_TRUE(is_near(x_lvalue, n));
  x_lvalue = m;
  EXPECT_TRUE(is_near(x, m));
  TriangularMatrix<M2&&, TriangleType::lower> x_rvalue = std::move(x);
  EXPECT_TRUE(is_near(x_rvalue, m));
  x_rvalue = n;
  EXPECT_TRUE(is_near(x_rvalue, n));
  //
  using V = TriangularMatrix<Eigen::Matrix<double, 3, 3>>;
  V v1 {1., 0, 0,
        2, 4, 0,
        3, -6, -3};
  bool test = false; try { v1(0, 1) = 3.2; } catch (const std::out_of_range& e) { test = true; }
  EXPECT_TRUE(test);
  EXPECT_EQ(v1(1,0), 2);
  TriangularMatrix<Eigen::Matrix<double, 3, 3>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  v2(2, 0) = 5.2;
  EXPECT_EQ(v1(2,0), 5.2);
  TriangularMatrix<Eigen::Matrix<double, 3, 3>&&> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 4.1);
  TriangularMatrix<const Eigen::Matrix<double, 3, 3>&> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  TriangularMatrix<Eigen::Matrix<double, 3, 3>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4);
}
