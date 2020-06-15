/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "matrix_tests.h"

using namespace OpenKalman;

using M2 = Eigen::Matrix<double, 2, 2>;
using Lower = EigenTriangularMatrix<M2, TriangleType::lower>;
using Upper = EigenTriangularMatrix<M2, TriangleType::upper>;
using Diagonal = EigenTriangularMatrix<M2, TriangleType::diagonal>;
using Diagonal2 = EigenTriangularMatrix<EigenDiagonal<Eigen::Matrix<double, 2, 1>>, TriangleType::diagonal>;
using Diagonal3 = EigenTriangularMatrix<EigenDiagonal<Eigen::Matrix<double, 2, 1>>, TriangleType::lower>;
using Mat = TypedMatrix<Axes<2>, Axes<2>, M2>;

TEST_F(matrix_tests, TriangularMatrix_class)
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
  d1 << 3, 7, 8, 3;
  EXPECT_TRUE(is_near(d1.base_matrix(), Mat {3, 7, 8, 3}));
  EXPECT_TRUE(is_near(d1, Mat {3, 0, 0, 3}));
  d1.template triangularView<Eigen::Lower>() = (M2() << 2, 5, 6, 2).finished();
  EXPECT_TRUE(is_near(d1, Mat {2, 0, 0, 2}));
  Diagonal2 d1b;
  d1b << 3, 3;
  EXPECT_TRUE(is_near(d1b.base_matrix(), Mat {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(d1b, Mat {3, 0, 0, 3}));
  d1b.template triangularView<Eigen::Lower>() = (M2() << 2, 5, 6, 2).finished();
  EXPECT_TRUE(is_near(d1b, Mat {2, 0, 0, 2}));
  Diagonal3 d1c;
  d1c << 3, 3;
  EXPECT_TRUE(is_near(d1c.base_matrix(), Mat {3, 0, 0, 3}));
  EXPECT_TRUE(is_near(d1c, Mat {3, 0, 0, 3}));
  d1c.template triangularView<Eigen::Lower>() = (M2() << 2, 5, 6, 2).finished();
  EXPECT_TRUE(is_near(d1c, Mat {2, 0, 0, 2}));
  //
  Lower l2 = (M2() << 3, 0, 1, 3).finished();
  EXPECT_TRUE(is_near(l1, l2));
  Upper u2 = (M2() << 3, 1, 0, 3).finished();
  EXPECT_TRUE(is_near(u1, u2));
  //
  EXPECT_TRUE(is_near(Lower(EigenDiagonal {3., 4}), (M2() << 3, 0, 0, 4).finished()));
  EXPECT_TRUE(is_near(Upper(EigenDiagonal {3., 4}), (M2() << 3, 0, 0, 4).finished()));
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
  Lower l5 = EigenTriangularMatrix<decltype(M2::Zero()), TriangleType::lower>(M2::Zero()); // compatible triangular matrix
  EXPECT_TRUE(is_near(l5, M2::Zero()));
  Upper u5 = EigenTriangularMatrix<decltype(M2::Zero()), TriangleType::upper>(M2::Zero()); // compatible triangular matrix
  EXPECT_TRUE(is_near(u5, M2::Zero()));
  //
  Lower l7 = EigenSelfAdjointMatrix<M2, TriangleType::lower>{9, 3, 3, 10}; // compatible self-adjoint matrix
  EXPECT_TRUE(is_near(l7, ml));
  Upper u7 = EigenSelfAdjointMatrix<M2, TriangleType::upper>{9, 3, 3, 10}; // compatible self-adjoint matrix
  EXPECT_TRUE(is_near(u7, mu));
  //
  Lower l8 = EigenSelfAdjointMatrix<M2, TriangleType::upper>{9, 3, 3, 10}; // compatible opposite self-adjoint matrix
  EXPECT_TRUE(is_near(l8, ml));
  Upper u8 = EigenSelfAdjointMatrix<M2, TriangleType::lower>{9, 3, 3, 10}; // compatible opposite self-adjoint matrix
  EXPECT_TRUE(is_near(u8, mu));
  Diagonal d9 {3, 7, 8, 3}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d9, Mat {3, 0, 0, 3}));
  Diagonal2 d9b {3, 3}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d9b, Mat {3, 0, 0, 3}));
  Diagonal3 d9c {3, 3}; // Construct from list of scalars.
  EXPECT_TRUE(is_near(d9c, Mat {3, 0, 0, 3}));
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
  l2 = EigenTriangularMatrix<decltype(M2::Zero()), TriangleType::lower>(M2::Zero()); // copy assignment from compatible triangular matrix
  EXPECT_TRUE(is_near(l2, M2::Zero()));
  u2 = EigenTriangularMatrix<decltype(M2::Zero()), TriangleType::upper>(M2::Zero()); // copy assignment from compatible triangular matrix
  EXPECT_TRUE(is_near(u2, M2::Zero()));
  //
  l2 = EigenSelfAdjointMatrix<M2, TriangleType::lower>{9, 3, 3, 10}; // compatible self-adjoint matrix
  EXPECT_TRUE(is_near(l2, ml));
  u2 = EigenSelfAdjointMatrix<M2, TriangleType::upper>{9, 3, 3, 10}; // compatible self-adjoint matrix
  EXPECT_TRUE(is_near(u2, mu));
  //
  l2 = EigenSelfAdjointMatrix<M2, TriangleType::upper>{9, 3, 3, 10}; // compatible opposite self-adjoint matrix
  EXPECT_TRUE(is_near(l2, ml));
  u2 = EigenSelfAdjointMatrix<M2, TriangleType::lower>{9, 3, 3, 10}; // compatible opposite self-adjoint matrix
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
  EXPECT_TRUE(is_near(l1, Mat {6., 0, 2, 6}));
  u1 += u2;
  EXPECT_TRUE(is_near(u1, Mat {6., 2, 0, 6}));
  //
  l1 -= EigenTriangularMatrix {3., 0, 1, 3};
  EXPECT_TRUE(is_near(l1, Mat {3., 0, 1, 3}));
  u1 -= Upper {3., 1, 0, 3};
  EXPECT_TRUE(is_near(u1, Mat {3., 1, 0, 3}));
  //
  l1 *= 3;
  EXPECT_TRUE(is_near(l1, Mat {9., 0, 3, 9}));
  u1 *= 3;
  EXPECT_TRUE(is_near(u1, Mat {9., 3, 0, 9}));
  //
  l1 /= 3;
  EXPECT_TRUE(is_near(l1, Mat {3., 0, 1, 3}));
  u1 /= 3;
  EXPECT_TRUE(is_near(u1, Mat {3., 1, 0, 3}));
  //
  l2 *= l1;
  EXPECT_TRUE(is_near(l2, Mat {9., 0, 6, 9}));
  u2 *= u1;
  EXPECT_TRUE(is_near(u2, Mat {9., 6, 0, 9}));
  //
  EXPECT_TRUE(is_near(l1, Mat {3., 0, 1, 3}));
  EXPECT_TRUE(is_near(u1, Mat {3., 1, 0, 3}));
  //
  // Tests for Cholesky_factor and Cholesky_square are done as part of TriangularMatrix_overloads, below.
  //
  EXPECT_TRUE(is_near(l1.solve((Eigen::Matrix<double, 2, 1>() << 3, 7).finished()), (Eigen::Matrix<double, 2, 1>() << 1, 2).finished()));
  EXPECT_TRUE(is_near(u1.solve((Eigen::Matrix<double, 2, 1>() << 3, 9).finished()), (Eigen::Matrix<double, 2, 1>() << 0, 3).finished()));
  //
  EXPECT_EQ(l1(0, 1), 0);
  EXPECT_EQ(u1(1, 0), 0);
  //
  EXPECT_EQ(l1(1, 1), 3);
  EXPECT_EQ(u1(1, 1), 3);
  //
  l1(0, 0) = 5;
  l1(1, 0) = 6;
  l1(0, 1) = 7; // Should have no effect
  EXPECT_EQ(l1(0, 1), 0);
  u1(0, 0) = 5;
  u1(0, 1) = 6;
  u1(1, 0) = 7; // Should have no effect
  EXPECT_EQ(u1(1, 0), 0);
  //
  l1(1, 1) = 8;
  EXPECT_TRUE(is_near(l1, Mat {5, 0, 6, 8}));
  u1(1, 1) = 8;
  EXPECT_TRUE(is_near(u1, Mat {5, 6, 0, 8}));
  //
  EXPECT_NEAR((EigenTriangularMatrix<Eigen::Matrix<double, 1, 1>, TriangleType::lower> {7.})(0), 7., 1e-6);
  EXPECT_NEAR((EigenTriangularMatrix<Eigen::Matrix<double, 1, 1>, TriangleType::upper> {7.})(0), 7., 1e-6);
  EXPECT_NEAR((Diagonal {2, 1, 0, 3})(0), 2, 1e-6);
  EXPECT_NEAR((Diagonal {2, 1, 0, 3})(1), 3, 1e-6);
  EXPECT_NEAR((Diagonal2 {2, 3})(0), 2, 1e-6);
  EXPECT_NEAR((Diagonal2 {2, 3})(1), 3, 1e-6);
  EXPECT_NEAR((Diagonal3 {2, 3})(0), 2, 1e-6);
  EXPECT_NEAR((Diagonal3 {2, 3})(1), 3, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((Upper {3., 1, 0, 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((Diagonal {3., 1, 0, 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((Diagonal2 {3., 3})(0, 0), 3, 1e-6);
  EXPECT_NEAR((Diagonal3 {3., 3})(0, 0), 3, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 3})(0, 1), 0, 1e-6);
  EXPECT_NEAR((Upper {3., 1, 0, 3})(0, 1), 1, 1e-6);
  EXPECT_NEAR((Diagonal {3., 1, 0, 3})(0, 1), 0, 1e-6);
  EXPECT_NEAR((Diagonal2 {3., 3})(0, 1), 0, 1e-6);
  EXPECT_NEAR((Diagonal3 {3., 3})(0, 1), 0, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 3})(1, 0), 1, 1e-6);
  EXPECT_NEAR((Upper {3., 1, 0, 3})(1, 0), 0, 1e-6);
  EXPECT_NEAR((Diagonal {3., 1, 0, 3})(1, 0), 0, 1e-6);
  EXPECT_NEAR((Diagonal2 {3., 3})(1, 0), 0, 1e-6);
  EXPECT_NEAR((Diagonal3 {3., 3})(1, 0), 0, 1e-6);
  //
  EXPECT_NEAR((Lower {3., 0, 1, 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((Upper {3., 1, 0, 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((Diagonal {3., 1, 0, 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((Diagonal2 {3., 3})(1, 1), 3, 1e-6);
  EXPECT_NEAR((Diagonal3 {3., 3})(1, 1), 3, 1e-6);
}

TEST_F(matrix_tests, TriangularMatrix_traits)
{
  M2 ml, mu;
  ml << 3, 0, 1, 3;
  mu << 3, 1, 0, 3;
  using Dl = EigenTriangularMatrix<M2, TriangleType::lower>;
  using Du = EigenTriangularMatrix<M2, TriangleType::upper>;
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
}

TEST_F(matrix_tests, TriangularMatrix_overloads)
{
  M2 ml, mu;
  ml << 3, 0, 1, 3;
  mu << 3, 1, 0, 3;
  //
  EXPECT_TRUE(is_near(strict_matrix(Lower(3., 0, 1, 3)), ml));
  EXPECT_TRUE(is_near(strict_matrix(Upper(3., 1, 0, 3)), mu));
  //
  EXPECT_TRUE(is_near(strict(Lower(M2::Zero())), (M2() << 0, 0, 0, 0).finished()));
  EXPECT_TRUE(is_near(strict(Upper(M2::Zero())), (M2() << 0, 0, 0, 0).finished()));
  static_assert(std::is_same_v<std::decay_t<decltype(strict(Lower {9, 3, 3, 10} * 2))>, Lower>);
  static_assert(std::is_same_v<std::decay_t<decltype(strict(Upper {9, 3, 3, 10} * 2))>, Upper>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(EigenTriangularMatrix<decltype(M2::Identity()), TriangleType::lower>(M2::Identity())), M2::Identity()));
  EXPECT_TRUE(is_near(Cholesky_square(EigenTriangularMatrix<decltype(M2::Identity()), TriangleType::upper>(M2::Identity())), M2::Identity()));
  static_assert(is_identity_v<decltype(Cholesky_square(EigenTriangularMatrix<decltype(M2::Identity()), TriangleType::lower>(M2::Identity())))>);
  static_assert(is_identity_v<decltype(Cholesky_square(EigenTriangularMatrix<decltype(M2::Identity()), TriangleType::upper>(M2::Identity())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(EigenTriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero())), M2::Zero()));
  EXPECT_TRUE(is_near(Cholesky_square(EigenTriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero())), M2::Zero()));
  static_assert(is_zero_v<decltype(Cholesky_square(EigenTriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero())))>);
  static_assert(is_zero_v<decltype(Cholesky_square(EigenTriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(EigenTriangularMatrix<decltype(EigenDiagonal{2., 3}), TriangleType::lower>(EigenDiagonal{2., 3})), EigenDiagonal{4., 9}));
  EXPECT_TRUE(is_near(Cholesky_square(EigenTriangularMatrix<decltype(EigenDiagonal{2., 3}), TriangleType::upper>(EigenDiagonal{2., 3})), EigenDiagonal{4., 9}));
  static_assert(is_EigenDiagonal_v<decltype(Cholesky_square(EigenTriangularMatrix<decltype(EigenDiagonal{2., 3}), TriangleType::lower>(EigenDiagonal{2., 3})))>);
  static_assert(is_EigenDiagonal_v<decltype(Cholesky_square(EigenTriangularMatrix<decltype(EigenDiagonal{2., 3}), TriangleType::upper>(EigenDiagonal{2., 3})))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(EigenTriangularMatrix<M2, TriangleType::diagonal>(ml)), EigenDiagonal{9., 9}));
  static_assert(is_EigenDiagonal_v<decltype(Cholesky_square(EigenTriangularMatrix<M2, TriangleType::diagonal>(ml)))>);
  //
  EXPECT_TRUE(is_near(Cholesky_square(Lower {3., 0, 1, 3}), Mat {9., 3, 3, 10}));
  EXPECT_TRUE(is_near(Cholesky_square(Upper {3., 1, 0, 3}), Mat {9., 3, 3, 10}));
  static_assert(is_Eigen_lower_storage_triangle_v<decltype(Cholesky_square(Lower {3, 0, 1, 3}))>);
  static_assert(is_Eigen_upper_storage_triangle_v<decltype(Cholesky_square(Upper {3, 1, 0, 3}))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(EigenTriangularMatrix<decltype(M2::Identity()), TriangleType::lower>(M2::Identity())), M2::Identity()));
  EXPECT_TRUE(is_near(Cholesky_factor(EigenTriangularMatrix<decltype(M2::Identity()), TriangleType::upper>(M2::Identity())), M2::Identity()));
  static_assert(is_identity_v<decltype(Cholesky_factor(EigenTriangularMatrix<decltype(M2::Identity()), TriangleType::lower>(M2::Identity())))>);
  static_assert(is_identity_v<decltype(Cholesky_factor(EigenTriangularMatrix<decltype(M2::Identity()), TriangleType::upper>(M2::Identity())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(EigenTriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero())), M2::Zero()));
  EXPECT_TRUE(is_near(Cholesky_factor(EigenTriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero())), M2::Zero()));
  static_assert(is_zero_v<decltype(Cholesky_factor(EigenTriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::lower>(MatrixTraits<M2>::zero())))>);
  static_assert(is_zero_v<decltype(Cholesky_factor(EigenTriangularMatrix<decltype(MatrixTraits<M2>::zero()), TriangleType::upper>(MatrixTraits<M2>::zero())))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(EigenTriangularMatrix<decltype(EigenDiagonal{4., 9}), TriangleType::lower>(EigenDiagonal{4., 9})), EigenDiagonal{2., 3}));
  EXPECT_TRUE(is_near(Cholesky_factor(EigenTriangularMatrix<decltype(EigenDiagonal{4., 9}), TriangleType::upper>(EigenDiagonal{4., 9})), EigenDiagonal{2., 3}));
  static_assert(is_EigenDiagonal_v<decltype(Cholesky_factor(EigenTriangularMatrix<decltype(EigenDiagonal{4., 9}), TriangleType::lower>(EigenDiagonal{4., 9})))>);
  static_assert(is_EigenDiagonal_v<decltype(Cholesky_factor(EigenTriangularMatrix<decltype(EigenDiagonal{4., 9}), TriangleType::upper>(EigenDiagonal{4., 9})))>);
  //
  EXPECT_TRUE(is_near(Cholesky_factor(EigenTriangularMatrix<M2, TriangleType::diagonal>(ml)), EigenDiagonal{std::sqrt(3.), std::sqrt(3.)}));
  static_assert(is_EigenDiagonal_v<decltype(Cholesky_factor(EigenTriangularMatrix<M2, TriangleType::diagonal>(ml)))>);
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
  EXPECT_TRUE(is_near(solve(Lower {3., 0, 1, 3}, (Eigen::Matrix<double, 2, 1>() << 3, 7).finished()), Mean {1., 2}));
  EXPECT_TRUE(is_near(solve(Upper {3., 1, 0, 3}, (Eigen::Matrix<double, 2, 1>() << 3, 9).finished()), Mean {0., 3}));
  //
  EXPECT_TRUE(is_near(reduce_columns(Lower {3., 0, 1, 3}), Mean {1.5, 2}));
  EXPECT_TRUE(is_near(reduce_columns(Upper {3., 1, 0, 3}), Mean {2, 1.5}));
  //
  auto sl1 = Lower {3., 0, 1, 3};
  rank_update(sl1, (M2() << 2, 0, 1, 2).finished(), 4);
  EXPECT_TRUE(is_near(sl1, Mat {5., 0, 2.2, std::sqrt(25.16)}));
  auto su1 = Upper {3., 1, 0, 3};
  rank_update(su1, (M2() << 2, 0, 1, 2).finished(), 4);
  EXPECT_TRUE(is_near(su1, Mat {5., 2.2, 0, std::sqrt(25.16)}));
  //
  const auto sl2 = Lower {3., 0, 1, 3};
  EXPECT_TRUE(is_near(rank_update(sl2, (M2() << 2, 0, 1, 2).finished(), 4), Mat {5., 0, 2.2, std::sqrt(25.16)}));
  const auto su2 = Upper {3., 1, 0, 3};
  EXPECT_TRUE(is_near(rank_update(su2, (M2() << 2, 0, 1, 2).finished(), 4), Mat {5., 2.2, 0, std::sqrt(25.16)}));
  //
  EXPECT_TRUE(is_near(rank_update(Lower {3., 0, 1, 3}, (M2() << 2, 0, 1, 2).finished(), 4), Mat {5., 0, 2.2, std::sqrt(25.16)}));
  EXPECT_TRUE(is_near(rank_update(Upper {3., 1, 0, 3}, (M2() << 2, 0, 1, 2).finished(), 4), Mat {5., 2.2, 0, std::sqrt(25.16)}));
  //
  EXPECT_TRUE(is_near(solve(Lower {3., 0, 1, 3}, (Eigen::Matrix<double, 2, 1>() << 3, 7).finished()), (Eigen::Matrix<double, 2, 1>() << 1, 2).finished()));
  EXPECT_TRUE(is_near(solve(Upper {3., 1, 0, 3}, (Eigen::Matrix<double, 2, 1>() << 3, 9).finished()), (Eigen::Matrix<double, 2, 1>() << 0, 3).finished()));
  //
  EXPECT_TRUE(is_near(LQ_decomposition(Lower {3., 0, 1, 3}), Mat {3., 0, 1, 3}));
  EXPECT_TRUE(is_near(QR_decomposition(Upper {3., 1, 0, 3}), Mat {3., 1, 0, 3}));
  EXPECT_TRUE(is_near(Cholesky_square(LQ_decomposition(Upper {3., 1, 0, 3})), Mat {10, 3, 3, 9}));
  EXPECT_TRUE(is_near(Cholesky_square(QR_decomposition(Lower {3., 0, 1, 3})), Mat {10, 3, 3, 9}));
}

TEST_F(matrix_tests, TriangularMatrix_blocks_lower)
{
  auto m0 = EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>, TriangleType::lower> {1, 0, 0,
                                                                                     2, 4, 0,
                                                                                     3, 5, 6};
  auto m1 = EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>, TriangleType::lower> {4, 0, 0,
                                                                                     5, 7, 0,
                                                                                     6, 8, 9};
  EXPECT_TRUE(is_near(concatenate(EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 0, 2, 3}, m1),
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1., 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}));
  EXPECT_TRUE(is_near(concatenate_vertical(m0, m1),
    TypedMatrix<Axes<6>, Axes<3>> {1., 0, 0,
                                   2, 4, 0,
                                   3, 5, 6,
                                   4, 0, 0,
                                   5, 7, 0,
                                   6, 8, 9}));
  EXPECT_TRUE(is_near(concatenate_horizontal(m0, m1),
    TypedMatrix<Axes<3>, Axes<6>> {1, 0, 0, 4, 0, 0,
                                   2, 4, 0, 5, 7, 0,
                                   3, 5, 6, 6, 8, 9}));
  EXPECT_TRUE(is_near(split_diagonal(EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple{}));
  EXPECT_TRUE(is_near(split<2, 3>(
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1., 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple{EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 0, 2, 3}, m1}));
  EXPECT_TRUE(is_near(split<2, 2>(
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1., 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple{EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 0, 2, 3},
               EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {4., 0, 5, 7}}));
  EXPECT_TRUE(is_near(split_vertical(EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_vertical<2, 3>(
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1, 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple{TypedMatrix<Axes<2>, Axes<5>> {1, 0, 0, 0, 0,
                                              2, 3, 0, 0, 0},
               TypedMatrix<Axes<3>, Axes<5>> {0, 0, 4, 0, 0,
                                              0, 0, 5, 7, 0,
                                              0, 0, 6, 8, 9}}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1, 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple{TypedMatrix<Axes<2>, Axes<5>> {1, 0, 0, 0, 0,
                                              2, 3, 0, 0, 0},
               TypedMatrix<Axes<2>, Axes<5>> {0, 0, 4, 0, 0,
                                              0, 0, 5, 7, 0}}));
  EXPECT_TRUE(is_near(split_horizontal(EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::lower> {1., 2, 2, 3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_horizontal<2, 3>(
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1, 0, 0, 0, 0,
                                                                              2, 3, 0, 0, 0,
                                                                              0, 0, 4, 0, 0,
                                                                              0, 0, 5, 7, 0,
                                                                              0, 0, 6, 8, 9}),
    std::tuple{TypedMatrix<Axes<5>, Axes<2>> {1, 0, 2, 3, 0, 0, 0, 0, 0, 0},
               TypedMatrix<Axes<5>, Axes<3>> {0, 0, 0, 0, 0, 0, 4, 0, 0, 5, 7, 0, 6, 8, 9}}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::lower> {1, 0, 0, 0, 0,
                                                                             2, 3, 0, 0, 0,
                                                                             0, 0, 4, 0, 0,
                                                                             0, 0, 5, 7, 0,
                                                                             0, 0, 6, 8, 9}),
    std::tuple{TypedMatrix<Axes<5>, Axes<2>> {1, 0, 2, 3, 0, 0, 0, 0, 0, 0},
               TypedMatrix<Axes<5>, Axes<2>> {0, 0, 0, 0, 4, 0, 5, 7, 6, 8}}));
  EXPECT_TRUE(is_near(column(m1, 2), Mean{0., 0, 9}));
  EXPECT_TRUE(is_near(column<1>(m1), Mean{0., 7, 8}));
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

TEST_F(matrix_tests, TriangularMatrix_blocks_upper)
{
  auto m0 = EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>, TriangleType::upper> {1, 2, 3,
                                                                                     0, 4, 5,
                                                                                     0, 0, 6};
  auto m1 = EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>, TriangleType::upper> {4., 5, 6,
                                                                                     0, 7, 8,
                                                                                     0, 0, 9};
  EXPECT_TRUE(is_near(concatenate(EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 0, 3}, m1),
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}));
  EXPECT_TRUE(is_near(concatenate_vertical(m0, m1),
    TypedMatrix<Axes<6>, Axes<3>> {1., 2, 3,
                                   0, 4, 5,
                                   0, 0, 6,
                                   4, 5, 6,
                                   0, 7, 8,
                                   0, 0, 9}));
  EXPECT_TRUE(is_near(concatenate_horizontal(m0, m1),
    TypedMatrix<Axes<3>, Axes<6>> {1, 2, 3, 4, 5, 6,
                                   0, 4, 5, 0, 7, 8,
                                   0, 0, 6, 0, 0, 9}));
  EXPECT_TRUE(is_near(split_diagonal<2, 3>(
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple{EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 0, 3}, m1}));
  EXPECT_TRUE(is_near(split(EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1., 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple{EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 0, 3},
               EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {4., 5, 0, 7}}));
  EXPECT_TRUE(is_near(split_vertical(EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_vertical<2, 3>(
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple{TypedMatrix<Axes<2>, Axes<5>> {1, 2, 0, 0, 0,
                                              0, 3, 0, 0, 0},
               TypedMatrix<Axes<3>, Axes<5>> {0, 0, 4, 5, 6,
                                              0, 0, 0, 7, 8,
                                              0, 0, 0, 0, 9}}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple{TypedMatrix<Axes<2>, Axes<5>> {1, 2, 0, 0, 0,
                                              0, 3, 0, 0, 0},
               TypedMatrix<Axes<2>, Axes<5>> {0, 0, 4, 5, 6,
                                              0, 0, 0, 7, 8}}));
  EXPECT_TRUE(is_near(split_horizontal(EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>, TriangleType::upper> {1., 2, 2, 3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_horizontal<2, 3>(
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple{TypedMatrix<Axes<5>, Axes<2>> {1, 2, 0, 3, 0, 0, 0, 0, 0, 0},
               TypedMatrix<Axes<5>, Axes<3>> {0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 7, 8, 0, 0, 9}}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(
    EigenTriangularMatrix<Eigen::Matrix<double, 5, 5>, TriangleType::upper> {1, 2, 0, 0, 0,
                                                                             0, 3, 0, 0, 0,
                                                                             0, 0, 4, 5, 6,
                                                                             0, 0, 0, 7, 8,
                                                                             0, 0, 0, 0, 9}),
    std::tuple{TypedMatrix<Axes<5>, Axes<2>> {1, 2, 0, 3, 0, 0, 0, 0, 0, 0},
               TypedMatrix<Axes<5>, Axes<2>> {0, 0, 0, 0, 4, 5, 0, 7, 0, 0}}));
  EXPECT_TRUE(is_near(column(m1, 2), Mean{6., 8, 9}));
  EXPECT_TRUE(is_near(column<1>(m1), Mean{5., 7, 0}));
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

TEST_F(matrix_tests, TriangularMatrix_arithmetic_lower)
{
  auto m1 = Lower {4., 0, 5, 6};
  auto m2 = Lower {1., 0, 2, 3};
  auto d = EigenDiagonal<Eigen::Matrix<double, 2, 1>> {1, 3};
  auto i = M2::Identity();
  auto z = EigenZero<M2> {};
  EXPECT_TRUE(is_near(m1 + m2, Mat {5, 0, 7, 9})); static_assert(is_lower_triangular_v<decltype(m1 + m2)>);
  EXPECT_TRUE(is_near(m1 + d, Mat {5, 0, 5, 9})); static_assert(is_lower_triangular_v<decltype(m1 + d)>);
  EXPECT_TRUE(is_near(d + m1, Mat {5, 0, 5, 9})); static_assert(is_lower_triangular_v<decltype(d + m1)>);
  EXPECT_TRUE(is_near(m1 + i, Mat {5, 0, 5, 7})); static_assert(is_lower_triangular_v<decltype(m1 + i)>);
  EXPECT_TRUE(is_near(i + m1, Mat {5, 0, 5, 7})); static_assert(is_lower_triangular_v<decltype(i + m1)>);
  EXPECT_TRUE(is_near(m1 + z, Mat {4, 0, 5, 6})); static_assert(is_lower_triangular_v<decltype(m1 + z)>);
  EXPECT_TRUE(is_near(z + m1, Mat {4, 0, 5, 6})); static_assert(is_lower_triangular_v<decltype(z + m1)>);

  EXPECT_TRUE(is_near(m1 - m2, Mat {3, 0, 3, 3})); static_assert(is_lower_triangular_v<decltype(m1 - m2)>);
  EXPECT_TRUE(is_near(m1 - d, Mat {3, 0, 5, 3})); static_assert(is_lower_triangular_v<decltype(m1 - d)>);
  EXPECT_TRUE(is_near(d - m1, Mat {-3, 0, -5, -3})); static_assert(is_lower_triangular_v<decltype(d - m1)>);
  EXPECT_TRUE(is_near(m1 - i, Mat {3, 0, 5, 5})); static_assert(is_lower_triangular_v<decltype(m1 - i)>);
  EXPECT_TRUE(is_near(i - m1, Mat {-3, 0, -5, -5})); static_assert(is_lower_triangular_v<decltype(i - m1)>);
  EXPECT_TRUE(is_near(m1 - z, Mat {4, 0, 5, 6})); static_assert(is_lower_triangular_v<decltype(m1 - z)>);
  EXPECT_TRUE(is_near(z - m1, Mat {-4, 0, -5, -6})); static_assert(is_lower_triangular_v<decltype(z - m1)>);

  EXPECT_TRUE(is_near(m1 * 2, Mat {8, 0, 10, 12})); static_assert(is_lower_triangular_v<decltype(m1 * 2)>);
  EXPECT_TRUE(is_near(2 * m1, Mat {8, 0, 10, 12})); static_assert(is_lower_triangular_v<decltype(2 * m1)>);
  EXPECT_TRUE(is_near(m1 / 2, Mat {2, 0, 2.5, 3})); static_assert(is_lower_triangular_v<decltype(m1 / 2)>);
  static_assert(is_lower_triangular_v<decltype(m1 / 0)>);
  EXPECT_TRUE(is_near(-m1, Mat {-4, 0, -5, -6}));  static_assert(is_lower_triangular_v<decltype(-m1)>);

  EXPECT_TRUE(is_near(m1 * m2, Mat {4, 0, 17, 18})); static_assert(is_lower_triangular_v<decltype(m1 * m2)>);
  EXPECT_TRUE(is_near(m1 * d, Mat {4, 0, 5, 18})); static_assert(is_lower_triangular_v<decltype(m1 * d)>);
  EXPECT_TRUE(is_near(d * m1, Mat {4, 0, 15, 18})); static_assert(is_lower_triangular_v<decltype(d * m2)>);
  EXPECT_TRUE(is_near(m1 * i, m1));  static_assert(is_lower_triangular_v<decltype(m1 * i)>);
  EXPECT_TRUE(is_near(i * m1, m1));  static_assert(is_lower_triangular_v<decltype(i * m1)>);
  EXPECT_TRUE(is_near(m1 * z, z));  static_assert(is_zero_v<decltype(m1 * z)>);
  EXPECT_TRUE(is_near(z * m1, z));  static_assert(is_zero_v<decltype(z * m1)>);

  EXPECT_TRUE(is_near(Mat {4, 0, 5, 6}.base_matrix() * m2, Mat {4, 0, 17, 18}));
  EXPECT_TRUE(is_near(m1 * Mat {1, 0, 2, 3}.base_matrix(), Mat {4, 0, 17, 18}));
}

TEST_F(matrix_tests, TriangularMatrix_arithmetic_upper)
{
  auto m1 = Upper {4., 5, 0, 6};
  auto m2 = Upper {1., 2, 0, 3};
  auto d = EigenDiagonal<Eigen::Matrix<double, 2, 1>> {1, 3};
  auto i = M2::Identity();
  auto z = EigenZero<M2> {};
  EXPECT_TRUE(is_near(m1 + m2, Mat {5, 7, 0, 9})); static_assert(is_upper_triangular_v<decltype(m1 + m2)>);
  EXPECT_TRUE(is_near(m1 + d, Mat {5, 5, 0, 9})); static_assert(is_upper_triangular_v<decltype(m1 + d)>);
  EXPECT_TRUE(is_near(d + m1, Mat {5, 5, 0, 9})); static_assert(is_upper_triangular_v<decltype(d + m1)>);
  EXPECT_TRUE(is_near(m1 + i, Mat {5, 5, 0, 7})); static_assert(is_upper_triangular_v<decltype(m1 + i)>);
  EXPECT_TRUE(is_near(i + m1, Mat {5, 5, 0, 7})); static_assert(is_upper_triangular_v<decltype(i + m1)>);
  EXPECT_TRUE(is_near(m1 + z, Mat {4, 5, 0, 6})); static_assert(is_upper_triangular_v<decltype(m1 + z)>);
  EXPECT_TRUE(is_near(z + m1, Mat {4, 5, 0, 6})); static_assert(is_upper_triangular_v<decltype(z + m1)>);

  EXPECT_TRUE(is_near(m1 - m2, Mat {3, 3, 0, 3})); static_assert(is_upper_triangular_v<decltype(m1 - m2)>);
  EXPECT_TRUE(is_near(m1 - d, Mat {3, 5, 0, 3})); static_assert(is_upper_triangular_v<decltype(m1 - d)>);
  EXPECT_TRUE(is_near(d - m1, Mat {-3, -5, 0, -3})); static_assert(is_upper_triangular_v<decltype(d - m1)>);
  EXPECT_TRUE(is_near(m1 - i, Mat {3, 5, 0, 5})); static_assert(is_upper_triangular_v<decltype(m1 - i)>);
  EXPECT_TRUE(is_near(i - m1, Mat {-3, -5, 0, -5})); static_assert(is_upper_triangular_v<decltype(i - m1)>);
  EXPECT_TRUE(is_near(m1 - z, Mat {4, 5, 0, 6})); static_assert(is_upper_triangular_v<decltype(m1 - z)>);
  EXPECT_TRUE(is_near(z - m1, Mat {-4, -5, 0, -6})); static_assert(is_upper_triangular_v<decltype(z - m1)>);

  EXPECT_TRUE(is_near(m1 * 2, Mat {8, 10, 0, 12})); static_assert(is_upper_triangular_v<decltype(m1 * 2)>);
  EXPECT_TRUE(is_near(2 * m1, Mat {8, 10, 0, 12})); static_assert(is_upper_triangular_v<decltype(2 * m1)>);
  EXPECT_TRUE(is_near(m1 / 2, Mat {2, 2.5, 0, 3})); static_assert(is_upper_triangular_v<decltype(m1 / 2)>);
  static_assert(is_upper_triangular_v<decltype(m1 / 0)>);
  EXPECT_TRUE(is_near(-m1, Mat {-4, -5, 0, -6}));  static_assert(is_upper_triangular_v<decltype(-m1)>);

  EXPECT_TRUE(is_near(m1 * m2, Mat {4, 23, 0, 18})); static_assert(is_upper_triangular_v<decltype(m1 * m2)>);
  EXPECT_TRUE(is_near(m1 * d, Mat {4, 15, 0, 18})); static_assert(is_upper_triangular_v<decltype(m1 * d)>);
  EXPECT_TRUE(is_near(d * m1, Mat {4, 5, 0, 18})); static_assert(is_upper_triangular_v<decltype(d * m1)>);
  EXPECT_TRUE(is_near(m1 * i, m1));  static_assert(is_upper_triangular_v<decltype(m1 * i)>);
  EXPECT_TRUE(is_near(i * m1, m1));  static_assert(is_upper_triangular_v<decltype(i * m1)>);
  EXPECT_TRUE(is_near(m1 * z, z));  static_assert(is_zero_v<decltype(m1 * z)>);
  EXPECT_TRUE(is_near(z * m1, z));  static_assert(is_zero_v<decltype(z * m1)>);

  EXPECT_TRUE(is_near(Mat {4, 5, 0, 6}.base_matrix() * m2, Mat {4, 23, 0, 18}));
  EXPECT_TRUE(is_near(m1 * Mat {1, 2, 0, 3}.base_matrix(), Mat {4, 23, 0, 18}));
}

TEST_F(matrix_tests, TriangularMatrix_arithmetic_mixed)
{
  auto m_upper = Upper {4., 5, 0, 6};
  auto m_lower = Lower {1., 0, 2, 3};
  EXPECT_TRUE(is_near(m_upper + m_lower, Mat {5, 5, 2, 9}));
  EXPECT_TRUE(is_near(m_lower + m_upper, Mat {5, 5, 2, 9}));
  EXPECT_TRUE(is_near(m_upper - m_lower, Mat {3, 5, -2, 3}));
  EXPECT_TRUE(is_near(m_lower - m_upper, Mat {-3, -5, 2, -3}));
  EXPECT_TRUE(is_near(m_upper * m_lower, Mat {14, 15, 12, 18}));
  EXPECT_TRUE(is_near(m_lower * m_upper, Mat {4, 5, 8, 28}));
}

TEST_F(matrix_tests, TriangularMatrix_references)
{
  M2 m, n;
  m << 2, 0, 1, 2;
  n << 3, 0, 1, 3;
  EigenTriangularMatrix<M2, TriangleType::lower> x = m;
  EigenTriangularMatrix<M2&, TriangleType::lower> x_lvalue = x;
  EXPECT_TRUE(is_near(x_lvalue, m));
  x = n;
  EXPECT_TRUE(is_near(x_lvalue, n));
  x_lvalue = m;
  EXPECT_TRUE(is_near(x, m));
  EigenTriangularMatrix<M2&&, TriangleType::lower> x_rvalue = std::move(x);
  EXPECT_TRUE(is_near(x_rvalue, m));
  x_rvalue = n;
  EXPECT_TRUE(is_near(x_rvalue, n));
  //
  using V = EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>;
  V v1 {1., 0, 0,
        2, 4, 0,
        3, -6, -3};
  v1(0, 1) = 3.2;
  EXPECT_EQ(v1(0,1), 0); // Assigning to the upper right triangle does not change anything. It's still zero.
  EXPECT_EQ(v1(1,0), 2);
  EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  v2(2, 0) = 5.2;
  EXPECT_EQ(v1(2,0), 5.2);
  EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>&&> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 4.1);
  EigenTriangularMatrix<const Eigen::Matrix<double, 3, 3>&> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4);
}
