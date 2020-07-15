/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "covariance_tests.h"

using namespace OpenKalman;

using M = Eigen::Matrix<double, 3, 3>;
using C = Coefficients<Axis, Angle, Axis>;
using Mat3 = TypedMatrix<C, C, M>;
using SAl = EigenSelfAdjointMatrix<M, TriangleType::lower>;
using SAu = EigenSelfAdjointMatrix<M, TriangleType::upper>;
using Tl = EigenTriangularMatrix<M, TriangleType::lower>;
using Tu = EigenTriangularMatrix<M, TriangleType::upper>;
using D = EigenDiagonal<Eigen::Matrix<double, 3, 1>>;

TEST_F(covariance_tests, Covariance_references_self_adjoint_lvalue)
{
  using V = Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1., 2, 3,
        2, 4, -6,
        3, -6, -3};
  EXPECT_EQ(v1(1,0), 2);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 2);
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  EXPECT_EQ(v1(0,1), 4.1);
  v2(0, 2) = 5.2;
  EXPECT_EQ(v1(0,2), 5.2);
  EXPECT_EQ(v1(2,0), 5.2);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>&&>> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 4.1);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<const Eigen::Matrix<double, 3, 3>&>> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4);
}


TEST_F(covariance_tests, Covariance_references_self_adjoint2)
{
  using V = Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1., 2, 3,
        2, 4, -6,
        3, -6, -3};
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  EXPECT_EQ(v1(0,1), 4.1);
  v2(0,2) = 5.2;
  EXPECT_EQ(v1(0,2), 5.2);
  EXPECT_EQ(v1(2,0), 5.2);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>&&> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 4.1);
  Covariance<Coefficients<Axis, Angle, Axis>, const EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>&> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v3(2,1), 7.3);
  EXPECT_EQ(v4(2,1), 7.3);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4);
}


TEST_F(covariance_tests, Covariance_references_triangular)
{
  using V = Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1., 2, 3,
        2, 20, -18,
        3, -18, 54};
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1 = {1.1, 2.1, 3.1,
        2.1, 20.1, -18.1,
        3.1, -18.1, 54.1};
  EXPECT_TRUE(is_near(base_matrix(v1), base_matrix(v2)));
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_NEAR(v2(1,0), 2.1, 1e-6);
  v2 = V {1.2, 2.2, 3.2,
          2.2, 20.2, -18.2,
          3.2, -18.2, 54.2};
  EXPECT_NEAR(v1(2,0), 3.2, 1e-6);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>> v2_5 = v2;
  EXPECT_NEAR(v2_5(1,0), 2.2, 1e-6);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>&&>> v3 = std::move(v2);
  EXPECT_NEAR(v3(1,0), 2.2, 1e-6);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<const Eigen::Matrix<double, 3, 3>&>> v4 = v3;
  v3 = V {1.3, 2.3, 3.3,
          2.3, 20.3, -18.3,
          3.3, -18.3, 54.3};
  EXPECT_NEAR(v3(2,2), 54.3, 1e-6);
  EXPECT_NEAR(v4(2,1), -18.3, 1e-6);
  V v5 = v3;
  EXPECT_NEAR(v3(1,1), 20.3, 1e-6);
  v5 = {1.4, 2.4, 3.4,
        2.4, 20.4, -18.4,
        3.4, -18.4, 54.4};
  EXPECT_NEAR(v3(1,1), 20.3, 1e-6);
  EXPECT_NEAR(v5(1,1), 20.4, 1e-6);
  EXPECT_NEAR(v3(2,1), -18.3, 1e-6);
  EXPECT_TRUE(is_near(v3, V {1.3, 2.3, 3.3,
                             2.3, 20.3, -18.3,
                             3.3, -18.3, 54.3}));
}


TEST_F(covariance_tests, Covariance_references_triangular2)
{
  using V = Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1., 2, 3,
        2, 20, -18,
        3, -18, 54};
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1 = {1.1, 2.1, 3.1,
        2.1, 20.1, -18.1,
        3.1, -18.1, 54.1};
  EXPECT_TRUE(is_near(base_matrix(v1), base_matrix(v2)));
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_NEAR(v2(1,0), 2.1, 1e-6);
  v2 = V {1.2, 2.2, 3.2,
          2.2, 20.2, -18.2,
          3.2, -18.2, 54.2};
  EXPECT_NEAR(v1(2,0), 3.2, 1e-6);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>&&> v3 = std::move(v2);
  EXPECT_NEAR(v3(1,0), 2.2, 1e-6);
  Covariance<Coefficients<Axis, Angle, Axis>, const EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>&> v4 = v3;
  v3 = V {1.3, 2.3, 3.3,
          2.3, 20.3, -18.3,
          3.3, -18.3, 54.3};
  EXPECT_NEAR(v4(2,1), -18.3, 1e-6);
  Covariance<Coefficients<Axis, Angle, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v5 = {1.4, 2.4, 3.4,
        2.4, 20.4, -18.4,
        3.4, -18.4, 54.4};
  EXPECT_NEAR(v3(1,1), 20.3, 1e-6);
  EXPECT_NEAR(v5(1,1), 20.4, 1e-6);
  EXPECT_NEAR(v3(2,1), -18.3, 1e-6);
  EXPECT_TRUE(is_near(v3, V {1.3, 2.3, 3.3,
                             2.3, 20.3, -18.3,
                             3.3, -18.3, 54.3}));
}


TEST_F(covariance_tests, SquareRootCovariance_references_triangular)
{
  using V = SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1., 0, 0,
        2, 4, 0,
        3, -6, -3};
  v1(0, 1) = 3.2;
  EXPECT_EQ(v1(0,1), 0); // Assigning to the upper right triangle does not change anything. It's still zero.
  EXPECT_EQ(v1(1,0), 2);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  v2(2, 0) = 5.2;
  EXPECT_EQ(v1(2,0), 5.2);
  v1 = {1.4, 0, 0,
        2.4, 4.4, 0,
        3.4, -6.4, -3.4};
  EXPECT_NEAR(v2(1,0), 2.4, 1e-6);
  v2 = V {1.1, 0, 0,
          2.1, 4.1, 0,
          3.1, -6.1, -3.1};
  EXPECT_NEAR(v1(2,0), 3.1, 1e-6);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>&&>> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 2.1);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<const Eigen::Matrix<double, 3, 3>&>> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  v3 = V {1.2, 0, 0,
          2.2, 4.2, 0,
          3.2, -6.2, -3.2};
  EXPECT_NEAR(v4(2,1), -6.2, 1e-6);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4.2);
  v5(1,1) = 8.5;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 8.5);
  v5 = {1.6, 0, 0,
        2.6, 4.6, 0,
        3.6, -6.6, -3.6};
  EXPECT_NEAR(v3(1,1), 8.4, 1e-6);
  EXPECT_NEAR(v5(1,1), 4.6, 1e-6);
}


TEST_F(covariance_tests, SquareRootCovariance_references_triangular2)
{
  using V = SquareRootCovariance<Coefficients<Angle, Axis, Axis>,
    EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>, TriangleType::lower>>;
  V v1 {1., 0, 0,
        2, 4, 0,
        3, -6, -3};
  v1(0, 1) = 3.2;
  EXPECT_EQ(v1(0,1), 0); // Assigning to the upper right triangle does not change anything. It's still zero.
  EXPECT_EQ(v1(1,0), 2);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1(1,0) = 4.1;
  EXPECT_EQ(v2(1,0), 4.1);
  v2(2, 0) = 5.2;
  EXPECT_EQ(v1(2,0), 5.2);
  v1 = {1.4, 0, 0,
        2.4, 4.4, 0,
        3.4, -6.4, -3.4};
  EXPECT_NEAR(v2(1,0), 2.4, 1e-6);
  v2 = V {1.1, 0, 0,
          2.1, 4.1, 0,
          3.1, -6.1, -3.1};
  EXPECT_NEAR(v1(2,0), 3.1, 1e-6);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>&&> v3 = std::move(v2);
  EXPECT_EQ(v3(1,0), 2.1);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, const EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>&> v4 = v3;
  v3(2,1) = 7.3;
  EXPECT_EQ(v4(2,1), 7.3);
  EXPECT_TRUE(is_near(v4, V {
    1.1, 0, 0,
    2.1, 4.1, 0,
    3.1, 7.3, -3.1}));
  v3 = V {1.2, 0, 0,
          2.2, 4.2, 0,
          3.2, -6.2, -3.2};
  EXPECT_NEAR(v4(2,1), -6.2, 1e-6);
  SquareRootCovariance<Coefficients<Angle, Axis, Axis>, EigenTriangularMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v3(1,1) = 8.4;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 4.2);
  v5(1,1) = 8.5;
  EXPECT_EQ(v3(1,1), 8.4);
  EXPECT_EQ(v5(1,1), 8.5);
  v5 = {1.6, 0, 0,
        2.6, 4.6, 0,
        3.6, -6.6, -3.6};
  EXPECT_NEAR(v3(1,1), 8.4, 1e-6);
  EXPECT_NEAR(v5(1,1), 4.6, 1e-6);
}


TEST_F(covariance_tests, SquareRootCovariance_references_self_adjoint)
{
  using V = SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1, 2, 3,
        2, 4, -6,
        3, -6, -3};
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1 = {1.4, 0, 0,
        2.4, 4.4, 0,
        3.4, -6.4, -3.4};
  EXPECT_NEAR(v2(1,0), 2.4, 1e-6);
  v2 = V {1.1, 2.1, 3.1,
          2.1, 4.1, -6.1,
          3.1, -6.1, -3.1};
  EXPECT_NEAR(v1(2,0), 3.1, 1e-6);
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>&&>> v3 = std::move(v2);
  EXPECT_NEAR(v3(1,0), 2.1, 1e-6);
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<const Eigen::Matrix<double, 3, 3>&>> v4 = v3;
  v3 = V {1.2, 2.2, 3.2,
          2.2, 4.2, -6.2,
          3.2, -6.2, -3.2};
  EXPECT_NEAR(v4(2,1), -6.2, 1e-6);
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v5 = {1.3, 2.3, 3.3,
        2.3, 4.3, -6.3,
        3.3, -6.3, -3.3};
  EXPECT_NEAR(v3(1,1), 4.2, 1e-6);
  EXPECT_NEAR(v5(1,1), 4.3, 1e-6);
}


TEST_F(covariance_tests, SquareRootCovariance_references_self_adjoint2)
{
  using V = SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>>;
  V v1 {1., 2, 3,
        2, 4, -6,
        3, -6, -3};
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  v1 = {1.4, 2.4, 3.4,
        2.4, 4.4, -6.4,
        3.4, -6.4, -3.4};
  EXPECT_NEAR(v2(1,0), 2.4, 1e-6);
  v2 = V {1.1, 2.1, 3.1,
          2.1, 4.1, -6.1,
          3.1, -6.1, -3.1};
  EXPECT_NEAR(v1(2,0), 3.1, 1e-6);
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>&&> v3 = std::move(v2);
  EXPECT_NEAR(v3(1,0), 2.1, 1e-6);
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, const EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>&> v4 = v3;
  v3 = V {1.2, 2.2, 3.2,
          2.2, 4.2, -6.2,
          3.2, -6.2, -3.2};
  EXPECT_NEAR(v4(2,1), -6.2, 1e-6);
  SquareRootCovariance<Coefficients<Axis, Angle, Axis>, EigenSelfAdjointMatrix<Eigen::Matrix<double, 3, 3>>> v5 = v3;
  v5 = {1.3, 2.3, 3.3,
        2.3, 4.3, -6.3,
        3.3, -6.3, -3.3};
  EXPECT_NEAR(v3(1,1), 4.2, 1e-6);
  EXPECT_NEAR(v5(1,1), 4.3, 1e-6);
}

