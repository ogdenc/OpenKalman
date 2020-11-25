/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "covariance_tests.hpp"

using namespace OpenKalman;

using M = native_matrix_t<double, 3, 3>;
using M1 = native_matrix_t<double, 3, 1>;
using C = Coefficients<Axis, angle::Radians, Axis>;
using Mat3 = Matrix<C, C, M>;
using Mat31 = Matrix<C, Axis, M1>;
template<typename Mat> using SAl = SelfAdjointMatrix<Mat, TriangleType::lower>;
template<typename Mat> using SAu = SelfAdjointMatrix<Mat, TriangleType::upper>;
template<typename Mat> using Tl = TriangularMatrix<Mat, TriangleType::lower>;
template<typename Mat> using Tu = TriangularMatrix<Mat, TriangleType::upper>;
template<typename Mat> using D = DiagonalMatrix<Mat>;


TEST_F(covariance_tests, References_Covariance_self_adjoint_lvalue1)
{
  using V = Covariance<C, SAl<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, SAl<M>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  v2(0,2) = 4.1;
  EXPECT_EQ(v1(2,0), 4.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {16.3, 8.3, 4.3,
          8.3, 29.3, -13.3,
          4.3, -13.3, 46.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_self_adjoint_lvalue2)
{
  using V = Covariance<C, SAu<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, SAu<M&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  v2(0,2) = 4.1;
  EXPECT_EQ(v1(2,0), 4.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {16.3, 8.3, 4.3,
          8.3, 29.3, -13.3,
          4.3, -13.3, 46.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_self_adjoint_const_lvalue1)
{
  using V = Covariance<C, SAl<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, const SAl<M>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_self_adjoint_const_lvalue2)
{
  using V = Covariance<C, SAu<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, SAu<const M&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_triangular_lvalue1)
{
  using V = Covariance<C, Tl<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, Tl<M>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  v2(0,2) = 4.1;
  EXPECT_EQ(v1(2,0), 4.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {16.3, 8.3, 4.3,
          8.3, 29.3, -13.3,
          4.3, -13.3, 46.3};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(covariance_tests, References_Covariance_triangular_lvalue2)
{
  using V = Covariance<C, Tu<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, Tu<M&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  v2(0,2) = 4.1;
  EXPECT_EQ(v1(2,0), 4.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {16.3, 8.3, 4.3,
          8.3, 29.3, -13.3,
          4.3, -13.3, 46.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_triangular_const_lvalue1)
{
  using V = Covariance<C, Tl<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, const Tl<M>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_triangular_const_lvalue2)
{
  using V = Covariance<C, Tu<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, Tu<const M&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_diagonal_lvalue1)
{
  using V = Covariance<C, D<M1>>;
  V v1 {1, 2, 3};
  Covariance<C, D<M1>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  v2[2] = 3.1;
  EXPECT_EQ(v1[2], 3.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {1.3, 2.3, 3.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_diagonal_lvalue2)
{
  using V = Covariance<C, D<M1>>;
  V v1 {1, 2, 3};
  Covariance<C, D<M1&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  v2[2] = 3.1;
  EXPECT_EQ(v1[2], 3.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {1.3, 2.3, 3.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_diagonal_const_lvalue1)
{
  using V = Covariance<C, D<M1>>;
  V v1 {1, 2, 3};
  Covariance<C, const D<M1>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_diagonal_const_lvalue2)
{
  using V = Covariance<C, D<M1>>;
  V v1 {1, 2, 3};
  Covariance<C, D<const M1&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_SquareRootCovariance_self_adjoint_lvalue1)
{
  using V = SquareRootCovariance<C, SAl<M>>;
  V v1 {4, 0, 0,
        2, 5, 0,
        1, -3, 6};
  SquareRootCovariance<C, SAl<M>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 2);
  v1(1,0) = 2.1;
  EXPECT_EQ(v2(1,0), 2.1);
  v2(2,0) = 1.1;
  EXPECT_EQ(v1(2,0), 1.1);
  v1 = {4.2, 0, 0,
        2.2, 5.2, 0,
        1.2, -3.2, 6.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {4.3, 0, 0,
          2.3, 5.3, 0,
          1.3, -3.3, 6.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_SquareRootCovariance_self_adjoint_lvalue2)
{
  using V = SquareRootCovariance<C, SAu<M>>;
  V v1 {4, 2, 1,
        0, 5, -3,
        0, 0, 6};
  SquareRootCovariance<C, SAu<M&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(0,1), 2);
  v1(0,1) = 2.1;
  EXPECT_EQ(v2(0,1), 2.1);
  v2(0,2) = 1.1;
  EXPECT_EQ(v1(0,2), 1.1);
  v1 = {4.2, 2.2, 1.2,
        0, 5.2, -3.2,
        0, 0, 6.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {4.3, 2.3, 1.3,
          0, 5.3, -3.3,
          0, 0, 6.3};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(covariance_tests, References_SquareRootCovariance_self_adjoint_const_lvalue1)
{
  using V = SquareRootCovariance<C, SAl<M>>;
  V v1 {4, 0, 0,
        2, 5, 0,
        1, -3, 6};
  SquareRootCovariance<C, const SAl<M>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 2);
  v1(1,0) = 2.1;
  EXPECT_EQ(v2(1,0), 2.1);
  v1 = {4.2, 0, 0,
        2.2, 5.2, 0,
        1.2, -3.2, 6.2};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_SquareRootCovariance_self_adjoint_const_lvalue2)
{
  using V = SquareRootCovariance<C, SAu<M>>;
  V v1 {4, 2, 1,
        0, 5, -3,
        0, 0, 6};
  SquareRootCovariance<C, SAu<const M&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(0,1), 2);
  v1(0,1) = 2.1;
  EXPECT_EQ(v2(0,1), 2.1);
  v1 = {4.2, 2.2, 1.2,
        0, 5.2, -3.2,
        0, 0, 6.2};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_SquareRootCovariance_triangular_lvalue1)
{
  using V = SquareRootCovariance<C, Tl<M>>;
  V v1 {4, 0, 0,
        2, 5, 0,
        1, -3, 6};
  SquareRootCovariance<C, Tl<M>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 2);
  v1(1,0) = 2.1;
  EXPECT_EQ(v2(1,0), 2.1);
  v2(2,0) = 1.1;
  EXPECT_EQ(v1(2,0), 1.1);
  v1 = {4.2, 0, 0,
        2.2, 5.2, 0,
        1.2, -3.2, 6.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {4.3, 0, 0,
          2.3, 5.3, 0,
          1.3, -3.3, 6.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_SquareRootCovariance_triangular_lvalue2)
{
  using V = SquareRootCovariance<C, Tu<M>>;
  V v1 {4, 2, 1,
        0, 5, -3,
        0, 0, 6};
  SquareRootCovariance<C, Tu<M&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(0,1), 2);
  v1(0,1) = 2.1;
  EXPECT_EQ(v2(0,1), 2.1);
  v2(0,2) = 1.1;
  EXPECT_EQ(v1(0,2), 1.1);
  v1 = {4.2, 2.2, 1.2,
        0, 5.2, -3.2,
        0, 0, -6.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {4.3, 2.3, 1.3,
          0, 5.3, -3.3,
          0, 0, -6.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_SquareRootCovariance_triangular_const_lvalue1)
{
  using V = SquareRootCovariance<C, Tl<M>>;
  V v1 {4, 0, 0,
        2, 5, 0,
        1, -3, 6};
  SquareRootCovariance<C, const Tl<M>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 2);
  v1(1,0) = 2.1;
  EXPECT_EQ(v2(1,0), 2.1);
  v1 = {4.2, 0, 0,
        2.2, 5.2, 0,
        1.2, -3.2, 6.2};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(covariance_tests, References_SquareRootCovariance_triangular_const_lvalue2)
{
  using V = SquareRootCovariance<C, Tu<M>>;
  V v1 {4, 2, 1,
        0, 5, -3,
        0, 0, 6};
  SquareRootCovariance<C, Tu<const M&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(0,1), 2);
  v1(0,1) = 2.1;
  EXPECT_EQ(v2(0,1), 2.1);
  v1 = {4.2, 2.2, 1.2,
        0, 5.2, -3.2,
        0, 0, -6.2};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(covariance_tests, References_SquareRootCovariance_diagonal_lvalue1)
{
  using V = SquareRootCovariance<C, D<M1>>;
  V v1 {1, 2, 3};
  SquareRootCovariance<C, D<M1>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  v2[2] = 3.1;
  EXPECT_EQ(v1[2], 3.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {1.3, 2.3, 3.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_SquareRootCovariance_diagonal_lvalue2)
{
  using V = SquareRootCovariance<C, D<M1>>;
  V v1 {1, 2, 3};
  SquareRootCovariance<C, D<M1&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  v2[2] = 3.1;
  EXPECT_EQ(v1[2], 3.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {1.3, 2.3, 3.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_SquareRootCovariance_diagonal_const_lvalue1)
{
  using V = SquareRootCovariance<C, D<M1>>;
  V v1 {1, 2, 3};
  SquareRootCovariance<C, const D<M1>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_SquareRootCovariance_diagonal_const_lvalue2)
{
  using V = SquareRootCovariance<C, D<M1>>;
  V v1 {1, 2, 3};
  SquareRootCovariance<C, D<const M1&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
}


///////////////////////////////////////////////////////////////////////////////////////////
//                                Cross-assignment                                       //
///////////////////////////////////////////////////////////////////////////////////////////

TEST_F(covariance_tests, References_Covariance_cross_self_adjoint_lvalue1)
{
  using V = SquareRootCovariance<C, SAl<M>>;
  V v1 {4, 0, 0,
        2, 5, 0,
        1, -3, 6};
  Covariance<C, SAl<M>&> v2 = v1;
  EXPECT_TRUE(is_near(v2, Mat3 {16, 8, 4, 8, 29, -13, 4, -13, 46}));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 2.1;
  EXPECT_EQ(v2(0,1), 8.4);
  v2(1,1) = 30.4;
  v2(2,1) = 17.4;
  v1 = {4.1, 0, 0,
        2.1, 5.1, 0,
        1.1, 3.1, 6.1};
  EXPECT_TRUE(is_near(v2, Mat3 {16.81, 8.61, 4.51, 8.61, 30.42, 18.12, 4.51, 18.12, 48.03}));
  v2 = Mat3 {17.64, 9.24, 5.04,
             9.24, 31.88, 19.28,
             5.04, 19.28, 50.12};
  EXPECT_TRUE(is_near(v1, Mat3 {4.2, 0, 0, 2.2, 5.2, 0, 1.2, 3.2, 6.2}));
}

TEST_F(covariance_tests, References_Covariance_cross_self_adjoint_lvalue2)
{
  using V = SquareRootCovariance<C, SAu<M>>;
  V v1 {4, 2, 1,
        0, 5, -3,
        0, 0, 6};
  Covariance<C, SAu<M&>> v2 = v1;
  EXPECT_TRUE(is_near(v2, Mat3 {16, 8, 4, 8, 29, -13, 4, -13, 46}));
  EXPECT_EQ(v2(1,0), 8);
  v1(0,1) = 2.1;
  EXPECT_EQ(v2(0,1), 8.4);
  v2(1,1) = 30.4;
  v2(1,2) = 17.4;
  v1 = {4.1, 2.1, 1.1,
        0, 5.1, 3.1,
        0, 0, 6.1};
  EXPECT_TRUE(is_near(v2, Mat3 {16.81, 8.61, 4.51, 8.61, 30.42, 18.12, 4.51, 18.12, 48.03}));
  v2 = Mat3 {17.64, 9.24, 5.04,
             9.24, 31.88, 19.28,
             5.04, 19.28, 50.12};
  EXPECT_TRUE(is_near(v1, Mat3 {4.2, 2.2, 1.2, 0, 5.2, 3.2, 0, 0, 6.2}));
}


TEST_F(covariance_tests, References_Covariance_cross_self_adjoint_const_lvalue1)
{
  using V = Covariance<C, SAl<M>>;
  V v1 {4, 2, 1,
        2, 5, -3,
        1, -3, -6};
  Covariance<C, const SAl<M>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 2);
  v1(1,0) = 2.1;
  EXPECT_EQ(v2(0,1), 2.1);
  v1 = {4.2, 2.2, 1.2,
        2.2, 5.2, -3.2,
        1.2, -3.2, -6.2};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_cross_self_adjoint_const_lvalue2)
{
  using V = Covariance<C, SAu<M>>;
  V v1 {4, 2, 1,
        2, 5, -3,
        1, -3, -6};
  Covariance<C, SAu<const M&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 2);
  v1(1,0) = 2.1;
  EXPECT_EQ(v2(0,1), 2.1);
  v1 = {4.2, 2.2, 1.2,
        2.2, 5.2, -3.2,
        1.2, -3.2, -6.2};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(covariance_tests, References_Covariance_cross_triangular_lvalue1)
{
  using V = Covariance<C, Tl<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, Tl<M>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  v2(0,2) = 4.1;
  EXPECT_EQ(v1(2,0), 4.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {16.3, 8.3, 4.3,
          8.3, 29.3, -13.3,
          4.3, -13.3, 46.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_cross_triangular_lvalue2)
{
  using V = Covariance<C, Tu<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, Tu<M&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  v2(0,2) = 4.1;
  EXPECT_EQ(v1(2,0), 4.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {16.3, 8.3, 4.3,
          8.3, 29.3, -13.3,
          4.3, -13.3, 46.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_cross_triangular_const_lvalue1)
{
  using V = Covariance<C, Tl<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, const Tl<M>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_cross_triangular_const_lvalue2)
{
  using V = Covariance<C, Tu<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, Tu<const M&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(covariance_tests, References_Covariance_cross_diagonal_lvalue1)
{
  using V = Covariance<C, D<M1>>;
  V v1 {1, 2, 3};
  Covariance<C, D<M1>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  v2[2] = 3.1;
  EXPECT_EQ(v1[2], 3.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {1.3, 2.3, 3.3};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_cross_diagonal_lvalue2)
{
  using V = Covariance<C, D<M1>>;
  V v1 {1, 2, 3};
  Covariance<C, D<M1&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  v2[2] = 3.1;
  EXPECT_EQ(v1[2], 3.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {1.3, 2.3, 3.3};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(covariance_tests, References_Covariance_cross_diagonal_const_lvalue1)
{
  using V = Covariance<C, D<M1>>;
  V v1 {1, 2, 3};
  Covariance<C, const D<M1>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
}


TEST_F(covariance_tests, References_Covariance_cross_diagonal_const_lvalue2)
{
  using V = Covariance<C, D<M1>>;
  V v1 {1, 2, 3};
  Covariance<C, D<const M1&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
}

