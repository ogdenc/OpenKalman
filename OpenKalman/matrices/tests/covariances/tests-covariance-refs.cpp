/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "covariance_tests.hpp"

using namespace OpenKalman;

using M = eigen_matrix_t<double, 3, 3>;
using M1 = eigen_matrix_t<double, 3, 1>;
using C = Coefficients<Axis, angle::Radians, Axis>;
using Mat3 = Matrix<C, C, M>;
using Mat31 = Matrix<C, Axis, M1>;
template<typename Mat> using SAl = SelfAdjointMatrix<Mat, TriangleType::lower>;
template<typename Mat> using SAu = SelfAdjointMatrix<Mat, TriangleType::upper>;
template<typename Mat> using Tl = TriangularMatrix<Mat, TriangleType::lower>;
template<typename Mat> using Tu = TriangularMatrix<Mat, TriangleType::upper>;
template<typename Mat> using D = DiagonalMatrix<Mat>;


// -----------Case 2 from Case 1----------- //

TEST(covariance_tests, References_2from1_Cov_SAl_1)
{
  using V = Covariance<C, SAl<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, SAl<M>&> v2 {v1};
  EXPECT_TRUE(is_near(v1, v2));
  Covariance<C, const SAl<M>&> v2c {v1};
  EXPECT_TRUE(is_near(v1, v2c));
  EXPECT_EQ(v2(1,0), 8);
  EXPECT_EQ(v2c(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  EXPECT_EQ(v2c(0,1), 8.1);
  v2(0,2) = 4.1;
  EXPECT_EQ(v1(2,0), 4.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_TRUE(is_near(v1, v2c));
  v2 = V {16.3, 8.3, 4.3,
          8.3, 29.3, -13.3,
          4.3, -13.3, 46.3};
  EXPECT_TRUE(is_near(v1, v2));

  // case 1 from case 2:
  V v1a = v2;
  EXPECT_TRUE(is_near(v1a, v1));
  Covariance<C, const SAl<M>> v1ac {v2};
  EXPECT_TRUE(is_near(v1ac, v1));
  EXPECT_TRUE(is_near(adjoint(v1), v1));

  // case 2 from case 2
  Covariance<C, SAl<M>&> v2a {v2};
  EXPECT_TRUE(is_near(v1, v2a));
  Covariance<C, const SAl<M>&> v2ac = v2;
  EXPECT_TRUE(is_near(v1, v2ac));
  v1 = {16.4, 8.4, 4.4,
        8.4, 29.4, -13.4,
        4.4, -13.4, 46.4};
  EXPECT_EQ(v2a(2,0), 4.4);
  EXPECT_EQ(v2ac(2,0), 4.4);
  EXPECT_TRUE(is_near(v1, v2a));
  EXPECT_TRUE(is_near(v1, v2ac));
  v2a = V {16.5, 8.5, 4.5,
          8.5, 29.5, -13.5,
          4.5, -13.5, 46.5};
  EXPECT_EQ(v1(2,0), 4.5);
  EXPECT_EQ(v2ac(2,0), 4.5);
  EXPECT_TRUE(is_near(v1, v2a));
}


TEST(covariance_tests, References_2from1_Cov_SAl_2)
{
  using V = Covariance<C, SAu<M>>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, SAu<M&>> v2 {v1};
  EXPECT_TRUE(is_near(v1, v2));
  Covariance<C, SAu<const M&>> v2c {v1};
  EXPECT_TRUE(is_near(v1, v2c));
  EXPECT_EQ(v2(1,0), 8);
  EXPECT_EQ(v2c(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  EXPECT_EQ(v2c(0,1), 8.1);
  v2(0,2) = 4.1;
  EXPECT_EQ(v1(2,0), 4.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_TRUE(is_near(v1, v2c));
  v2 = V {16.3, 8.3, 4.3,
          8.3, 29.3, -13.3,
          4.3, -13.3, 46.3};
  EXPECT_TRUE(is_near(v1, v2));

  // case 1 from case 2:
  V v1a = v2;
  EXPECT_TRUE(is_near(v1a, v1));
  Covariance<C, SAu<const M>> v1ac {v2};
  EXPECT_TRUE(is_near(v1ac, v1));
  EXPECT_TRUE(is_near(adjoint(v1), v1));
  EXPECT_TRUE(is_near(adjoint(v2), v1));
}


TEST(covariance_tests, References_2from1_SqCov_Tl_1)
{
  using V = SquareRootCovariance<C, Tl<M>>;
  V v1 {4, 0, 0,
        2, 5, 0,
        1, -3, 6};
  SquareRootCovariance<C, Tl<M>&> v2 {v1};
  EXPECT_TRUE(is_near(v1, v2));
  SquareRootCovariance<C, const Tl<M>&> v2c {v1};
  EXPECT_TRUE(is_near(v1, v2c));
  EXPECT_EQ(v2(1,0), 2);
  EXPECT_EQ(v2c(1,0), 2);
  v1(1,0) = 2.1;
  EXPECT_EQ(v2(1,0), 2.1);
  EXPECT_EQ(v2c(1,0), 2.1);
  v2(2,0) = 1.1;
  EXPECT_EQ(v1(2,0), 1.1);
  v1 = {4.2, 0, 0,
        2.2, 5.2, 0,
        1.2, -3.2, 6.2};
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_TRUE(is_near(v1, v2c));
  v2 = V {4.3, 0, 0,
          2.3, 5.3, 0,
          1.3, -3.3, 6.3};
  EXPECT_TRUE(is_near(v1, v2));

  // case 1 from case 2:
  V v1a = v2;
  EXPECT_TRUE(is_near(v1a, v1));
  SquareRootCovariance<C, const Tl<M>> v1ac {v2};
  EXPECT_TRUE(is_near(v1ac, v1));
  EXPECT_TRUE(is_near(adjoint(v1), adjoint(v2)));
}


TEST(covariance_tests, References_2from1_SqCov_Tu_2)
{
  using V = SquareRootCovariance<C, Tu<M>>;
  V v1 {4, 2, 1,
        0, 5, -3,
        0, 0, 6};
  SquareRootCovariance<C, Tu<M&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  SquareRootCovariance<C, Tu<const M&>> v2c = v1;
  EXPECT_TRUE(is_near(v1, v2c));
  EXPECT_EQ(v2(0,1), 2);
  EXPECT_EQ(v2c(0,1), 2);
  v1(0,1) = 2.1;
  EXPECT_EQ(v2(0,1), 2.1);
  EXPECT_EQ(v2c(0,1), 2.1);
  v2(0,2) = 1.1;
  EXPECT_EQ(v1(0,2), 1.1);
  v1 = {4.2, 2.2, 1.2,
        0, 5.2, -3.2,
        0, 0, -6.2};
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_TRUE(is_near(v1, v2c));
  v2 = V {4.3, 2.3, 1.3,
          0, 5.3, -3.3,
          0, 0, -6.3};
  EXPECT_TRUE(is_near(v1, v2));

  // case 1 from case 2:
  V v1a = v2;
  EXPECT_TRUE(is_near(v1a, v1));
  SquareRootCovariance<C, Tu<const M>> v1ac = v2;
  EXPECT_TRUE(is_near(v1ac, v1));
  EXPECT_TRUE(is_near(adjoint(v1), adjoint(v2)));

  // case 2 from case 2:
  SquareRootCovariance<C, Tu<M&>> v2a = v2;
  EXPECT_TRUE(is_near(v1, v2a));
  SquareRootCovariance<C, Tu<const M&>> v2ac = v2;
  EXPECT_TRUE(is_near(v1, v2ac));
  v1 = {4.4, 2.4, 1.4,
        0, 5.4, -3.4,
        0, 0, -6.4};
  EXPECT_EQ(v2a(0,2), 1.4);
  EXPECT_EQ(v2ac(0,2), 1.4);
  EXPECT_TRUE(is_near(v1, v2a));
  EXPECT_TRUE(is_near(v1, v2ac));
  v2a = V {4.5, 2.5, 1.5,
          0, 5.5, -3.5,
          0, 0, -6.5};
  EXPECT_EQ(v1(0,2), 1.5);
  EXPECT_EQ(v2ac(0,2), 1.5);
  EXPECT_TRUE(is_near(v1, v2a));
}


TEST(covariance_tests, References_2from1_Cov_D_1)
{
  using V = Covariance<C, D<M1>>;
  V v1 {1, 2, 3};
  Covariance<C, D<M1>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  Covariance<C, const D<M1>&> v2c = v1;
  EXPECT_TRUE(is_near(v1, v2c));
  EXPECT_EQ(v2[1], 2);
  EXPECT_EQ(v2c[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  EXPECT_EQ(v2c[1], 2.1);
  v2[2] = 3.1;
  EXPECT_EQ(v1[2], 3.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_TRUE(is_near(v1, v2c));
  v2 = V {1.3, 2.3, 3.3};
  EXPECT_TRUE(is_near(v1, v2));

  // case 1 from case 2:
  V v1a = v2;
  EXPECT_TRUE(is_near(v1a, v1));
  Covariance<C, const D<M1>> v1ac = v2;
  EXPECT_TRUE(is_near(v1ac, v1));
  EXPECT_TRUE(is_near(adjoint(v1), v1));
}


TEST(covariance_tests, References_2from1_Cov_D_2)
{
  using V = Covariance<C, D<M1>>;
  V v1 {1, 2, 3};
  Covariance<C, D<M1&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  Covariance<C, D<const M1&>> v2c = v1;
  EXPECT_TRUE(is_near(v1, v2c));
  EXPECT_EQ(v2[1], 2);
  EXPECT_EQ(v2c[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  EXPECT_EQ(v2c[1], 2.1);
  v2[2] = 3.1;
  EXPECT_EQ(v1[2], 3.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_TRUE(is_near(v1, v2c));
  v2 = V {1.3, 2.3, 3.3};
  EXPECT_TRUE(is_near(v1, v2));

  // case 1 from case 2:
  V v1a = v2;
  EXPECT_TRUE(is_near(v1a, v1));
  Covariance<C, D<const M1>> v1ac = v2;
  EXPECT_TRUE(is_near(v1ac, v1));
  EXPECT_TRUE(is_near(adjoint(v1), v1));
}


TEST(covariance_tests, References_2from1_SqCov_TuD_2)
{
  using V = SquareRootCovariance<C, Tu<D<M1>>>;
  V v1 {1, 2, 3};
  SquareRootCovariance<C, Tu<D<M1&>>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  SquareRootCovariance<C, Tu<D<const M1&>>> v2c = v1;
  EXPECT_TRUE(is_near(v1, v2c));
  EXPECT_EQ(v2[1], 2);
  EXPECT_EQ(v2c[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  EXPECT_EQ(v2c[1], 2.1);
  v2[2] = 3.1;
  EXPECT_EQ(v1[2], 3.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_TRUE(is_near(v1, v2c));
  v2 = V {1.3, 2.3, 3.3};
  EXPECT_TRUE(is_near(v1, v2));

  // case 1 from case 2:
  V v1a = v2;
  EXPECT_TRUE(is_near(v1a, v1));
  SquareRootCovariance<C, Tu<D<const M1>>> v1ac = v2;
  EXPECT_TRUE(is_near(v1ac, v1));
  EXPECT_TRUE(is_near(adjoint(v1), adjoint(v2)));
}


TEST(covariance_tests, References_2from1_SqCov_D_1)
{
  using V = SquareRootCovariance<C, D<M1>>;
  V v1 {1, 2, 3};
  SquareRootCovariance<C, D<M1>&> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  SquareRootCovariance<C, const D<M1>&> v2c = v1;
  EXPECT_TRUE(is_near(v1, v2c));
  EXPECT_EQ(v2[1], 2);
  EXPECT_EQ(v2c[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  EXPECT_EQ(v2c[1], 2.1);
  v2[2] = 3.1;
  EXPECT_EQ(v1[2], 3.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_TRUE(is_near(v1, v2c));
  v2 = V {1.3, 2.3, 3.3};
  EXPECT_TRUE(is_near(v1, v2));

  // case 1 from case 2:
  V v1a = v2;
  EXPECT_TRUE(is_near(v1a, v1));
  SquareRootCovariance<C, const D<M1>> v1ac = v2;
  EXPECT_TRUE(is_near(v1ac, v1));
  EXPECT_TRUE(is_near(adjoint(v1), adjoint(v2)));
}


TEST(covariance_tests, References_2from1_SqCov_D_2)
{
  using V = SquareRootCovariance<C, D<M1>>;
  V v1 {1, 2, 3};
  SquareRootCovariance<C, D<M1&>> v2 = v1;
  EXPECT_TRUE(is_near(v1, v2));
  SquareRootCovariance<C, D<const M1&>> v2c = v1;
  EXPECT_TRUE(is_near(v1, v2c));
  EXPECT_EQ(v2[1], 2);
  EXPECT_EQ(v2c[1], 2);
  v1[1] = 2.1;
  EXPECT_EQ(v2[1], 2.1);
  EXPECT_EQ(v2c[1], 2.1);
  v2[2] = 3.1;
  EXPECT_EQ(v1[2], 3.1);
  v1 = {1.2, 2.2, 3.2};
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_TRUE(is_near(v1, v2c));
  v2 = V {1.3, 2.3, 3.3};
  EXPECT_TRUE(is_near(v1, v2));

  // case 1 from case 2:
  V v1a = v2;
  EXPECT_TRUE(is_near(v1a, v1));
  SquareRootCovariance<C, D<const M1>> v1ac = v2;
  EXPECT_TRUE(is_near(v1ac, v1));
  EXPECT_TRUE(is_near(adjoint(v1), adjoint(v2)));
}


// -----------Case 2 from Case 3----------- //

TEST(covariance_tests, References_2from3_SqCov_1)
{
  using V3 = Covariance<C, Tl<M>>; // case 3
  V3 v3 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  using V1 = SquareRootCovariance<C, Tl<M>>;
  EXPECT_TRUE(is_near(v3.square_root(), Mat3 {4, 0, 0, 2, 5, 0, 1, -3, 6}));
  auto v2 = v3.square_root(); // case 3 -> case 2
  EXPECT_TRUE(is_near(v2, Mat3 {4, 0, 0, 2, 5, 0, 1, -3, 6}));
  SquareRootCovariance<C, const Tl<M>&> v2c {v3.square_root()}; // case 3 -> case 2 -> case 2 (const)
  EXPECT_TRUE(is_near(v2c, Mat3 {4, 0, 0, 2, 5, 0, 1, -3, 6}));
  EXPECT_NEAR(v2(1,0), 2, 1e-6);
  EXPECT_NEAR(v2c(1,0), 2, 1e-6);
  v3(2,2) = 47.21;
  EXPECT_NEAR(v2(2,2), 6.1, 1e-6);
  EXPECT_NEAR(v2c(2,2), 6.1, 1e-6);
  v2(1,1) = 5.1;
  v2(2,1) = -3.1;
  EXPECT_TRUE(is_near(v3, Mat3 {16, 8, 4, 8, 30.01, -13.81, 4, -13.81, 47.82}));
  v3 = {17.64, 9.24, 5.04,
        9.24, 31.88, -14,
        5.04, -14, 50.12};
  EXPECT_TRUE(is_near(v2, Mat3 {4.2, 0, 0, 2.2, 5.2, 0, 1.2, -3.2, 6.2}));
  EXPECT_TRUE(is_near(v2c, Mat3 {4.2, 0, 0, 2.2, 5.2, 0, 1.2, -3.2, 6.2}));
  v2 = V1 {4.3, 0, 0,
           2.3, 5.3, 0,
           1.3, -3.3, 6.3};
  EXPECT_TRUE(is_near(v3, Mat3 {18.49, 9.89, 5.59, 9.89, 33.38, -14.5, 5.59, -14.5, 52.27}));

  // case 3 from case 2:
  V3 v3a = v2.square(); // case 2 -> case 4 -> case 3
  EXPECT_TRUE(is_near(v3a, v3));
  Covariance<C, const Tl<M>> v3ac = v2.square();
  EXPECT_TRUE(is_near(v3ac, v3));
  V3 v3b = SquareRootCovariance<C, Tl<M>&> {v2}.square(); // case 2 -> case 3
  EXPECT_TRUE(is_near(v3b, v3));
  Covariance<C, const Tl<M>> v3bc = SquareRootCovariance<C, const Tl<M>&> {v2}.square();
  EXPECT_TRUE(is_near(v3bc, v3));
  EXPECT_TRUE(is_near(adjoint(v2.square()), adjoint(v3))); // adjoint(case 4 from case 2) == adjoint(case 3)

  // case 2 from case 2:
  SquareRootCovariance<C, Tl<M>&> v2a = v2; // case 2
  EXPECT_TRUE(is_near(v2a, v2));
  SquareRootCovariance<C, const Tl<M>&> v2ac = v2; // case 2
  EXPECT_TRUE(is_near(v2ac, v2));
  v2 = V1 {4.4, 0, 0,
           2.4, 5.4, 0,
           1.4, -3.4, 6.4};
  EXPECT_EQ(v2a(2,1), -3.4);
  EXPECT_EQ(v2ac(2,1), -3.4);
  EXPECT_TRUE(is_near(v2a, v2));
  EXPECT_TRUE(is_near(v2ac, v2));
  EXPECT_NEAR(v3(2,2), 54.48, 1e-6);
  v2a = V1 {4.5, 0, 0,
            2.5, 5.5, 0,
            1.5, -3.5, 6.5};
  EXPECT_NEAR(v3(2,1), -15.5, 1e-6);
  EXPECT_TRUE(is_near(v2a, v2));
}

TEST(covariance_tests, References_2from3_Cov_2)
{
  using V3 = SquareRootCovariance<C, SAu<M>>; // case 3
  V3 v3 {4, 2, 1,
         0, 5, -3,
         0, 0, 6};
  using V1 = Covariance<C, SAu<M>>;
  EXPECT_TRUE(is_near(v3.square(), Mat3 {16, 8, 4, 8, 29, -13, 4, -13, 46}));
  auto v2 = v3.square(); // case 3 -> case 2
  EXPECT_TRUE(is_near(v2, Mat3 {16, 8, 4, 8, 29, -13, 4, -13, 46}));
  Covariance<C, SAu<const M&>> v2c = v3.square(); // case 3 -> case 2 -> case 2 (const)
  EXPECT_TRUE(is_near(v2c, Mat3 {16, 8, 4, 8, 29, -13, 4, -13, 46}));
  EXPECT_NEAR(v2(1,0), 8, 1e-6);
  EXPECT_NEAR(v2c(1,0), 8, 1e-6);
  v3(2,2) = 6.1;
  EXPECT_NEAR(v2(2,2), 47.21, 1e-6);
  EXPECT_NEAR(v2c(2,2), 47.21, 1e-6);
  v2(1,2) = -13.5;
  v2(2,2) = 47.82;
  EXPECT_NEAR(v3(1,2), -3.1, 1e-6);
  EXPECT_TRUE(is_near(v2, Mat3 {16, 8, 4, 8, 29, -13.5, 4, -13.5, 47.82}));
  v3 = {4.2, 2.2, 1.2,
        0, 5.2, -3.2,
        0, 0, 6.2};
  EXPECT_TRUE(is_near(v2, Mat3 {17.64, 9.24, 5.04, 9.24, 31.88, -14, 5.04, -14, 50.12}));
  EXPECT_TRUE(is_near(v2c, Mat3 {17.64, 9.24, 5.04, 9.24, 31.88, -14, 5.04, -14, 50.12}));
  v2 = V1 {18.49, 9.89, 5.59,
           9.89, 33.38, -14.5,
           5.59, -14.5, 52.27};
  EXPECT_TRUE(is_near(v3, Mat3 {4.3, 2.3, 1.3, 0, 5.3, -3.3, 0, 0, 6.3}));

  // case 3 from case 2:
  V3 v3a = v2.square_root();
  EXPECT_TRUE(is_near(v3a, v3));
  SquareRootCovariance<C, SAu<const M>> v3ac = v2.square_root();
  EXPECT_TRUE(is_near(v3ac, v3));
  V3 v3b = Covariance<C, SAu<M&>> {v2}.square_root();
  EXPECT_TRUE(is_near(v3b, v3));
  SquareRootCovariance<C, SAu<const M>> v3bc = Covariance<C, const SAu<M&>> {v2}.square_root();
  EXPECT_TRUE(is_near(v3bc, v3));
  EXPECT_TRUE(is_near(adjoint(v3a), adjoint(v2.square_root())));
}


// -----------Case 2 from nestable----------- //

TEST(covariance_tests, References_2from_nestable_SAl_1)
{
  using V = SAl<M>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, SAl<M>&> v2 {v1};
  EXPECT_TRUE(is_near(v1, v2));
  Covariance<C, const SAl<M>&> v2c {v1};
  EXPECT_TRUE(is_near(v1, v2c));
  EXPECT_EQ(v2(1,0), 8);
  EXPECT_EQ(v2c(1,0), 8);
  v1(1,0) = 8.1;
  EXPECT_EQ(v2(0,1), 8.1);
  EXPECT_EQ(v2c(0,1), 8.1);
  v2(0,2) = 4.1;
  EXPECT_EQ(v1(2,0), 4.1);
  v1 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_TRUE(is_near(v1, v2c));
  v2 = V {16.3, 8.3, 4.3,
          8.3, 29.3, -13.3,
          4.3, -13.3, 46.3};
  EXPECT_TRUE(is_near(v1, v2));

  // case 1 from case 2:
  Covariance<C, SAl<M>> v1a = v2;
  EXPECT_TRUE(is_near(v1a, v1));
  Covariance<C, const SAl<M>> v1ac = v2;
  EXPECT_TRUE(is_near(v1ac, v1));
  EXPECT_TRUE(is_near(adjoint(v1), v1));

  // case 2 from case 2
  Covariance<C, SAl<M>&> v2a = v2;
  EXPECT_TRUE(is_near(v1, v2a));
  Covariance<C, const SAl<M>&> v2ac = v2;
  EXPECT_TRUE(is_near(v1, v2ac));
  v1 = {16.4, 8.4, 4.4,
        8.4, 29.4, -13.4,
        4.4, -13.4, 46.4};
  EXPECT_EQ(v2a(2,0), 4.4);
  EXPECT_EQ(v2ac(2,0), 4.4);
  EXPECT_TRUE(is_near(v1, v2a));
  EXPECT_TRUE(is_near(v1, v2ac));
  v2a = V {16.5, 8.5, 4.5,
           8.5, 29.5, -13.5,
           4.5, -13.5, 46.5};
  EXPECT_EQ(v1(2,0), 4.5);
  EXPECT_EQ(v2ac(2,0), 4.5);
  EXPECT_TRUE(is_near(v1, v2a));
}


TEST(covariance_tests, References_2from_nestable_Tu_2)
{
  using V = Tu<M>;
  V v1 {4, 2, 1,
        0, 5, -3,
        0, 0, 6};
  SquareRootCovariance<C, Tu<M&>> v2 {v1};
  EXPECT_TRUE(is_near(v1, v2));
  SquareRootCovariance<C, Tu<const M&>> v2c {v1};
  EXPECT_TRUE(is_near(v1, v2c));
  EXPECT_EQ(v2(0,1), 2);
  EXPECT_EQ(v2c(0,1), 2);
  v1(0,1) = 2.1;
  EXPECT_EQ(v2(0,1), 2.1);
  EXPECT_EQ(v2c(0,1), 2.1);
  v2(0,2) = 1.1;
  EXPECT_EQ(v1(0,2), 1.1);
  v1 = {4.2, 2.2, 1.2,
        0, 5.2, -3.2,
        0, 0, -6.2};
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_TRUE(is_near(v1, v2c));
  v2 = V {4.3, 2.3, 1.3,
          0, 5.3, -3.3,
          0, 0, -6.3};
  EXPECT_TRUE(is_near(v1, v2));

  // case 1 from case 2:
  SquareRootCovariance<C, Tu<M>> v1a = v2;
  EXPECT_TRUE(is_near(v1a, v1));
  SquareRootCovariance<C, Tu<const M>> v1ac = v2;
  EXPECT_TRUE(is_near(v1ac, v1));
  EXPECT_TRUE(is_near(adjoint(v1), adjoint(v2)));

  // case 2 from case 2:
  SquareRootCovariance<C, Tu<M&>> v2a = v2;
  EXPECT_TRUE(is_near(v1, v2a));
  SquareRootCovariance<C, Tu<const M&>> v2ac = v2;
  EXPECT_TRUE(is_near(v1, v2ac));
  v1 = {4.4, 2.4, 1.4,
        0, 5.4, -3.4,
        0, 0, -6.4};
  EXPECT_EQ(v2a(0,2), 1.4);
  EXPECT_EQ(v2ac(0,2), 1.4);
  EXPECT_TRUE(is_near(v1, v2a));
  EXPECT_TRUE(is_near(v1, v2ac));
  v2a = V {4.5, 2.5, 1.5,
           0, 5.5, -3.5,
           0, 0, -6.5};
  EXPECT_EQ(v1(0,2), 1.5);
  EXPECT_EQ(v2ac(0,2), 1.5);
  EXPECT_TRUE(is_near(v1, v2a));
}


// -----------Case 4 from Case 3----------- //

TEST(covariance_tests, References_4from3_Cov_Tl_1)
{
  using V = Covariance<C, Tl<M>>;
  V v3 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, Tl<M>&> v4 = v3;
  EXPECT_TRUE(is_near(v3, v4));
  Covariance<C, const Tl<M>&> v4c = v3;
  EXPECT_TRUE(is_near(v3, v4c));
  EXPECT_EQ(v4(1,0), 8);
  EXPECT_EQ(v4c(1,0), 8);
  v3(1,0) = 8.1;
  EXPECT_EQ(v4(0,1), 8.1);
  EXPECT_EQ(v4c(0,1), 8.1);
  v4(0,2) = 4.1;
  EXPECT_EQ(v3(2,0), 4.1);
  v3 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v3, v4));
  EXPECT_TRUE(is_near(v3, v4c));
  v4 = V {16.3, 8.3, 4.3,
          8.3, 29.3, -13.3,
          4.3, -13.3, 46.3};
  EXPECT_TRUE(is_near(v3, v4));
  EXPECT_TRUE(is_near(adjoint(v3), v3));

  // Case 3 from Case 4
  V v3a {v4};
  EXPECT_TRUE(is_near(v3a, v3));
  Covariance<C, Tl<const M>> v3ac = v4;
  EXPECT_TRUE(is_near(v3ac, v3));
  EXPECT_TRUE(is_near(adjoint(v3a), v3a));

  // Case 4 from Case 4
  Covariance<C, Tl<M>&> v4a = v4; // copy constructor
  EXPECT_TRUE(is_near(v3, v4a));
  Covariance<C, const Tl<M>&> v4ac = v4;
  EXPECT_TRUE(is_near(v3, v4ac));
  v3 = {16.4, 8.4, 4.4,
        8.4, 29.4, -13.4,
        4.4, -13.4, 46.4};
  EXPECT_TRUE(is_near(v3, v4a));
  EXPECT_TRUE(is_near(v3, v4ac));
  v4a = V {16.5, 8.5, 4.5,
          8.5, 29.5, -13.5,
          4.5, -13.5, 46.5};
  EXPECT_TRUE(is_near(v3, v4a));
}

TEST(covariance_tests, References_4from3_Cov_Tu_2)
{
  using V = Covariance<C, Tu<M>>;
  V v3 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  Covariance<C, Tu<M&>> v4 = v3;
  EXPECT_TRUE(is_near(v3, v4));
  Covariance<C, Tu<const M&>> v4c = v3;
  EXPECT_TRUE(is_near(v3, v4c));
  EXPECT_EQ(v4(1,0), 8);
  EXPECT_EQ(v4c(1,0), 8);
  v3(1,0) = 8.1;
  EXPECT_EQ(v4(0,1), 8.1);
  EXPECT_EQ(v4c(0,1), 8.1);
  v4(0,2) = 4.1;
  EXPECT_EQ(v3(2,0), 4.1);
  v3 = {16.2, 8.2, 4.2,
        8.2, 29.2, -13.2,
        4.2, -13.2, 46.2};
  EXPECT_TRUE(is_near(v3, v4));
  EXPECT_TRUE(is_near(v3, v4c));
  v4 = V {16.3, 8.3, 4.3,
          8.3, 29.3, -13.3,
          4.3, -13.3, 46.3};
  EXPECT_TRUE(is_near(v3, v4));
  EXPECT_TRUE(is_near(adjoint(v3), v3));

  // Case 3 from Case 4
  V v3a = v4;
  EXPECT_TRUE(is_near(v3a, v3));
  Covariance<C, Tu<const M>> v3ac = v4;
  EXPECT_TRUE(is_near(v3ac, v3));
  EXPECT_TRUE(is_near(adjoint(v3a), v3a));

  // Case 4 from Case 4
  Covariance<C, Tu<M&>> v4x = v4; // copy constructor
  EXPECT_TRUE(is_near(v3, v4x));
  Covariance<C, Tu<M&>> v4a = std::move(v4x); // move constructor
  EXPECT_TRUE(is_near(v3, v4a));
  Covariance<C, const Tu<M&>> v4ac = v4;
  EXPECT_TRUE(is_near(v3, v4ac));
  v3 = {16.4, 8.4, 4.4,
        8.4, 29.4, -13.4,
        4.4, -13.4, 46.4};
  EXPECT_TRUE(is_near(v3, v4a));
  EXPECT_TRUE(is_near(v3, v4ac));
  v4a = V {16.5, 8.5, 4.5,
          8.5, 29.5, -13.5,
          4.5, -13.5, 46.5};
  EXPECT_TRUE(is_near(v3, v4a));
}


TEST(covariance_tests, References_4from3_SqCov_SAl_1)
{
  using V = SquareRootCovariance<C, SAl<M>>;
  V v3 {4, 0, 0,
        2, 5, 0,
        1, -3, 6};
  SquareRootCovariance<C, SAl<M>&> v4 = v3;
  EXPECT_TRUE(is_near(v3, v4));
  SquareRootCovariance<C, const SAl<M>&> v4c = v3;
  EXPECT_TRUE(is_near(v3, v4c));
  EXPECT_EQ(v4(1,0), 2);
  EXPECT_EQ(v4c(1,0), 2);
  v3(1,0) = 2.1;
  EXPECT_EQ(v4(1,0), 2.1);
  EXPECT_EQ(v4c(1,0), 2.1);
  v4(2,0) = 1.1;
  EXPECT_EQ(v3(2,0), 1.1);
  v3 = {4.2, 0, 0,
        2.2, 5.2, 0,
        1.2, -3.2, 6.2};
  EXPECT_TRUE(is_near(v3, v4));
  EXPECT_TRUE(is_near(v3, v4c));
  v4 = V {4.3, 0, 0,
          2.3, 5.3, 0,
          1.3, -3.3, 6.3};
  EXPECT_TRUE(is_near(v3, v4));
  EXPECT_TRUE(is_near(adjoint(v3), adjoint(v4)));

  // Case 3 from Case 4
  V v3a = v4;
  EXPECT_TRUE(is_near(v3a, v3));
  SquareRootCovariance<C, SAl<const M>> v3ac = v4;
  EXPECT_TRUE(is_near(v3ac, v3));
  EXPECT_TRUE(is_near(adjoint(v3a), adjoint(v3)));

  // Case 4 from Case 4
  SquareRootCovariance<C, SAl<M>&> v4a = v4;
  EXPECT_TRUE(is_near(v3, v4a));
  SquareRootCovariance<C, const SAl<M>&> v4ac = v4;
  EXPECT_TRUE(is_near(v3, v4ac));
  v3 = {4.4, 0, 0,
        2.4, 5.4, 0,
        1.4, -3.4, 6.4};
  EXPECT_TRUE(is_near(v3, v4a));
  EXPECT_TRUE(is_near(v3, v4ac));
  v4a = V {4.5, 0, 0,
          2.5, 5.5, 0,
          1.5, -3.5, 6.5};
  EXPECT_TRUE(is_near(v3, v4a));
}


TEST(covariance_tests, References_4from3_SqCov_SAu_2)
{
  using V = SquareRootCovariance<C, SAu<M>>;
  V v3 {4, 2, 1,
        0, 5, -3,
        0, 0, 6};
  SquareRootCovariance<C, SAu<M&>> v2 = v3;
  EXPECT_TRUE(is_near(v3, v2));
  SquareRootCovariance<C, SAu<const M&>> v2c = v3;
  EXPECT_TRUE(is_near(v3, v2c));
  EXPECT_EQ(v2(0,1), 2);
  EXPECT_EQ(v2c(0,1), 2);
  v3(0,1) = 2.1;
  EXPECT_EQ(v2(0,1), 2.1);
  EXPECT_EQ(v2c(0,1), 2.1);
  v2(0,2) = 1.1;
  EXPECT_EQ(v3(0,2), 1.1);
  v3 = {4.2, 2.2, 1.2,
        0, 5.2, -3.2,
        0, 0, 6.2};
  EXPECT_TRUE(is_near(v3, v2));
  EXPECT_TRUE(is_near(v3, v2c));
  v2 = V {4.3, 2.3, 1.3,
          0, 5.3, -3.3,
          0, 0, 6.3};
  EXPECT_TRUE(is_near(v3, v2));
  EXPECT_TRUE(is_near(adjoint(v3), adjoint(v2)));

  // Case 3 from Case 4
  V v3a = v2;
  EXPECT_TRUE(is_near(v3a, v3));
  SquareRootCovariance<C, SAu<const M>> v3ac = v2;
  EXPECT_TRUE(is_near(v3ac, v3));
  EXPECT_TRUE(is_near(adjoint(v3a), adjoint(v3)));

  // Case 4 from Case 4
  SquareRootCovariance<C, SAu<M&>> v4a = v2;
  EXPECT_TRUE(is_near(v3, v4a));
  SquareRootCovariance<C, const SAu<M&>> v4ac = v2;
  EXPECT_TRUE(is_near(v3, v4ac));
  v3 = {4.4, 2.4, 1.4,
        0, 5.4, -3.4,
        0, 0, 6.4};
  EXPECT_TRUE(is_near(v3, v4a));
  EXPECT_TRUE(is_near(v3, v4ac));
  v4a = V {4.5, 2.5, 1.5,
          0, 5.5, -3.5,
          0, 0, 6.5};
  EXPECT_TRUE(is_near(v3, v4a));
}


// -----------Case 4 from Case 1----------- //

TEST(covariance_tests, References_4from1_Cov_l_1)
{
  using V = SquareRootCovariance<C, Tl<M>>; // case 1
  V v1 {4, 0, 0,
        2, 5, 0,
        1, -3, 6};
  using V3 = Covariance<C, Tl<M>>;
  Mat3 m16 = Mat3 {16, 8, 4, 8, 29, -13, 4, -13, 46};
  EXPECT_TRUE(is_near(v1.square(), m16));
  auto v4 = v1.square(); // case 1 -> case 4
  EXPECT_TRUE(is_near(v4, m16));
  Covariance<C, const Tl<M>&> v4c {v1.square()}; // case 1 -> case 4 -> case 4 (const)
  EXPECT_TRUE(is_near(v4c, m16));
  EXPECT_NEAR(v4(1,0), 8, 1e-6);
  EXPECT_NEAR(v4c(1,0), 8, 1e-6);
  v1(2,2) = 6.1;
  EXPECT_NEAR(v4(2,2), 47.21, 1e-6);
  EXPECT_NEAR(v4c(2,2), 47.21, 1e-6);
  v4(2,1) = -13.5;
  v4(2,2) = 47.82;
  EXPECT_NEAR(v1(2,1), -3.1, 1e-6);
  EXPECT_TRUE(is_near(v4, Mat3 {16, 8, 4, 8, 29, -13.5, 4, -13.5, 47.82}));
  v1 = {4.2, 0, 0,
        2.2, 5.2, 0,
        1.2, -3.2, 6.2};
  Mat3 m1764 = Mat3 {17.64, 9.24, 5.04, 9.24, 31.88, -14, 5.04, -14, 50.12};
  EXPECT_TRUE(is_near(v4, m1764));
  EXPECT_TRUE(is_near(v4c, m1764));
  v4 = V3 {18.49, 9.89, 5.59,
           9.89, 33.38, -14.5,
           5.59, -14.5, 52.27};
  EXPECT_TRUE(is_near(v1, Mat3 {4.3, 0, 0, 2.3, 5.3, 0, 1.3, -3.3, 6.3}));
  EXPECT_TRUE(is_near(adjoint(v1), adjoint(v4.square_root())));

  // Case 1 from Case 4
  V v1a = v4.square_root(); // case 1
  EXPECT_TRUE(is_near(v1a, v1));
  SquareRootCovariance<C, Tl<const M>> v1ac = v4.square_root(); // case 1
  EXPECT_TRUE(is_near(v1ac, v1));
  EXPECT_TRUE(is_near(adjoint(v1a), adjoint(v4.square_root())));

  // Case 4 from Case 4
  Mat3 m1849 {18.49, 9.89, 5.59, 9.89, 33.38, -14.5, 5.59, -14.5, 52.27};
  Covariance<C, Tl<M>&> v4a = v4; // case 4
  EXPECT_TRUE(is_near(v4a, m1849));
  Covariance<C, const Tl<M>&> v4ac = v4; // case 4 -> const case 4
  EXPECT_TRUE(is_near(v4ac, m1849));
  v1 = {4.4, 0, 0,
        2.4, 5.4, 0,
        1.4, -3.4, 6.4};
  Mat3 m1936 {19.36, 10.56, 6.16, 10.56, 34.92, -15, 6.16, -15, 54.48};
  EXPECT_TRUE(is_near(v4a, m1936));
  EXPECT_TRUE(is_near(v4, m1936));
  EXPECT_TRUE(is_near(v4ac, m1936));
  EXPECT_TRUE(is_near(v4c, m1936));
  Mat3 m2025 {20.25, 11.25, 6.75, 11.25, 36.5, -15.5, 6.75, -15.5, 56.75};
  v4a = V3 {m2025};
  EXPECT_TRUE(is_near(v1, Mat3 {4.5, 0, 0, 2.5, 5.5, 0, 1.5, -3.5, 6.5}));
  EXPECT_TRUE(is_near(v4, m2025));
  EXPECT_TRUE(is_near(v4c, m2025));
}


TEST(covariance_tests, References_4from1_SqCov_l_1)
{
  using V = Covariance<C, SAl<M>>; // case 1
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  using V3 = SquareRootCovariance<C, SAl<M>>;
  auto v4 = v1.square_root(); // case 1 -> case 4
  EXPECT_TRUE(is_near(v4, Mat3 {4, 0, 0, 2, 5, 0, 1, -3, 6}));
  SquareRootCovariance<C, const SAl<M>&> v4c = v1.square_root(); // case 1 -> case 4 -> case 4 (const)
  EXPECT_TRUE(is_near(v4c, Mat3 {4, 0, 0, 2, 5, 0, 1, -3, 6}));
  EXPECT_EQ(v4(1,0), 2);
  EXPECT_EQ(v4c(1,0), 2);
  v1(2,2) = 47.21;
  EXPECT_EQ(v4(2,2), 6.1);
  EXPECT_EQ(v4c(2,2), 6.1);
  v4(1,1) = 5.1;
  v4(2,1) = -3.1;
  EXPECT_TRUE(is_near(v1, Mat3 {16, 8, 4, 8, 30.01, -13.81, 4, -13.81, 47.82}));
  v1 = {17.64, 9.24, 5.04,
        9.24, 31.88, -14,
        5.04, -14, 50.12};
  EXPECT_TRUE(is_near(v4, Mat3 {4.2, 0, 0, 2.2, 5.2, 0, 1.2, -3.2, 6.2}));
  EXPECT_TRUE(is_near(v4c, Mat3 {4.2, 0, 0, 2.2, 5.2, 0, 1.2, -3.2, 6.2}));
  Mat3 m43 {4.3, 0, 0, 2.3, 5.3, 0, 1.3, -3.3, 6.3};
  v4 = V3 {m43};
  EXPECT_TRUE(is_near(v1, Mat3 {18.49, 9.89, 5.59, 9.89, 33.38, -14.5, 5.59, -14.5, 52.27}));
  EXPECT_TRUE(is_near(adjoint(v1.square_root()), adjoint(v4)));

  // Case 1 from Case 4
  V v1a = v4.square(); // case 1
  EXPECT_TRUE(is_near(v1a, v1));
  Covariance<C, SAl<const M>> v1ac = v4.square(); // case 1
  EXPECT_TRUE(is_near(v1ac, v1));

  // Case 4 from Case 4
  SquareRootCovariance<C, SAl<M>&> v4a = v4;
  EXPECT_TRUE(is_near(v4a, m43));
  SquareRootCovariance<C, const SAl<M>&> v4ac = v4;
  EXPECT_TRUE(is_near(v4ac, m43));
  v1 = {19.36, 10.56, 6.16,
        10.56, 34.92, -15,
        6.16, -15, 54.48};
  Mat3 m44 {4.4, 0, 0, 2.4, 5.4, 0, 1.4, -3.4, 6.4};
  EXPECT_TRUE(is_near(v4a, m44));
  EXPECT_TRUE(is_near(v4ac, m44));
  Mat3 m45 {4.5, 0, 0, 2.5, 5.5, 0, 1.5, -3.5, 6.5};
  v4a = V3 {m45};
  EXPECT_EQ(v1(2,1), -15.5);
  EXPECT_TRUE(is_near(v4a, m45));
}


// -----------Case 4 from Case 2----------- //

TEST(covariance_tests, References_4from2_Cov_l_1)
{
  using V1 = SquareRootCovariance<C, Tl<M>>; // case 1
  V1 v1 {4, 0, 0,
         2, 5, 0,
         1, -3, 6};
  SquareRootCovariance<C, Tl<M>&> v2 = v1; // case 2
  using V3 = Covariance<C, Tl<M>>;
  Covariance<C, Tl<M>&> v4 = v2.square(); // case 4
  EXPECT_TRUE(is_near(v4, Mat3 {16, 8, 4, 8, 29, -13, 4, -13, 46}));
  Covariance<C, const Tl<M>&> v4c = v2.square(); // case 4
  EXPECT_TRUE(is_near(v4c, Mat3 {16, 8, 4, 8, 29, -13, 4, -13, 46}));
  EXPECT_EQ(v4(1,0), 8);
  EXPECT_EQ(v4c(1,0), 8);
  v2(2,2) = 6.1;
  EXPECT_NEAR(v4(2,2), 47.21, 1e-6);
  EXPECT_NEAR(v4c(2,2), 47.21, 1e-6);
  v4(2,1) = -13.5;
  EXPECT_NEAR(v2(2,1), -3.1, 1e-6);
  v2 = V1 {4.2, 0, 0,
           2.2, 5.2, 0,
           1.2, -3.2, 6.2};
  Mat3 m1764 {17.64, 9.24, 5.04, 9.24, 31.88, -14, 5.04, -14, 50.12};
  EXPECT_TRUE(is_near(v4, m1764));
  EXPECT_TRUE(is_near(v4c, m1764));
  Mat3 m1849 {18.49, 9.89, 5.59, 9.89, 33.38, -14.5, 5.59, -14.5, 52.27};
  v4 = V3 {m1849};
  EXPECT_TRUE(is_near(v2, Mat3 {4.3, 0, 0, 2.3, 5.3, 0, 1.3, -3.3, 6.3}));
  EXPECT_TRUE(is_near(adjoint(v2.square()), v2.square()));

  // Case 2 from Case 4
  SquareRootCovariance<C, Tl<M>&> v2a {v4.square_root()}; // case 4 & -> case 2
  EXPECT_TRUE(is_near(v2a, v2));
  SquareRootCovariance<C, const Tl<M>&> v2ac = v4c.square_root(); // const case 4 & -> const case 4 && -> const case 2
  EXPECT_TRUE(is_near(v2ac, v2));
  EXPECT_TRUE(is_near(adjoint(v2a), adjoint(v4.square_root())));

  // Case 4 from Case 4
  Covariance<C, Tl<M>&> v4a = v4;
  EXPECT_TRUE(is_near(v4a, m1849));
  Covariance<C, const Tl<M>&> v4ac = v4;
  EXPECT_TRUE(is_near(v4ac, m1849));
  v2 = V1 {4.4, 0, 0,
          2.4, 5.4, 0,
          1.4, -3.4, 6.4};
  Mat3 m1936 {19.36, 10.56, 6.16, 10.56, 34.92, -15, 6.16, -15, 54.48};
  EXPECT_TRUE(is_near(v4a, m1936));
  EXPECT_TRUE(is_near(v4ac, m1936));
  v4a = V3 {20.25, 11.25, 6.75,
           11.25, 36.5, -15.5,
           6.75, -15.5, 56.75};
  EXPECT_TRUE(is_near(v2, Mat3 {4.5, 0, 0, 2.5, 5.5, 0, 1.5, -3.5, 6.5}));
}


// -----------Case 4 from nestable----------- //

TEST(covariance_tests, References_4from_nestable_Cov_l_1)
{
  using V1 = Tl<M>;
  V1 v1 {4, 0, 0,
        2, 5, 0,
        1, -3, 6};
  Mat3 m16 = Mat3 {16, 8, 4, 8, 29, -13, 4, -13, 46};
  using V3 = Covariance<C, Tl<M>>;
  Covariance<C, Tl<M>&> v4 {v1};
  EXPECT_TRUE(is_near(v4, m16));
  Covariance<C, const Tl<M>&> v4c {v1};
  EXPECT_TRUE(is_near(v4c, m16));
  EXPECT_NEAR(v4(1,0), 8, 1e-6);
  EXPECT_NEAR(v4c(1,0), 8, 1e-6);
  v1(2,2) = 6.1;
  EXPECT_NEAR(v4(2,2), 47.21, 1e-6);
  EXPECT_NEAR(v4c(2,2), 47.21, 1e-6);
  v4(2,1) = -13.5;
  v4(2,2) = 47.82;
  EXPECT_NEAR(v1(2,1), -3.1, 1e-6);
  EXPECT_TRUE(is_near(v4, Mat3 {16, 8, 4, 8, 29, -13.5, 4, -13.5, 47.82}));
  v1 = {4.2, 0, 0,
        2.2, 5.2, 0,
        1.2, -3.2, 6.2};
  Mat3 m1764 = Mat3 {17.64, 9.24, 5.04, 9.24, 31.88, -14, 5.04, -14, 50.12};
  EXPECT_TRUE(is_near(v4, m1764));
  EXPECT_TRUE(is_near(v4c, m1764));
  v4 = V3 {18.49, 9.89, 5.59,
           9.89, 33.38, -14.5,
           5.59, -14.5, 52.27};
  EXPECT_TRUE(is_near(v1, Mat3 {4.3, 0, 0, 2.3, 5.3, 0, 1.3, -3.3, 6.3}));

  // Case 1 from Case 4
  SquareRootCovariance<C, Tl<M>> v1a = v4.square_root(); // case 1
  EXPECT_TRUE(is_near(v1a, v1));
  SquareRootCovariance<C, Tl<const M>> v1ac = v4.square_root(); // case 1
  EXPECT_TRUE(is_near(v1ac, v1));

  // Case 4 from Case 4
  Mat3 m1849 {18.49, 9.89, 5.59, 9.89, 33.38, -14.5, 5.59, -14.5, 52.27};
  Covariance<C, Tl<M>&> v4a = v4; // case 4
  EXPECT_TRUE(is_near(v4a, m1849));
  Covariance<C, const Tl<M>&> v4ac = v4; // case 4
  EXPECT_TRUE(is_near(v4ac, m1849));
  v1 = {4.4, 0, 0,
        2.4, 5.4, 0,
        1.4, -3.4, 6.4};
  Mat3 m1936 {19.36, 10.56, 6.16, 10.56, 34.92, -15, 6.16, -15, 54.48};
  EXPECT_TRUE(is_near(v4a, m1936));
  EXPECT_TRUE(is_near(v4, m1936));
  EXPECT_TRUE(is_near(v4ac, m1936));
  EXPECT_TRUE(is_near(v4c, m1936));
  Mat3 m2025 {20.25, 11.25, 6.75, 11.25, 36.5, -15.5, 6.75, -15.5, 56.75};
  v4a = V3 {m2025};
  EXPECT_TRUE(is_near(v1, Mat3 {4.5, 0, 0, 2.5, 5.5, 0, 1.5, -3.5, 6.5}));
  EXPECT_TRUE(is_near(v4, m2025));
  EXPECT_TRUE(is_near(v4c, m2025));
}


TEST(covariance_tests, References_4from_nestable_SqCov_l_2)
{
  using V = SAl<M>;
  V v1 {16, 8, 4,
        8, 29, -13,
        4, -13, 46};
  using V3 = SquareRootCovariance<C, SAl<M>>;
  SquareRootCovariance<C, SAl<M&>> v4 {v1};
  EXPECT_TRUE(is_near(v4, Mat3 {4, 0, 0, 2, 5, 0, 1, -3, 6}));
  SquareRootCovariance<C, const SAl<M&>> v4c {v1};
  EXPECT_TRUE(is_near(v4c, Mat3 {4, 0, 0, 2, 5, 0, 1, -3, 6}));
  EXPECT_EQ(v4(1,0), 2);
  EXPECT_EQ(v4c(1,0), 2);
  v1(2,2) = 47.21;
  EXPECT_EQ(v4(2,2), 6.1);
  EXPECT_EQ(v4c(2,2), 6.1);
  v4(1,1) = 5.1;
  v4(2,1) = -3.1;
  EXPECT_TRUE(is_near(v1, Mat3 {16, 8, 4, 8, 30.01, -13.81, 4, -13.81, 47.82}));
  v1 = {17.64, 9.24, 5.04,
        9.24, 31.88, -14,
        5.04, -14, 50.12};
  EXPECT_TRUE(is_near(v4, Mat3 {4.2, 0, 0, 2.2, 5.2, 0, 1.2, -3.2, 6.2}));
  EXPECT_TRUE(is_near(v4c, Mat3 {4.2, 0, 0, 2.2, 5.2, 0, 1.2, -3.2, 6.2}));
  Mat3 m43 {4.3, 0, 0, 2.3, 5.3, 0, 1.3, -3.3, 6.3};
  v4 = V3 {m43};
  EXPECT_TRUE(is_near(v1, Mat3 {18.49, 9.89, 5.59, 9.89, 33.38, -14.5, 5.59, -14.5, 52.27}));

  // Case 1 from Case 4
  Covariance<C, SAl<M>> v1a = v4.square(); // case 1
  EXPECT_TRUE(is_near(v1a, v1));
  Covariance<C, SAl<const M>> v1ac = v4.square(); // case 1
  EXPECT_TRUE(is_near(v1ac, v1));

  // Case 4 from Case 4
  SquareRootCovariance<C, SAl<M&>> v4a = v4;
  EXPECT_TRUE(is_near(v4a, m43));
  SquareRootCovariance<C, const SAl<M&>> v4ac = v4;
  EXPECT_TRUE(is_near(v4ac, m43));
  v1 = {19.36, 10.56, 6.16,
        10.56, 34.92, -15,
        6.16, -15, 54.48};
  Mat3 m44 {4.4, 0, 0, 2.4, 5.4, 0, 1.4, -3.4, 6.4};
  EXPECT_TRUE(is_near(v4a, m44));
  EXPECT_TRUE(is_near(v4ac, m44));
  Mat3 m45 {4.5, 0, 0, 2.5, 5.5, 0, 1.5, -3.5, 6.5};
  v4a = V3 {m45};
  EXPECT_EQ(v1(2,1), -15.5);
  EXPECT_TRUE(is_near(v4a, m45));
}
