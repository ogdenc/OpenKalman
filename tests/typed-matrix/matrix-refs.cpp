/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "matrices.hpp"

using namespace OpenKalman;

using M12 = Eigen::Matrix<double, 1, 2>;
using M21 = Eigen::Matrix<double, 2, 1>;
using M22 = Eigen::Matrix<double, 2, 2>;
using M23 = Eigen::Matrix<double, 2, 3>;
using M32 = Eigen::Matrix<double, 3, 2>;
using M33 = Eigen::Matrix<double, 3, 3>;
using I22 = IdentityMatrix<M22>;
using Z22 = ZeroMatrix<M22>;
using C2 = Coefficients<Axis, Angle>;
using C3 = Coefficients<Axis, Angle, Axis>;


TEST_F(matrices, References_TypedMatrix_lvalue)
{
  using V = Matrix<C3, C3, M33>;
  V v1 {1, 2, 3,
        4, 5, 6,
        7, 8, 9};
  Matrix<C3, C3, M33&> v2 = v1;
  EXPECT_EQ(&base_matrix(v1), &base_matrix(v2));
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1, 0), 4);
  v1(1, 0) = 4.1;
  EXPECT_EQ(v2(1, 0), 4.1);
  v2(2, 1) = 8.1;
  EXPECT_EQ(v1(2, 1), 8.1);
  v1 = {1.2, 2.2, 3.2,
        4.2, 5.2, 6.2,
        7.2, 8.2, 9.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {1.3, 2.3, 3.3,
          4.3, 5.3, 6.3,
          7.3, 8.3, 9.3};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(matrices, References_TypedMatrix_const_lvalue)
{
  Matrix<C3, C3, M33> v1 {1, 2, 3,
                               4, 5, 6,
                               7, 8, 9};
  Matrix<C3, C3, const M33&> v2 = v1;
  EXPECT_EQ(&base_matrix(v1), &base_matrix(v2));
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1, 0), 4);
  v1(1, 0) = 4.1;
  EXPECT_EQ(v2(1, 0), 4.1);
  v1 = {1.2, 2.2, 3.2,
        4.2, 5.2, 6.2,
        7.2, 8.2, 9.2};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(matrices, References_TypedMatrix_rvalue)
{
  using V = Matrix<C3, C3, M33>;
  V m1 {1, 2, 3,
        4, 5, 6,
        7, 8, 9};
  Matrix<C3, C3, M33> v1 = m1;
  Matrix<C3, C3, M33&&> v2 = std::move(v1);
  EXPECT_TRUE(is_near(v2, m1));
  EXPECT_EQ(v2(1,0), 4);
}

TEST_F(matrices, References_Mean_lvalue)
{
  using V = Mean<C3, M33>;
  V v1 {1, 2, 3,
        4, 5, 6,
        7, 8, 9};
  EXPECT_EQ(v1(1, 0), 4 - 2*pi);
  Mean<C3, M33&> v2 = v1;
  EXPECT_EQ(&base_matrix(v1), &base_matrix(v2));
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1, 0), 4 - 2*pi);
  v1(1, 0) = 4.1;
  EXPECT_EQ(v2(1, 0), 4.1 - 2*pi);
  v2(2, 1) = 8.1;
  EXPECT_EQ(v1(2, 1), 8.1);
  v1 = {1.2, 2.2, 3.2,
        4.2, 5.2, 6.2,
        7.2, 8.2, 9.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {1.3, 2.3, 3.3,
          4.3, 5.3, 6.3,
          7.3, 8.3, 9.3};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(matrices, References_Mean_const_lvalue)
{
  Mean<C3, M33> v1 {1, 2, 3,
                    4, 5, 6,
                    7, 8, 9};
  EXPECT_EQ(v1(1, 0), 4 - 2*pi);
  Mean<C3, const M33&> v2 = v1;
  EXPECT_EQ(&base_matrix(v1), &base_matrix(v2));
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_TRUE(is_near(base_matrix(v1), base_matrix(v2)));
  EXPECT_EQ(get_element(base_matrix(v2), 1, 0), 4 - 2*pi);
  EXPECT_EQ(get_element(v2, 1, 0), 4 - 2*pi);
  EXPECT_EQ(get_element(base_matrix(v2), 1, 0), 4 - 2*pi);
  EXPECT_TRUE(is_near(base_matrix(v1), base_matrix(v2)));
  v1(1, 0) = 4.1;
  EXPECT_EQ(v2(1, 0), 4.1 - 2*pi);
  EXPECT_EQ(base_matrix(v2)(1, 0), 4.1 - 2*pi);
  v1 = {1.2, 2.2, 3.2,
        4.2, 5.2, 6.2,
        7.2, 8.2, 9.2};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(matrices, References_Mean_rvalue)
{
  using V = Mean<C3, M33>;
  V m1 {1, 2, 3,
        4, 5, 6,
        7, 8, 9};
  Mean<C3, M33> v1 = m1;
  EXPECT_EQ(v1(1,0), 4 - 2*pi);
  Mean<C3, M33&&> v2 = std::move(v1);
  EXPECT_TRUE(is_near(v2, m1));
  EXPECT_EQ(v2(1,0), 4 - 2*pi);
}

TEST_F(matrices, References_Mean_lvalue_axes)
{
  using V = Mean<Axes<3>, M33>;
  V v1 {1, 2, 3,
        4, 5, 6,
        7, 8, 9};
  Mean<Axes<3>, M33&> v2 = v1;
  EXPECT_EQ(&base_matrix(v1), &base_matrix(v2));
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1, 0), 4);
  v1(1, 0) = 4.1;
  EXPECT_EQ(v2(1, 0), 4.1);
  v2(2, 1) = 8.1;
  EXPECT_EQ(v1(2, 1), 8.1);
  v1 = {1.2, 2.2, 3.2,
        4.2, 5.2, 6.2,
        7.2, 8.2, 9.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {1.3, 2.3, 3.3,
          4.3, 5.3, 6.3,
          7.3, 8.3, 9.3};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(matrices, References_Mean_const_lvalue_axes)
{
  Mean<Axes<3>, M33> v1 {1, 2, 3,
                    4, 5, 6,
                    7, 8, 9};
  Mean<Axes<3>, const M33&> v2 = v1;
  EXPECT_EQ(&base_matrix(v1), &base_matrix(v2));
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1, 0), 4);
  v1(1, 0) = 4.1;
  EXPECT_EQ(v2(1, 0), 4.1);
  v1 = {1.2, 2.2, 3.2,
        4.2, 5.2, 6.2,
        7.2, 8.2, 9.2};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(matrices, References_Mean_rvalue_axes)
{
  using V = Mean<Axes<3>, M33>;
  V m1 {1, 2, 3,
        4, 5, 6,
        7, 8, 9};
  Mean<Axes<3>, M33> v1 = m1;
  Mean<Axes<3>, M33&&> v2 = std::move(v1);
  EXPECT_TRUE(is_near(v2, m1));
  EXPECT_EQ(v2(1,0), 4);
}

TEST_F(matrices, References_EuclideanMean_lvalue)
{
  using V = EuclideanMean<C2, M33>;
  V v1 {1, 2, 3,
        4, 5, 6,
        7, 8, 9};
  EuclideanMean<C2, M33&> v2 = v1;
  EXPECT_EQ(&base_matrix(v1), &base_matrix(v2));
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1, 0), 4);
  v1(1, 0) = 4.1;
  EXPECT_EQ(v2(1, 0), 4.1);
  v2(2, 1) = 8.1;
  EXPECT_EQ(v1(2, 1), 8.1);
  v1 = {1.2, 2.2, 3.2,
        4.2, 5.2, 6.2,
        7.2, 8.2, 9.2};
  EXPECT_TRUE(is_near(v1, v2));
  v2 = V {1.3, 2.3, 3.3,
          4.3, 5.3, 6.3,
          7.3, 8.3, 9.3};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(matrices, References_EuclideanMean_const_lvalue)
{
  EuclideanMean<C2, M33> v1 {1, 2, 3,
                             4, 5, 6,
                             7, 8, 9};
  EuclideanMean<C2, const M33&> v2 = v1;
  EXPECT_EQ(&base_matrix(v1), &base_matrix(v2));
  EXPECT_TRUE(is_near(v1, v2));
  EXPECT_EQ(v2(1, 0), 4);
  v1(1, 0) = 4.1;
  EXPECT_EQ(v2(1, 0), 4.1);
  v1 = {1.2, 2.2, 3.2,
        4.2, 5.2, 6.2,
        7.2, 8.2, 9.2};
  EXPECT_TRUE(is_near(v1, v2));
}

TEST_F(matrices, References_EuclideanMean_rvalue)
{
  using V = EuclideanMean<C2, M33>;
  V m1 {1, 2, 3,
        4, 5, 6,
        7, 8, 9};
  EuclideanMean<C2, M33> v1 = m1;
  EuclideanMean<C2, M33&&> v2 = std::move(v1);
  EXPECT_TRUE(is_near(v2, m1));
  EXPECT_EQ(v2(1,0), 4);
}

