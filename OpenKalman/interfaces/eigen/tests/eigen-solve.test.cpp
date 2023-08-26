/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, solve_constant_diagonal_A)
{
  auto cd22 = M22::Identity() + M22::Identity();
  auto cd20_2 = Eigen::Replicate<decltype(cd22), 1, Eigen::Dynamic> {cd22, 1, 1};
  auto cd02_2 = Eigen::Replicate<decltype(cd22), Eigen::Dynamic, 1> {cd22, 1, 1};
  auto cd00_22 = Eigen::Replicate<decltype(cd22), Eigen::Dynamic, Eigen::Dynamic> {cd22, 1, 1};

  auto m23_x = make_eigen_matrix<double, 2, 3>(5, 7, 9, 6, 8, 10);

  auto m23 = make_eigen_matrix<double, 2, 3>(10, 14, 18, 12, 16, 20);
  auto m2x_3 = M2x {m23};
  auto mx3_2 = Mx3 {m23};
  auto mxx_23 = Mxx {m23};

  EXPECT_TRUE(is_near(solve(M22::Identity(), m23), m23));

  EXPECT_TRUE(is_near(solve(cd22, m23), m23_x));
  EXPECT_TRUE(is_near(solve(cd22, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(cd22, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(cd22, mxx_23), m23_x));
  EXPECT_TRUE(is_near(solve(cd20_2, m23), m23_x));
  EXPECT_TRUE(is_near(solve(cd20_2, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(cd20_2, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(cd20_2, mxx_23), m23_x));
  EXPECT_TRUE(is_near(solve(cd02_2, m23), m23_x));
  EXPECT_TRUE(is_near(solve(cd02_2, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(cd02_2, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(cd02_2, mxx_23), m23_x));
  EXPECT_TRUE(is_near(solve(cd00_22, m23), m23_x));
  EXPECT_TRUE(is_near(solve(cd00_22, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(cd00_22, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(cd00_22, mxx_23), m23_x));
}


TEST(eigen3, solve_constant_A)
{
  auto c11_2 = M11::Identity() + M11::Identity();

  auto m12_1 = make_dense_writable_matrix_from<M12>(1, 1);

  auto m12_2 = make_dense_writable_matrix_from<M12>(2, 2);
  auto m10_2_2 = M1x {m12_2};
  auto m02_1_2 = Mx2 {m12_2};
  auto m00_12_2 = Mxx {m12_2};

  EXPECT_TRUE(is_near(solve(c11_2, m12_2), m12_1));
  EXPECT_TRUE(is_near(solve(c11_2, m10_2_2), m12_1));
  EXPECT_TRUE(is_near(solve(c11_2, m02_1_2), m12_1));
  EXPECT_TRUE(is_near(solve(c11_2, m00_12_2), m12_1));

  auto c22 = Eigen::Replicate<decltype(c11_2), 2, 2> {c11_2, 2, 2};
  auto c20_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic> {c11_2, 2, 2};
  auto c02_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2> {c11_2, 2, 2};
  auto c00_22 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic> {c11_2, 2, 2};

  auto c11_8 = c11_2 + c11_2 + c11_2 + c11_2;

  auto c23 = Eigen::Replicate<decltype(c11_8), 2, 3> {c11_8, 2, 3};
  auto c20_3 = Eigen::Replicate<decltype(c11_8), 2, Eigen::Dynamic> {c11_8, 2, 3};
  auto c03_2 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, 3> {c11_8, 2, 3};
  auto c00_23 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, Eigen::Dynamic> {c11_8, 2, 3};

  auto m23_x = make_eigen_matrix<double, 2, 3>(6, 9, 12, 6, 9, 12);

  auto m23 = make_eigen_matrix<double, 2, 3>(24, 36, 48, 24, 36, 48);
  auto m2x_3 = M2x {m23};
  auto mx3_2 = Mx3 {m23};
  auto mxx_23 = Mxx {m23};

  EXPECT_TRUE(is_near(solve(c22, m23), m23_x));
  EXPECT_TRUE(is_near(solve(c22, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(c22, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(c22, mxx_23), m23_x));
  EXPECT_TRUE(is_near(solve(c20_2, m23), m23_x));
  EXPECT_TRUE(is_near(solve(c20_2, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(c20_2, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(c20_2, mxx_23), m23_x));
  EXPECT_TRUE(is_near(solve(c02_2, m23), m23_x));
  EXPECT_TRUE(is_near(solve(c02_2, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(c02_2, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(c02_2, mxx_23), m23_x));
  EXPECT_TRUE(is_near(solve(c00_22, m23), m23_x));
  EXPECT_TRUE(is_near(solve(c00_22, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(c00_22, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(c00_22, mxx_23), m23_x));

  auto m33_x = make_eigen_matrix<double, 1, 3>(3, 4.5, 6);

  EXPECT_TRUE(is_near(solve<false, true>(c23, m23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c23, m2x_3).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c23, mx3_2).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c23, mxx_23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c20_3, m23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c20_3, m2x_3).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c20_3, mx3_2).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c20_3, mxx_23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c03_2, m23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c03_2, m2x_3).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c03_2, mx3_2).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c03_2, mxx_23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c00_23, m23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c00_23, m2x_3).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c00_23, mx3_2).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c00_23, mxx_23).colwise().sum(), m33_x));
}


TEST(eigen3, solve_one_by_one)
{
  M11 m11_0; m11_0 << 0;
  M1x m10_1_0(1,1); m10_1_0 << 0;
  Mx1 m01_1_0(1,1); m01_1_0 << 0;
  Mxx m00_11_0(1,1); m00_11_0 << 0;

  M11 m11_6; m11_6 << 6;
  M1x m10_1_6(1,1); m10_1_6 << 6;
  Mx1 m01_1_6(1,1); m01_1_6 << 6;
  Mxx m00_11_6(1,1); m00_11_6 << 6;

  auto inf = std::numeric_limits<double>::infinity();

  EXPECT_EQ(trace(solve(m11_0, m11_6)), inf);
  EXPECT_EQ(trace(solve(m11_0, m10_1_6)), inf);
  EXPECT_EQ(trace(solve(m11_0, m01_1_6)), inf);
  EXPECT_EQ(trace(solve(m11_0, m00_11_6)), inf);
  EXPECT_EQ(trace(solve(m10_1_0, m11_6)), inf);
  EXPECT_EQ(trace(solve(m10_1_0, m10_1_6)), inf);
  EXPECT_EQ(trace(solve(m10_1_0, m01_1_6)), inf);
  EXPECT_EQ(trace(solve(m10_1_0, m00_11_6)), inf);
  EXPECT_EQ(trace(solve(m01_1_0, m11_6)), inf);
  EXPECT_EQ(trace(solve(m01_1_0, m10_1_6)), inf);
  EXPECT_EQ(trace(solve(m01_1_0, m01_1_6)), inf);
  EXPECT_EQ(trace(solve(m01_1_0, m00_11_6)), inf);
  EXPECT_EQ(trace(solve(m00_11_0, m11_6)), inf);
  EXPECT_EQ(trace(solve(m00_11_0, m10_1_6)), inf);
  EXPECT_EQ(trace(solve(m00_11_0, m01_1_6)), inf);
  EXPECT_EQ(trace(solve(m00_11_0, m00_11_6)), inf);

  auto m12_68 = make_dense_writable_matrix_from<M12>(6, 8);
  auto m10_2_68 = M1x {m12_68};
  auto m02_1_68 = Mx2 {m12_68};
  auto m00_12_68 = Mxx {m12_68};

  EXPECT_EQ(solve(m11_0, m12_68)(0,0), inf);
  EXPECT_EQ(solve(m10_1_0, m12_68)(0,1), inf);
  EXPECT_EQ(solve(m01_1_0, m12_68)(0,0), inf);
  EXPECT_EQ(solve(m00_11_0, m12_68)(0,1), inf);
  EXPECT_EQ(solve(m11_0, m10_2_68)(0,0), inf);
  EXPECT_EQ(solve(m10_1_0, m10_2_68)(0,1), inf);
  EXPECT_EQ(solve(m01_1_0, m10_2_68)(0,0), inf);
  EXPECT_EQ(solve(m00_11_0, m10_2_68)(0,1), inf);
  EXPECT_EQ(solve(m11_0, m02_1_68)(0,0), inf);
  EXPECT_EQ(solve(m10_1_0, m02_1_68)(0,1), inf);
  EXPECT_EQ(solve(m01_1_0, m02_1_68)(0,0), inf);
  EXPECT_EQ(solve(m00_11_0, m02_1_68)(0,1), inf);
  EXPECT_EQ(solve(m11_0, m00_12_68)(0,0), inf);
  EXPECT_EQ(solve(m10_1_0, m00_12_68)(0,1), inf);
  EXPECT_EQ(solve(m01_1_0, m00_12_68)(0,0), inf);
  EXPECT_EQ(solve(m00_11_0, m00_12_68)(0,1), inf);

  M11 m11_2; m11_2 << 2;
  M1x m10_1_2(1,1); m10_1_2 << 2;
  Mx1 m01_1_2(1,1); m01_1_2 << 2;
  Mxx m00_11_2(1,1); m00_11_2 << 2;

  M11 m11_3; m11_3 << 3;

  EXPECT_TRUE(is_near(solve(m11_2, m11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m11_2, m10_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m11_2, m01_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m11_2, m00_11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m10_1_2, m11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m10_1_2, m10_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m10_1_2, m01_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m10_1_2, m00_11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m01_1_2, m11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m01_1_2, m10_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m01_1_2, m01_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m01_1_2, m00_11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m00_11_2, m11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m00_11_2, m10_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m00_11_2, m01_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m00_11_2, m00_11_6), m11_3));

  auto m12_34 = make_dense_writable_matrix_from<M12>(3, 4);

  EXPECT_TRUE(is_near(solve(m11_2, m12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m11_2, m10_2_68), m12_34));
  EXPECT_TRUE(is_near(solve(m11_2, m02_1_68), m12_34));
  EXPECT_TRUE(is_near(solve(m11_2, m00_12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m10_1_2, m12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m10_1_2, m10_2_68), m12_34));
  EXPECT_TRUE(is_near(solve(m10_1_2, m02_1_68), m12_34));
  EXPECT_TRUE(is_near(solve(m10_1_2, m00_12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m01_1_2, m12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m01_1_2, m10_2_68), m12_34));
  EXPECT_TRUE(is_near(solve(m01_1_2, m02_1_68), m12_34));
  EXPECT_TRUE(is_near(solve(m01_1_2, m00_12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m00_11_2, m12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m00_11_2, m10_2_68), m12_34));
  EXPECT_TRUE(is_near(solve(m00_11_2, m02_1_68), m12_34));
  EXPECT_TRUE(is_near(solve(m00_11_2, m00_12_68), m12_34));
}


TEST(eigen3, solve_general_matrix)
{
  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);
  auto m2x_2 = M2x {m22};
  auto mx2_2 = Mx2 {m22};
  auto mxx_22 = Mxx {m22};

  auto m23_56 = make_eigen_matrix<double, 2, 3>(5, 7, 9, 6, 8, 10);
  auto m20_3_56 = M2x {m23_56};
  auto m03_2_56 = Mx3 {m23_56};
  auto m00_23_56 = Mxx {m23_56};

  auto m23_445 = make_eigen_matrix<double, 2, 3>(-4, -6, -8, 4.5, 6.5, 8.5);

  EXPECT_TRUE(is_near(solve(m22, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m22, m20_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(m22, m03_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(m22, m00_23_56), m23_445));
  EXPECT_TRUE(is_near(solve(mx2_2, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(mx2_2, m20_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(mx2_2, m03_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(mx2_2, m00_23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m2x_2, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m2x_2, m20_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(m2x_2, m03_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(m2x_2, m00_23_56), m23_445));
  EXPECT_TRUE(is_near(solve(mxx_22, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(mxx_22, m20_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(mxx_22, m03_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(mxx_22, m00_23_56), m23_445));
}

