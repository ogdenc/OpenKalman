/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen3.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


namespace
{
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M12 = eigen_matrix_t<double, 1, 2>;
  using M13 = eigen_matrix_t<double, 1, 3>;
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M23 = eigen_matrix_t<double, 2, 3>;
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M34 = eigen_matrix_t<double, 3, 4>;
  using M43 = eigen_matrix_t<double, 4, 3>;
  using M44 = eigen_matrix_t<double, 4, 4>;
  using M55 = eigen_matrix_t<double, 5, 5>;

  using M00 = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using M10 = eigen_matrix_t<double, 1, dynamic_size>;
  using M01 = eigen_matrix_t<double, dynamic_size, 1>;
  using M20 = eigen_matrix_t<double, 2, dynamic_size>;
  using M02 = eigen_matrix_t<double, dynamic_size, 2>;
  using M30 = eigen_matrix_t<double, 3, dynamic_size>;
  using M03 = eigen_matrix_t<double, dynamic_size, 3>;
  using M04 = eigen_matrix_t<double, dynamic_size, 4>;
  using M50 = eigen_matrix_t<double, 5, dynamic_size>;
  using M05 = eigen_matrix_t<double, dynamic_size, 5>;

  using cdouble = std::complex<double>;

  using CM21 = eigen_matrix_t<cdouble, 2, 1>;
  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;
}


TEST(eigen3, solve_zero_B)
{
  auto m22 = make_dense_writable_matrix_from<M22>(1, 2, 3, 4);
  auto m20_2 = M20 {m22};
  auto m02_2 = M02 {m22};
  auto m00_22 = M00 {m22};

  auto z11 = M11::Identity() - M11::Identity();

  auto z12 = Eigen::Replicate<decltype(z11), 1, 2> {z11, 1, 2};
  auto z10_2 = Eigen::Replicate<decltype(z11), 1, Eigen::Dynamic> {z11, 1, 2};
  auto z02_1 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 1, 2};
  auto z00_12 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 1, 2};

  auto z22 = M22::Identity() - M22::Identity();
  auto z20_2 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 2};
  auto z02_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 2, 2};
  auto z00_22 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  EXPECT_TRUE(is_near(solve<true>(m22, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve<true>(m00_22, z00_22), M22::Zero()));
  try { solve<true>(M12 {z12}, z12); EXPECT_TRUE(false); } catch (const std::runtime_error&) {}
  try { solve<true>(M00 {z12}, z00_12); EXPECT_TRUE(false); } catch (const std::runtime_error&) {}

  auto cd22 = M22::Identity() + M22::Identity();
  auto cd00_22 = Eigen::Replicate<decltype(cd22), Eigen::Dynamic, Eigen::Dynamic> {cd22, 1, 1};

  EXPECT_TRUE(is_near(solve<true>(cd22, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve<true>(cd00_22, z00_22), M22::Zero()));

  auto c11 = M11::Identity() + M11::Identity();
  auto c12 = Eigen::Replicate<decltype(c11), 1, 2> {c11, 1, 2};
  auto c00_12 = Eigen::Replicate<decltype(c11), Eigen::Dynamic, Eigen::Dynamic> {c11, 1, 2};

  EXPECT_TRUE(is_near(solve<true>(c12, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve<true>(c00_12, z00_12), M22::Zero()));

  EXPECT_TRUE(is_near(solve(m22, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m20_2, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m02_2, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m00_22, z22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m22, z20_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m20_2, z20_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m02_2, z20_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m00_22, z20_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m22, z02_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m20_2, z02_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m02_2, z02_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m00_22, z02_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m22, z00_22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m20_2, z00_22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m02_2, z00_22), M22::Zero()));
  EXPECT_TRUE(is_near(solve(m00_22, z00_22), M22::Zero()));

  EXPECT_TRUE(is_near(solve(z12, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z10_2, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z02_1, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z00_12, z12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z12, z10_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z10_2, z10_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z02_1, z10_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z00_12, z10_2), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z12, z02_1), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z10_2, z02_1), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z02_1, z02_1), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z00_12, z02_1), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z12, z00_12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z10_2, z00_12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z02_1, z00_12), M22::Zero()));
  EXPECT_TRUE(is_near(solve(z00_12, z00_12), M22::Zero()));
}


TEST(eigen3, solve_zero_A)
{
  auto z11 = M11::Identity() - M11::Identity();

  auto z22 = M22::Identity() - M22::Identity();
  auto z20_2 = Eigen::Replicate<decltype(z11), 2, Eigen::Dynamic> {z11, 2, 2};
  auto z02_2 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, 2> {z11, 2, 2};
  auto z00_22 = Eigen::Replicate<decltype(z11), Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  auto m23_56 = make_eigen_matrix<double, 2, 3>(5, 7, 9, 6, 8, 10);
  auto m20_3_56 = M20 {m23_56};
  auto m03_2_56 = M03 {m23_56};
  auto m00_23_56 = M00 {m23_56};

  EXPECT_TRUE(is_near(solve(z22, m23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z22, m20_3_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z22, m03_2_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z22, m00_23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z20_2, m23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z20_2, m20_3_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z20_2, m03_2_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z20_2, m00_23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z02_2, m23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z02_2, m20_3_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z02_2, m03_2_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z02_2, m00_23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z00_22, m23_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z00_22, m20_3_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z00_22, m03_2_56), M23::Zero()));
  EXPECT_TRUE(is_near(solve(z00_22, m00_23_56), M23::Zero()));
}


TEST(eigen3, solve_constant_diagonal_A)
{
  auto cd22 = M22::Identity() + M22::Identity();
  auto cd20_2 = Eigen::Replicate<decltype(cd22), 1, Eigen::Dynamic> {cd22, 1, 1};
  auto cd02_2 = Eigen::Replicate<decltype(cd22), Eigen::Dynamic, 1> {cd22, 1, 1};
  auto cd00_22 = Eigen::Replicate<decltype(cd22), Eigen::Dynamic, Eigen::Dynamic> {cd22, 1, 1};

  auto m23_x = make_eigen_matrix<double, 2, 3>(5, 7, 9, 6, 8, 10);

  auto m23 = make_eigen_matrix<double, 2, 3>(10, 14, 18, 12, 16, 20);
  auto m20_3 = M20 {m23};
  auto m03_2 = M03 {m23};
  auto m00_23 = M00 {m23};

  EXPECT_TRUE(is_near(solve(M22::Identity(), m23), m23));

  EXPECT_TRUE(is_near(solve(cd22, m23), m23_x));
  EXPECT_TRUE(is_near(solve(cd22, m20_3), m23_x));
  EXPECT_TRUE(is_near(solve(cd22, m03_2), m23_x));
  EXPECT_TRUE(is_near(solve(cd22, m00_23), m23_x));
  EXPECT_TRUE(is_near(solve(cd20_2, m23), m23_x));
  EXPECT_TRUE(is_near(solve(cd20_2, m20_3), m23_x));
  EXPECT_TRUE(is_near(solve(cd20_2, m03_2), m23_x));
  EXPECT_TRUE(is_near(solve(cd20_2, m00_23), m23_x));
  EXPECT_TRUE(is_near(solve(cd02_2, m23), m23_x));
  EXPECT_TRUE(is_near(solve(cd02_2, m20_3), m23_x));
  EXPECT_TRUE(is_near(solve(cd02_2, m03_2), m23_x));
  EXPECT_TRUE(is_near(solve(cd02_2, m00_23), m23_x));
  EXPECT_TRUE(is_near(solve(cd00_22, m23), m23_x));
  EXPECT_TRUE(is_near(solve(cd00_22, m20_3), m23_x));
  EXPECT_TRUE(is_near(solve(cd00_22, m03_2), m23_x));
  EXPECT_TRUE(is_near(solve(cd00_22, m00_23), m23_x));
}


TEST(eigen3, solve_constant_A)
{
  auto c11_2 = M11::Identity() + M11::Identity();

  auto m12_1 = make_dense_writable_matrix_from<M12>(1, 1);

  auto m12_2 = make_dense_writable_matrix_from<M12>(2, 2);
  auto m10_2_2 = M10 {m12_2};
  auto m02_1_2 = M02 {m12_2};
  auto m00_12_2 = M00 {m12_2};

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

  auto m23_2 = make_dense_writable_matrix_from<M23>(2, 2, 2, 2, 2, 2);

  EXPECT_TRUE(is_near(solve(c22, c23), m23_2));
  EXPECT_TRUE(is_near(solve(c22, c20_3), m23_2));
  EXPECT_TRUE(is_near(solve(c22, c03_2), m23_2));
  EXPECT_TRUE(is_near(solve(c22, c00_23), m23_2));
  EXPECT_TRUE(is_near(solve(c20_2, c23), m23_2));
  EXPECT_TRUE(is_near(solve(c20_2, c20_3), m23_2));
  EXPECT_TRUE(is_near(solve(c20_2, c03_2), m23_2));
  EXPECT_TRUE(is_near(solve(c20_2, c00_23), m23_2));
  EXPECT_TRUE(is_near(solve(c02_2, c23), m23_2));
  EXPECT_TRUE(is_near(solve(c02_2, c20_3), m23_2));
  EXPECT_TRUE(is_near(solve(c02_2, c03_2), m23_2));
  EXPECT_TRUE(is_near(solve(c02_2, c00_23), m23_2));
  EXPECT_TRUE(is_near(solve(c00_22, c23), m23_2));
  EXPECT_TRUE(is_near(solve(c00_22, c20_3), m23_2));
  EXPECT_TRUE(is_near(solve(c00_22, c03_2), m23_2));
  EXPECT_TRUE(is_near(solve(c00_22, c00_23), m23_2));

  static_assert(constant_matrix<decltype(solve(c22, c23))>);
  static_assert(not constant_matrix<decltype(solve(c20_2, c23))>);
  static_assert(constant_matrix<decltype(solve(c02_2, c23))>);
  static_assert(not constant_matrix<decltype(solve(c00_22, c23))>);

  auto c12_2 = Eigen::Replicate<decltype(c11_2), 1, 2> {c11_2, 1, 2};
  auto c10_2_2 = Eigen::Replicate<decltype(c11_2), 1, Eigen::Dynamic> {c11_2, 1, 2};
  auto c02_1_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2> {c11_2, 1, 2};
  auto c00_12_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic> {c11_2, 1, 2};

  auto c12_8 = Eigen::Replicate<decltype(c11_8), 1, 2> {c11_8, 1, 2};
  auto c10_2_8 = Eigen::Replicate<decltype(c11_8), 1, Eigen::Dynamic> {c11_8, 1, 2};
  auto c02_1_8 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, 2> {c11_8, 1, 2};
  auto c00_12_8 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, Eigen::Dynamic> {c11_8, 1, 2};

  auto m22_2 = make_dense_writable_matrix_from<M22>(2, 2, 2, 2);

  EXPECT_TRUE(is_near(solve(c12_2, c12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c12_2, c10_2_8), m22_2));
  EXPECT_TRUE(is_near(solve(c12_2, c02_1_8), m22_2));
  EXPECT_TRUE(is_near(solve(c12_2, c00_12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c10_2_2, c12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c10_2_2, c10_2_8), m22_2));
  EXPECT_TRUE(is_near(solve(c10_2_2, c02_1_8), m22_2));
  EXPECT_TRUE(is_near(solve(c10_2_2, c00_12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c02_1_2, c12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c02_1_2, c10_2_8), m22_2));
  EXPECT_TRUE(is_near(solve(c02_1_2, c02_1_8), m22_2));
  EXPECT_TRUE(is_near(solve(c02_1_2, c00_12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c00_12_2, c12_8), m22_2));
  EXPECT_TRUE(is_near(solve(c00_12_2, c10_2_8), m22_2));
  EXPECT_TRUE(is_near(solve(c00_12_2, c02_1_8), m22_2));
  EXPECT_TRUE(is_near(solve(c00_12_2, c00_12_8), m22_2));

  auto m23_x = make_eigen_matrix<double, 2, 3>(6, 9, 12, 6, 9, 12);

  auto m23 = make_eigen_matrix<double, 2, 3>(24, 36, 48, 24, 36, 48);
  auto m20_3 = M20 {m23};
  auto m03_2 = M03 {m23};
  auto m00_23 = M00 {m23};

  EXPECT_TRUE(is_near(solve(c22, m23), m23_x));
  EXPECT_TRUE(is_near(solve(c22, m20_3), m23_x));
  EXPECT_TRUE(is_near(solve(c22, m03_2), m23_x));
  EXPECT_TRUE(is_near(solve(c22, m00_23), m23_x));
  EXPECT_TRUE(is_near(solve(c20_2, m23), m23_x));
  EXPECT_TRUE(is_near(solve(c20_2, m20_3), m23_x));
  EXPECT_TRUE(is_near(solve(c20_2, m03_2), m23_x));
  EXPECT_TRUE(is_near(solve(c20_2, m00_23), m23_x));
  EXPECT_TRUE(is_near(solve(c02_2, m23), m23_x));
  EXPECT_TRUE(is_near(solve(c02_2, m20_3), m23_x));
  EXPECT_TRUE(is_near(solve(c02_2, m03_2), m23_x));
  EXPECT_TRUE(is_near(solve(c02_2, m00_23), m23_x));
  EXPECT_TRUE(is_near(solve(c00_22, m23), m23_x));
  EXPECT_TRUE(is_near(solve(c00_22, m20_3), m23_x));
  EXPECT_TRUE(is_near(solve(c00_22, m03_2), m23_x));
  EXPECT_TRUE(is_near(solve(c00_22, m00_23), m23_x));

  auto m33_x = make_eigen_matrix<double, 1, 3>(3, 4.5, 6);

  EXPECT_TRUE(is_near(solve<false, true>(c23, m23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c23, m20_3).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c23, m03_2).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c23, m00_23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c20_3, m23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c20_3, m20_3).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c20_3, m03_2).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c20_3, m00_23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c03_2, m23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c03_2, m20_3).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c03_2, m03_2).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c03_2, m00_23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c00_23, m23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c00_23, m20_3).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c00_23, m03_2).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c00_23, m00_23).colwise().sum(), m33_x));
}


TEST(eigen3, solve_one_by_one)
{
  auto m11_0 = make_dense_writable_matrix_from<M11>(0);
  auto m10_1_0 = M10 {m11_0};
  auto m01_1_0 = M01 {m11_0};
  auto m00_11_0 = M00 {m11_0};

  auto m11_6 = make_dense_writable_matrix_from<M11>(6);
  auto m10_1_6 = M10 {m11_6};
  auto m01_1_6 = M01 {m11_6};
  auto m00_11_6 = M00 {m11_6};

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
  auto m10_2_68 = M10 {m12_68};
  auto m02_1_68 = M02 {m12_68};
  auto m00_12_68 = M00 {m12_68};

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

  auto m11_2 = make_dense_writable_matrix_from<M11>(2);
  auto m10_1_2 = M10 {m11_2};
  auto m01_1_2 = M01 {m11_2};
  auto m00_11_2 = M00 {m11_2};

  auto m11_3 = make_dense_writable_matrix_from<M11>(3);

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
  auto m20_2 = M20 {m22};
  auto m02_2 = M02 {m22};
  auto m00_22 = M00 {m22};

  auto m23_56 = make_eigen_matrix<double, 2, 3>(5, 7, 9, 6, 8, 10);
  auto m20_3_56 = M20 {m23_56};
  auto m03_2_56 = M03 {m23_56};
  auto m00_23_56 = M00 {m23_56};

  auto m23_445 = make_eigen_matrix<double, 2, 3>(-4, -6, -8, 4.5, 6.5, 8.5);

  EXPECT_TRUE(is_near(solve(m22, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m22, m20_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(m22, m03_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(m22, m00_23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m02_2, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m02_2, m20_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(m02_2, m03_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(m02_2, m00_23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m20_2, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m20_2, m20_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(m20_2, m03_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(m20_2, m00_23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m00_22, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m00_22, m20_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(m00_22, m03_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(m00_22, m00_23_56), m23_445));
}


TEST(eigen3, LQ_and_QR_decompositions)
{
  auto m22_lq = make_dense_writable_matrix_from<M22>(-0.1, 0, 1.096, -1.272);

  auto m22_lq_decomp = make_dense_writable_matrix_from<M22>(0.06, 0.08, 0.36, -1.640);
  auto m20_2_lq_decomp = M20 {m22_lq_decomp};
  auto m02_2_lq_decomp = M02 {m22_lq_decomp};
  auto m00_22_lq_decomp = M00 {m22_lq_decomp};

  EXPECT_TRUE(is_near(LQ_decomposition(m22_lq_decomp), m22_lq));
  EXPECT_TRUE(is_near(LQ_decomposition(m20_2_lq_decomp), m22_lq));
  EXPECT_TRUE(is_near(LQ_decomposition(m02_2_lq_decomp), m22_lq));
  EXPECT_TRUE(is_near(LQ_decomposition(m00_22_lq_decomp), m22_lq));

  auto m22_qr = make_dense_writable_matrix_from<M22>(-0.1, 1.096, 0, -1.272);

  auto m22_qr_decomp = make_dense_writable_matrix_from<M22>(0.06, 0.36, 0.08, -1.640);
  auto m20_2_qr_decomp = M20 {m22_qr_decomp};
  auto m02_2_qr_decomp = M02 {m22_qr_decomp};
  auto m00_22_qr_decomp = M00 {m22_qr_decomp};

  EXPECT_TRUE(is_near(QR_decomposition(m22_qr_decomp), m22_qr));
  EXPECT_TRUE(is_near(QR_decomposition(m02_2_qr_decomp), m22_qr));
  EXPECT_TRUE(is_near(QR_decomposition(m20_2_qr_decomp), m22_qr));
  EXPECT_TRUE(is_near(QR_decomposition(m00_22_qr_decomp), m22_qr));

  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  auto m32 = make_dense_writable_matrix_from<M32>(1, 4, 2, 5, 3, 6);

  EXPECT_TRUE(is_near(LQ_decomposition(m23), adjoint(QR_decomposition(m32))));
  EXPECT_TRUE(is_near(LQ_decomposition(m32), adjoint(QR_decomposition(m23))));

  auto cm23 = make_dense_writable_matrix_from<CM23>(cdouble {1,6}, cdouble {2,5}, cdouble {3,4}, cdouble {4,3}, cdouble {5,2}, cdouble {6,1});
  auto cm32conj = make_dense_writable_matrix_from<CM32>(cdouble {1,-6}, cdouble {4,-3}, cdouble {2,-5}, cdouble {5,-2}, cdouble {3,-4}, cdouble {6,-1});

  EXPECT_TRUE(is_near(LQ_decomposition(cm23), adjoint(QR_decomposition(cm32conj))));
  EXPECT_TRUE(is_near(LQ_decomposition(cm32conj), adjoint(QR_decomposition(cm23))));
}

