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

  using M00 = eigen_matrix_t<double, dynamic_extent, dynamic_extent>;
  using M10 = eigen_matrix_t<double, 1, dynamic_extent>;
  using M01 = eigen_matrix_t<double, dynamic_extent, 1>;
  using M20 = eigen_matrix_t<double, 2, dynamic_extent>;
  using M02 = eigen_matrix_t<double, dynamic_extent, 2>;
  using M30 = eigen_matrix_t<double, 3, dynamic_extent>;
  using M03 = eigen_matrix_t<double, dynamic_extent, 3>;
  using M04 = eigen_matrix_t<double, dynamic_extent, 4>;
  using M50 = eigen_matrix_t<double, 5, dynamic_extent>;
  using M05 = eigen_matrix_t<double, dynamic_extent, 5>;

  using cdouble = std::complex<double>;

  using CM21 = eigen_matrix_t<cdouble, 2, 1>;
  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;
}


TEST(eigen3, solve_zero)
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

  auto m22 = make_native_matrix<M22>(1, 2, 3, 4);
  auto m20_2 = M20 {m22};
  auto m02_2 = M02 {m22};
  auto m00_22 = M00 {m22};

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
}

// Constant A is simply converted to ConstantMatrix. Thus, see testing of ConstantMatrix.

TEST(eigen3, solve_general_matrix)
{
  auto m22 = make_native_matrix<M22>(1, 2, 3, 4);
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


TEST(eigen3, solve_one_by_one)
{
  auto m11_2 = make_native_matrix<M11>(2);
  auto m10_1_2 = M10 {m11_2};
  auto m01_1_2 = M01 {m11_2};
  auto m00_11_2 = M00 {m11_2};

  auto m11_6 = make_native_matrix<M11>(6);
  auto m10_1_6 = M10 {m11_6};
  auto m01_1_6 = M01 {m11_6};
  auto m00_11_6 = M00 {m11_6};

  auto m11_3 = make_native_matrix<M11>(3);

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

  auto m12_68 = make_native_matrix<M12>(6, 8);
  auto m10_2_68 = M10 {m12_68};
  auto m02_1_68 = M02 {m12_68};
  auto m00_12_68 = M00 {m12_68};

  auto m12_34 = make_native_matrix<M12>(3, 4);

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

  auto m11_0 = make_native_matrix<M11>(0);

  EXPECT_TRUE(is_near(solve(m11_0, m12_68), M12::Zero()));
  EXPECT_TRUE(is_near(solve(M10 {m11_0}, m12_68), M12::Zero()));
  EXPECT_TRUE(is_near(solve(M01 {m11_0}, m12_68), M12::Zero()));
  EXPECT_TRUE(is_near(solve(M00 {m11_0}, m12_68), M12::Zero()));
}

