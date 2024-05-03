/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, solve_constant_diagonal_A)
{
  auto cd22 = M22::Identity() + M22::Identity();
  auto cdxx_22 = Mxx::Identity(2, 2) + Mxx::Identity(2, 2);

  auto m23_x = make_eigen_matrix<double, 2, 3>(5, 7, 9, 6, 8, 10);

  auto m23 = make_eigen_matrix<double, 2, 3>(10, 14, 18, 12, 16, 20);
  auto m2x_3 = M2x {m23};
  auto mx3_2 = Mx3 {m23};
  auto mxx_23 = Mxx {m23};

  EXPECT_TRUE(is_near(solve(M22::Identity(), m23), m23));

  EXPECT_TRUE(is_near(solve(cd22, m23), m23_x)); 
  EXPECT_TRUE(is_near(solve(cd22, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(cd22, mx3_2), m23_x)); static_assert(dimension_size_of_index_is<decltype(solve(cd22, mx3_2)), 0, 2>);
  EXPECT_TRUE(is_near(solve(cd22, mxx_23), m23_x)); static_assert(dimension_size_of_index_is<decltype(solve(cd22, mxx_23)), 0, 2>);
  EXPECT_TRUE(is_near(solve(cdxx_22, m23), m23_x));
  EXPECT_TRUE(is_near(solve(cdxx_22, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(cdxx_22, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(cdxx_22, mxx_23), m23_x));
}


TEST(eigen3, solve_constant_A)
{
  auto c11_2 = M11::Identity() + M11::Identity();

  auto m12_1 = make_dense_object_from<M12>(1, 1);

  auto m12_2 = make_dense_object_from<M12>(2, 2);
  auto m1x_2_2 = M1x {m12_2};
  auto mx2_1_2 = Mx2 {m12_2};
  auto mxx_12_2 = Mxx {m12_2};

  EXPECT_TRUE(is_near(solve(c11_2, m12_2), m12_1));
  EXPECT_TRUE(is_near(solve(c11_2, m1x_2_2), m12_1));
  EXPECT_TRUE(is_near(solve(c11_2, mx2_1_2), m12_1)); static_assert(dimension_size_of_index_is<decltype(solve(c11_2, mx2_1_2)), 0, 1>);
  EXPECT_TRUE(is_near(solve(c11_2, mxx_12_2), m12_1)); static_assert(dimension_size_of_index_is<decltype(solve(c11_2, mxx_12_2)), 0, 1>);

  auto c22 = Eigen::Replicate<decltype(c11_2), 2, 2> {c11_2, 2, 2};
  auto c2x_2 = Eigen::Replicate<decltype(c11_2), 2, Eigen::Dynamic> {c11_2, 2, 2};
  auto cx2_2 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, 2> {c11_2, 2, 2};
  auto cxx_22 = Eigen::Replicate<decltype(c11_2), Eigen::Dynamic, Eigen::Dynamic> {c11_2, 2, 2};

  auto c11_8 = c11_2 + c11_2 + c11_2 + c11_2;

  auto c23 = Eigen::Replicate<decltype(c11_8), 2, 3> {c11_8, 2, 3};
  auto c2x_3 = Eigen::Replicate<decltype(c11_8), 2, Eigen::Dynamic> {c11_8, 2, 3};
  auto cx3_2 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, 3> {c11_8, 2, 3};
  auto cxx_23 = Eigen::Replicate<decltype(c11_8), Eigen::Dynamic, Eigen::Dynamic> {c11_8, 2, 3};

  auto m23_x = make_eigen_matrix<double, 2, 3>(6, 9, 12, 6, 9, 12);

  auto m23 = make_eigen_matrix<double, 2, 3>(24, 36, 48, 24, 36, 48);
  auto m2x_3 = M2x {m23};
  auto mx3_2 = Mx3 {m23};
  auto mxx_23 = Mxx {m23};

  EXPECT_TRUE(is_near(solve(c22, m23), m23_x));
  EXPECT_TRUE(is_near(solve(c22, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(c22, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(c22, mxx_23), m23_x));
  EXPECT_TRUE(is_near(solve(c2x_2, m23), m23_x));
  EXPECT_TRUE(is_near(solve(c2x_2, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(c2x_2, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(c2x_2, mxx_23), m23_x));
  EXPECT_TRUE(is_near(solve(cx2_2, m23), m23_x));
  EXPECT_TRUE(is_near(solve(cx2_2, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(cx2_2, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(cx2_2, mxx_23), m23_x));
  EXPECT_TRUE(is_near(solve(cxx_22, m23), m23_x));
  EXPECT_TRUE(is_near(solve(cxx_22, m2x_3), m23_x));
  EXPECT_TRUE(is_near(solve(cxx_22, mx3_2), m23_x));
  EXPECT_TRUE(is_near(solve(cxx_22, mxx_23), m23_x));

  auto m33_x = make_eigen_matrix<double, 1, 3>(3, 4.5, 6);

  EXPECT_TRUE(is_near(solve<false, true>(c23, m23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c23, m2x_3).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c23, mx3_2).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c23, mxx_23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c2x_3, m23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c2x_3, m2x_3).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c2x_3, mx3_2).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(c2x_3, mxx_23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(cx3_2, m23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(cx3_2, m2x_3).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(cx3_2, mx3_2).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(cx3_2, mxx_23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(cxx_23, m23).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(cxx_23, m2x_3).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(cxx_23, mx3_2).colwise().sum(), m33_x));
  EXPECT_TRUE(is_near(solve<false, true>(cxx_23, mxx_23).colwise().sum(), m33_x));
}


TEST(eigen3, solve_one_by_one)
{
  M11 m11_0; m11_0 << 0;
  M1x m1x_1_0(1,1); m1x_1_0 << 0;
  Mx1 mx1_1_0(1,1); mx1_1_0 << 0;
  Mxx mxx_11_0(1,1); mxx_11_0 << 0;

  M11 m11_6; m11_6 << 6;
  M1x m1x_1_6(1,1); m1x_1_6 << 6;
  Mx1 mx1_1_6(1,1); mx1_1_6 << 6;
  Mxx mxx_11_6(1,1); mxx_11_6 << 6;

  auto inf = std::numeric_limits<double>::infinity();

  EXPECT_EQ(trace(solve(m11_0, m11_6)), inf); 
  EXPECT_EQ(trace(solve(m11_0, m1x_1_6)), inf);
  EXPECT_EQ(trace(solve(m11_0, mx1_1_6)), inf);
  EXPECT_EQ(trace(solve(m11_0, mxx_11_6)), inf);
  EXPECT_EQ(trace(solve(m1x_1_0, m11_6)), inf);
  EXPECT_EQ(trace(solve(m1x_1_0, m1x_1_6)), inf);
  EXPECT_EQ(trace(solve(m1x_1_0, mx1_1_6)), inf);
  EXPECT_EQ(trace(solve(m1x_1_0, mxx_11_6)), inf);
  EXPECT_EQ(trace(solve(mx1_1_0, m11_6)), inf);
  EXPECT_EQ(trace(solve(mx1_1_0, m1x_1_6)), inf);
  EXPECT_EQ(trace(solve(mx1_1_0, mx1_1_6)), inf);
  EXPECT_EQ(trace(solve(mx1_1_0, mxx_11_6)), inf);
  EXPECT_EQ(trace(solve(mxx_11_0, m11_6)), inf);
  EXPECT_EQ(trace(solve(mxx_11_0, m1x_1_6)), inf);
  EXPECT_EQ(trace(solve(mxx_11_0, mx1_1_6)), inf);
  EXPECT_EQ(trace(solve(mxx_11_0, mxx_11_6)), inf);

  auto m12_68 = make_dense_object_from<M12>(6, 8);
  auto m1x_2_68 = M1x {m12_68};
  auto mx2_1_68 = Mx2 {m12_68};
  auto mxx_12_68 = Mxx {m12_68};

  EXPECT_EQ(solve(m11_0, m12_68)(0,0), inf);
  EXPECT_EQ(solve(m1x_1_0, m12_68)(0,1), inf);
  EXPECT_EQ(solve(mx1_1_0, m12_68)(0,0), inf);
  EXPECT_EQ(solve(mxx_11_0, m12_68)(0,1), inf);
  EXPECT_EQ(solve(m11_0, m1x_2_68)(0,0), inf);
  EXPECT_EQ(solve(m1x_1_0, m1x_2_68)(0,1), inf);
  EXPECT_EQ(solve(mx1_1_0, m1x_2_68)(0,0), inf);
  EXPECT_EQ(solve(mxx_11_0, m1x_2_68)(0,1), inf);
  EXPECT_EQ(solve(m11_0, mx2_1_68)(0,0), inf);
  EXPECT_EQ(solve(m1x_1_0, mx2_1_68)(0,1), inf);
  EXPECT_EQ(solve(mx1_1_0, mx2_1_68)(0,0), inf);
  EXPECT_EQ(solve(mxx_11_0, mx2_1_68)(0,1), inf);
  EXPECT_EQ(solve(m11_0, mxx_12_68)(0,0), inf);
  EXPECT_EQ(solve(m1x_1_0, mxx_12_68)(0,1), inf);
  EXPECT_EQ(solve(mx1_1_0, mxx_12_68)(0,0), inf);
  EXPECT_EQ(solve(mxx_11_0, mxx_12_68)(0,1), inf);

  M11 m11_2; m11_2 << 2;
  M1x m1x_1_2(1,1); m1x_1_2 << 2;
  Mx1 mx1_1_2(1,1); mx1_1_2 << 2;
  Mxx mxx_11_2(1,1); mxx_11_2 << 2;

  M11 m11_3; m11_3 << 3;

  EXPECT_TRUE(is_near(solve(m11_2, m11_6), m11_3)); 
  EXPECT_TRUE(is_near(solve(m11_2, m1x_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m11_2, mx1_1_6), m11_3)); static_assert(one_dimensional<decltype(solve(m11_2, mx1_1_6))>);
  EXPECT_TRUE(is_near(solve(m11_2, mxx_11_6), m11_3)); static_assert(dimension_size_of_index_is<decltype(solve(m11_2, mxx_11_6)), 0, 1>);
  EXPECT_TRUE(is_near(solve(m1x_1_2, m11_6), m11_3));
  EXPECT_TRUE(is_near(solve(m1x_1_2, m1x_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m1x_1_2, mx1_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(m1x_1_2, mxx_11_6), m11_3));
  EXPECT_TRUE(is_near(solve(mx1_1_2, m11_6), m11_3));
  EXPECT_TRUE(is_near(solve(mx1_1_2, m1x_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(mx1_1_2, mx1_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(mx1_1_2, mxx_11_6), m11_3));
  EXPECT_TRUE(is_near(solve(mxx_11_2, m11_6), m11_3));
  EXPECT_TRUE(is_near(solve(mxx_11_2, m1x_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(mxx_11_2, mx1_1_6), m11_3));
  EXPECT_TRUE(is_near(solve(mxx_11_2, mxx_11_6), m11_3));

  auto m12_34 = make_dense_object_from<M12>(3, 4);

  EXPECT_TRUE(is_near(solve(m11_2, m12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m11_2, m1x_2_68), m12_34));
  EXPECT_TRUE(is_near(solve(m11_2, mx2_1_68), m12_34)); static_assert(dimension_size_of_index_is<decltype(solve(m11_2, mx2_1_68)), 0, 1>);
  EXPECT_TRUE(is_near(solve(m11_2, mxx_12_68), m12_34)); static_assert(dimension_size_of_index_is<decltype(solve(m11_2, mxx_12_68)), 0, 1>);
  EXPECT_TRUE(is_near(solve(m1x_1_2, m12_68), m12_34));
  EXPECT_TRUE(is_near(solve(m1x_1_2, m1x_2_68), m12_34));
  EXPECT_TRUE(is_near(solve(m1x_1_2, mx2_1_68), m12_34));
  EXPECT_TRUE(is_near(solve(m1x_1_2, mxx_12_68), m12_34));
  EXPECT_TRUE(is_near(solve(mx1_1_2, m12_68), m12_34));
  EXPECT_TRUE(is_near(solve(mx1_1_2, m1x_2_68), m12_34));
  EXPECT_TRUE(is_near(solve(mx1_1_2, mx2_1_68), m12_34));
  EXPECT_TRUE(is_near(solve(mx1_1_2, mxx_12_68), m12_34));
  EXPECT_TRUE(is_near(solve(mxx_11_2, m12_68), m12_34));
  EXPECT_TRUE(is_near(solve(mxx_11_2, m1x_2_68), m12_34));
  EXPECT_TRUE(is_near(solve(mxx_11_2, mx2_1_68), m12_34));
  EXPECT_TRUE(is_near(solve(mxx_11_2, mxx_12_68), m12_34));
}


TEST(eigen3, solve_general_matrix)
{
  auto m22 = make_dense_object_from<M22>(1, 2, 3, 4);
  auto m2x_2 = M2x {m22};
  auto mx2_2 = Mx2 {m22};
  auto mxx_22 = Mxx {m22};

  auto m23_56 = make_eigen_matrix<double, 2, 3>(5, 7, 9, 6, 8, 10);
  auto m2x_3_56 = M2x {m23_56};
  auto mx3_2_56 = Mx3 {m23_56};
  auto mxx_23_56 = Mxx {m23_56};

  auto m23_445 = make_eigen_matrix<double, 2, 3>(-4, -6, -8, 4.5, 6.5, 8.5);

  EXPECT_TRUE(is_near(solve(m22, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m22, m2x_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(m22, mx3_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(m22, mxx_23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m2x_2, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(m2x_2, m2x_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(m2x_2, mx3_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(m2x_2, mxx_23_56), m23_445));
  EXPECT_TRUE(is_near(solve(mx2_2, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(mx2_2, m2x_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(mx2_2, mx3_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(mx2_2, mxx_23_56), m23_445));
  EXPECT_TRUE(is_near(solve(mxx_22, m23_56), m23_445));
  EXPECT_TRUE(is_near(solve(mxx_22, m2x_3_56), m23_445));
  EXPECT_TRUE(is_near(solve(mxx_22, mx3_2_56), m23_445));
  EXPECT_TRUE(is_near(solve(mxx_22, mxx_23_56), m23_445));
}


TEST(eigen3, solve_triangular)
{
  auto m22_3104 = make_dense_object_from<M22>(3, 1, 0, 4);
  auto m2x_3104 = M2x {m22_3104};
  auto mx2_3104 = Mx2 {m22_3104};
  auto mxx_3104 = Mxx {m22_3104};

  auto m22_5206 = make_dense_object_from<M22>(5, 2, 0, 6);

  auto m22_1512024 = make_eigen_matrix<double, 2, 2>(15, 12, 0, 24);
  auto m2x_1512024 = M2x {m22_1512024};
  auto mx2_1512024 = Mx2 {m22_1512024};
  auto mxx_1512024 = Mxx {m22_1512024};

  static_assert(triangular_matrix<decltype(solve(m22_3104.template triangularView<Eigen::Upper>(), m22_1512024.template triangularView<Eigen::Upper>())), TriangleType::upper>);
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::Upper>(), m22_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::Upper>(), m2x_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::Upper>(), mx2_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m22_3104.template triangularView<Eigen::Upper>(), mxx_1512024.template triangularView<Eigen::Upper>()), m22_5206));

  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::Upper>(), m22_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::Upper>(), m2x_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::Upper>(), mx2_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(m2x_3104.template triangularView<Eigen::Upper>(), mxx_1512024.template triangularView<Eigen::Upper>()), m22_5206));

  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::Upper>(), m22_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::Upper>(), m2x_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::Upper>(), mx2_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  static_assert(triangular_matrix<decltype(solve(mx2_3104.template triangularView<Eigen::Upper>(), mx2_1512024.template triangularView<Eigen::Upper>())), TriangleType::upper>);
  EXPECT_TRUE(is_near(solve(mx2_3104.template triangularView<Eigen::Upper>(), mxx_1512024.template triangularView<Eigen::Upper>()), m22_5206));

  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::Upper>(), m22_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::Upper>(), m2x_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::Upper>(), mx2_1512024.template triangularView<Eigen::Upper>()), m22_5206));
  EXPECT_TRUE(is_near(solve(mxx_3104.template triangularView<Eigen::Upper>(), mxx_1512024.template triangularView<Eigen::Upper>()), m22_5206));
}

