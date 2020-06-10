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

using M2 = Eigen::Matrix<double, 2, 3>;
using M3E = Eigen::Matrix<double, 3, 3>;
using M3 = Eigen::Matrix<double, 3, 2>;
using M4 = Eigen::Matrix<double, 4, 2>;
using C = Coefficients<Axis, Angle, Axis>;
using Mat2 = TypedMatrix<Coefficients<Axis, Angle>, Axes<3>, M2>;
using Mat3E = TypedMatrix<Axes<3>, Axes<3>, M3E>;
using Mat3 = TypedMatrix<C, Axes<2>, M3>;
using Mat4 = TypedMatrix<Axes<4>, Axes<2>, M4>;
using To2 = ToEuclideanExpr<Coefficients<Axis, Angle>, M2>;
using To3 = ToEuclideanExpr<C, M3>;
using ToFrom4 = ToEuclideanExpr<C, FromEuclideanExpr<C, M4>>;

using GetCoeff = std::function<double(const Eigen::Index)>;
inline GetCoeff g(const double x, const double y = 0, const double z = 0)
{
  return [=](const Eigen::Index i) { if (i==0) return x; else if (i==1) return y; else return z;};
}


TEST_F(matrix_tests, toEuclidean_Coefficients)
{
  EXPECT_NEAR((Axis::to_Euclidean_array<double, 0>[0](g(3.))), 3., 1e-6);
  EXPECT_NEAR((Angle::to_Euclidean_array<double, 0>[0](g(M_PI/3))), 0.5, 1e-6);
  EXPECT_NEAR((Angle::to_Euclidean_array<double, 0>[1](g(M_PI/3))), std::sqrt(3)/2, 1e-6);

  EXPECT_NEAR((to_Euclidean<Axis, double>(0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((to_Euclidean<Angle, double>(0, g(M_PI / 3))), 0.5, 1e-6);
  EXPECT_NEAR((to_Euclidean<Angle, double>(1, g(M_PI / 6))), 0.5, 1e-6);

  EXPECT_NEAR((to_Euclidean<Coefficients<Axis>, double>(0, g(3.))), 3., 1e-6);
  EXPECT_NEAR((to_Euclidean<Coefficients<Axis, Axis>, double>(0, g(3., 2.))), 3., 1e-6);
  EXPECT_NEAR((to_Euclidean<Coefficients<Axis, Axis>, double>(1, g(3., 2.))), 2., 1e-6);

  EXPECT_NEAR((to_Euclidean<Coefficients<Angle>, double>(0, g(M_PI / 3))), 0.5, 1e-6);
  EXPECT_NEAR((to_Euclidean<Coefficients<Angle>, double>(1, g(M_PI / 6))), 0.5, 1e-6);
  EXPECT_NEAR((to_Euclidean<Coefficients<Angle, Angle>, double>(0, g(M_PI / 3, M_PI / 6))), 0.5, 1e-6);
  EXPECT_NEAR((to_Euclidean<Coefficients<Angle, Angle>, double>(1, g(M_PI / 3, M_PI / 6))), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_Euclidean<Coefficients<Angle, Angle>, double>(2, g(M_PI / 3, M_PI / 6))), std::sqrt(3) / 2, 1e-6);
  EXPECT_NEAR((to_Euclidean<Coefficients<Angle, Angle>, double>(3, g(M_PI / 3, M_PI / 6))), 0.5, 1e-6);

  EXPECT_NEAR((to_Euclidean<Coefficients<Axis, Angle>, double>(0, g(3., M_PI / 3))), 3., 1e-6);
  EXPECT_NEAR((to_Euclidean<Coefficients<Axis, Angle>, double>(1, g(3., M_PI / 3))), 0.5, 1e-6);
  EXPECT_NEAR((to_Euclidean<Coefficients<Axis, Angle>, double>(2, g(3., M_PI / 3))), std::sqrt(3) / 2, 1e-6);
}


TEST_F(matrix_tests, ToEuclideanExpr_class)
{
  M4 m;
  m << 1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4;
  To3 d1;
  d1 << 1, 2, M_PI/6, M_PI/3, 3, 4;
  EXPECT_TRUE(is_near(d1.base_matrix(), Mat3 {1, 2, M_PI/6, M_PI/3, 3, 4}));
  EXPECT_TRUE(is_near(d1, m));
  ToFrom4 d1b;
  d1b << 1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4;
  EXPECT_TRUE(is_near(d1b.base_matrix(), Mat3 {1, 2, M_PI/6, M_PI/3, 3, 4}));
  EXPECT_TRUE(is_near(d1b, Mat4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}));
  //
  To3 d2 = (M3() << 1, 2, M_PI/6, M_PI/3, 3, 4).finished();
  EXPECT_TRUE(is_near(d2, m));
  To3 d3 = d2;
  EXPECT_TRUE(is_near(d3, m));
  To3 d4 = To3{1, 2, M_PI/6, M_PI/3, 3, 4};
  EXPECT_TRUE(is_near(d4, m));
  To3 d5 = MatrixTraits<M3>::zero();
  EXPECT_TRUE(is_near(d5, Mat4 {0, 0, 1, 1, 0, 0, 0, 0}));
  To3 d6 = EigenZero<M3>();
  EXPECT_TRUE(is_near(d6, Mat4 {0, 0, 1, 1, 0, 0, 0, 0}));
  To3 d7 = To3(EigenZero<M3>());
  EXPECT_TRUE(is_near(d7, Mat4 {0, 0, 1, 1, 0, 0, 0, 0}));
  To3 d8 {1, 2, M_PI/6, M_PI/3, 3, 4};
  EXPECT_TRUE(is_near(d8, m));
  EXPECT_TRUE(is_near(To3(EigenZero<M3>()), Mat4 {0, 0, 1, 1, 0, 0, 0, 0}));
  ToFrom4 d9 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4};
  EXPECT_TRUE(is_near(d9.base_matrix(), Mat3 {1, 2, M_PI/6, M_PI/3, 3, 4}));
  EXPECT_TRUE(is_near(d9, Mat4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}));
  //
  d5 = d1;
  EXPECT_TRUE(is_near(d5, m));
  d6 = To3 {1, 2, M_PI/6, M_PI/3, 3, 4};
  EXPECT_TRUE(is_near(d5, m));
  d7 = m;
  EXPECT_TRUE(is_near(d7, m));
  d7 = M4::Zero();
  d7 = {1, 2, M_PI/6, M_PI/3, 3, 4};
  EXPECT_TRUE(is_near(d7, m));
  d9 = M4::Zero();
  d9 = {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4};
  EXPECT_TRUE(is_near(d9, m));
  //
  d1 += d2;
  EXPECT_TRUE(is_near(d1, Mat4 {2, 4, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 6, 8}));
  d1 -= To3 {1, 2, M_PI/6, M_PI/3, 3, 4};
  EXPECT_TRUE(is_near(d1, Mat4 {1, 2, 1, 1, 0, 0, 3, 4}));
  d1 *= 3;
  EXPECT_TRUE(is_near(d1, Mat4 {3, 6, 1, 1, 0, 0, 9, 12}));
  d1 /= 3;
  EXPECT_TRUE(is_near(d1, Mat4 {1, 2, 1, 1, 0, 0, 3, 4}));
  EXPECT_TRUE(is_near(d1.zero(), M4::Zero()));
  EXPECT_TRUE(is_near(ToEuclideanExpr<Axes<2>, Eigen::Matrix<double, 2, 2>>::identity(), Eigen::Matrix<double, 2, 2>::Identity()));
  ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 1>> e1 = {3, M_PI/4};
  EXPECT_EQ(e1[0], 3);
  EXPECT_NEAR(e1(1), std::sqrt(2.)/2, 1e-6);
  EXPECT_NEAR(e1(2), std::sqrt(2.)/2, 1e-6);
  EXPECT_EQ(d1(0, 1), 2);
  EXPECT_EQ(d1(3, 1), 4);
  ToEuclideanExpr<Coefficients<Axes<2>>, Eigen::Matrix<double, 2, 2>> e2 = {1, 2, 3, 4};
  e2(0,0) = 5;
  EXPECT_EQ(e2(0, 0), 5);
  e2(0,1) = 6;
  EXPECT_EQ(e2(0, 1), 6);
  e2(1,0) = 7;
  EXPECT_EQ(e2(1, 0), 7);
  e2(1,1) = 8;
  EXPECT_EQ(e2(1, 1), 8);
  EXPECT_TRUE(is_near(e2, (Eigen::Matrix<double, 2, 2>() << 5, 6, 7, 8).finished()));
  EXPECT_NEAR((ToEuclideanExpr<C, Eigen::Matrix<double, 3, 1>>{1, M_PI/6, 3})(1), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((ToEuclideanExpr<C, Eigen::Matrix<double, 3, 1>>{1, M_PI/6, 3})(2), 0.5, 1e-6);
  EXPECT_NEAR((To3 {1, 2, M_PI/6, M_PI/3, 3, 4})(0, 0), 1, 1e-6);
  EXPECT_NEAR((To3 {1, 2, M_PI/6, M_PI/3, 3, 4})(1, 0), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((To3 {1, 2, M_PI/6, M_PI/3, 3, 4})(1, 1), 0.5, 1e-6);
  EXPECT_NEAR((To3 {1, 2, M_PI/6, M_PI/3, 3, 4})(2, 0), 0.5, 1e-6);
}


TEST_F(matrix_tests, ToEuclideanExpr_traits)
{
  static_assert(is_ToEuclideanExpr_v<decltype(To3 {1, 2, M_PI/6, M_PI/3, 3, 4})>);
  static_assert(is_typed_matrix_base_v<decltype(To3 {1, 2, M_PI/6, M_PI/3, 3, 4})>);
  static_assert(not is_FromEuclideanExpr_v<decltype(To3 {1, 2, M_PI/6, M_PI/3, 3, 4})>);
  static_assert(not is_native_Eigen_type_v<decltype(To3 {1, 2, M_PI/6, M_PI/3, 3, 4})>);
  static_assert(not is_Eigen_matrix_v<decltype(To3 {1, 2, M_PI/6, M_PI/3, 3, 4})>);
  static_assert(not is_identity_v<decltype(To3 {1, 2, M_PI/6, M_PI/3, 3, 4})>);
  static_assert(not is_zero_v<decltype(To3 {1, 2, M_PI/6, M_PI/3, 3, 4})>);
  // MatrixTraits
  EXPECT_TRUE(is_near(MatrixTraits<To3>::make((Eigen::Matrix<double, 3, 2>() << 1, 2, M_PI/6, M_PI/3, 3, 4).finished()),
    Mat4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}));
  EXPECT_TRUE(is_near(MatrixTraits<To3>::make(1, 2, M_PI/6, M_PI/3, 3, 4),
    Mat4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}));
  EXPECT_TRUE(is_near(MatrixTraits<To3>::zero(), Eigen::Matrix<double, 4, 2>::Zero()));
  EXPECT_TRUE(is_near(ToEuclideanExpr<Axes<2>, Eigen::Matrix<double, 2, 2>>::identity(), Eigen::Matrix<double, 2, 2>::Identity()));
}


TEST_F(matrix_tests, ToEuclideanExpr_overloads)
{
  EXPECT_TRUE(is_near(base_matrix(To3 {1, 2, M_PI/6, M_PI/3, 3, 4}), Mat3 {1, 2, M_PI/6, M_PI/3, 3, 4}));
  EXPECT_TRUE(is_near(strict_matrix(To3 {1, 2, M_PI/6, M_PI/3, 3, 4}), Mat4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}));
  EXPECT_TRUE(is_near(strict(To3 {1, 2, M_PI/6, M_PI/3, 3, 4}), Mat4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}));
  EXPECT_TRUE(is_near(from_Euclidean(To3 {1, 2, 2*M_PI + M_PI/6, -4*M_PI + M_PI/3, 3, 4}), Mat3 {1, 2, M_PI/6, M_PI/3, 3, 4}));
  EXPECT_TRUE(is_near(from_Euclidean<C>(To3 {1, 2, 2*M_PI + M_PI/6, -4*M_PI + M_PI/3, 3, 4}), Mat3 {1, 2, M_PI/6, M_PI/3, 3, 4}));
  EXPECT_TRUE(is_near(to_diagonal(ToEuclideanExpr<C, Eigen::Matrix<double, 3, 1>>{1, M_PI/6, 3}), EigenDiagonal {1., std::sqrt(3)/2, 0.5, 3}));
  EXPECT_TRUE(is_near(transpose(To3 {1, 2, M_PI/6, M_PI/3, 3, 4}), (Eigen::Matrix<double, 2, 4>() << 1, std::sqrt(3)/2, 0.5, 3, 2, 0.5, std::sqrt(3)/2, 4).finished()));
  EXPECT_TRUE(is_near(adjoint(To3 {1, 2, M_PI/6, M_PI/3, 3, 4}), (Eigen::Matrix<double, 2, 4>() << 1, std::sqrt(3)/2, 0.5, 3, 2, 0.5, std::sqrt(3)/2, 4).finished()));
  EXPECT_NEAR(determinant(To2 {1, 2, 3, M_PI/6, M_PI/3, M_PI/4}), 0.25 * (3 * std::sqrt(2) - 3 * std::sqrt(6.) + 6), 1e-6);
  EXPECT_NEAR(trace(To2 {1, 2, 3, M_PI/6, M_PI/3, M_PI/4}), 1.5 + std::sqrt(2.)/2, 1e-6);
  EXPECT_TRUE(is_near(solve(
    To2 {1, 2, 3, M_PI/6, M_PI/3, M_PI/4},
    (Eigen::Matrix<double, 3, 1>() << 14, std::sqrt(3)/2 + 1 + std::sqrt(2)*3/2, 0.5 + std::sqrt(3) + std::sqrt(2)*3/2).finished()),
    (Eigen::Matrix<double, 3, 1>() << 1, 2, 3).finished()));
  EXPECT_TRUE(is_near(reduce_columns(To3 {1, 2, M_PI/6, M_PI/3, 3, 4}),
    (Eigen::Matrix<double, 4, 1>() << 1.5, 0.5*(std::sqrt(3)/2 + 0.5), 0.5*(std::sqrt(3)/2 + 0.5), 3.5).finished()));
  EXPECT_TRUE(is_near(LQ_decomposition(To2 {1, 2, 3, M_PI/6, M_PI/3, M_PI/4}),
    LQ_decomposition((Eigen::Matrix<double, 3, 3>() <<
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2).finished())));
  EXPECT_TRUE(is_near(QR_decomposition(To2 {1, 2, 3, M_PI/6, M_PI/3, M_PI/4}),
    QR_decomposition((Eigen::Matrix<double, 3, 3>() <<
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2).finished())));

  auto m = strict_matrix(MatrixTraits<Eigen::Matrix<double, 3, 2>>::zero());
  auto offset = Eigen::Matrix<double, 3, 2>::Constant(1);
  for (int i=0; i<100; i++)
  {
    m = (m * i + offset + from_Euclidean(randomize<To3>(0.7))) / (i + 1);
  }
  EXPECT_TRUE(is_near(m, offset, 0.1));
  EXPECT_FALSE(is_near(m, offset, 1e-6));
}


TEST_F(matrix_tests, ToEuclideanExpr_blocks)
{
  EXPECT_TRUE(is_near(concatenate_vertical(
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 3>> {1., 2, 3, M_PI/6, M_PI/3, M_PI/4},
    ToEuclideanExpr<Coefficients<Angle, Axis>, Eigen::Matrix<double, 2, 3>> {M_PI/4, M_PI/3, M_PI/6, 4, 5, 6}),
    (Eigen::Matrix<double, 6, 3>() <<
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2,
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
      4, 5, 6).finished()));
  EXPECT_TRUE(is_near(concatenate_horizontal(
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 3>> {1., 2, 3, M_PI/6, M_PI/3, M_PI/4},
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 3>> {4., 5, 6, M_PI/4, M_PI/3, M_PI/6}),
    (Eigen::Matrix<double, 3, 6>() <<
      1, 2, 3, 4, 5, 6,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2, std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2, std::sqrt(2)/2, std::sqrt(3)/2, 0.5).finished()));
  EXPECT_TRUE(is_near(split_vertical(
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 2>> {1., 2, M_PI/6, M_PI/3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_vertical<3, 3>(
    ToEuclideanExpr<Coefficients<Axis, Angle, Angle, Axis>, Eigen::Matrix<double, 4, 3>> {
      1., 2, 3,
      M_PI/6, M_PI/3, M_PI/4,
      M_PI/4, M_PI/3, M_PI/6,
      4, 5, 6}),
    std::tuple{(Eigen::Matrix<double, 3, 3>() <<
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2).finished(),
               (Eigen::Matrix<double, 3, 3>() <<
                 std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                 std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
                 4, 5, 6).finished()}));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(
    ToEuclideanExpr<Coefficients<Axis, Angle, Angle, Axis>, Eigen::Matrix<double, 4, 3>> {
      1., 2, 3,
      M_PI/6, M_PI/3, M_PI/4,
      M_PI/4, M_PI/3, M_PI/6,
      4, 5, 6}),
    std::tuple{
      (Eigen::Matrix<double, 3, 3>() <<
        1, 2, 3,
        std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
        0.5, std::sqrt(3)/2, std::sqrt(2)/2).finished(),
      (Eigen::Matrix<double, 2, 3>() <<
        std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
        std::sqrt(2)/2, std::sqrt(3)/2, 0.5).finished()}));
  EXPECT_TRUE(is_near(split_vertical<Coefficients<Axis, Angle>, Coefficients<Angle, Axis>>(
    ToEuclideanExpr<Coefficients<Axis, Angle, Angle, Axis>, Eigen::Matrix<double, 4, 3>> {
      1., 2, 3,
      M_PI/6, M_PI/3, M_PI/4,
      M_PI/4, M_PI/3, M_PI/6,
      4, 5, 6}),
    std::tuple{(Eigen::Matrix<double, 3, 3>() <<
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2).finished(),
               (Eigen::Matrix<double, 3, 3>() <<
                 std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                 std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
                 4, 5, 6).finished()}));
  EXPECT_TRUE(is_near(split_vertical<Coefficients<Axis, Angle>, Coefficients<Angle>>(
    ToEuclideanExpr<Coefficients<Axis, Angle, Angle, Axis>, Eigen::Matrix<double, 4, 3>> {
      1., 2, 3,
      M_PI/6, M_PI/3, M_PI/4,
      M_PI/4, M_PI/3, M_PI/6,
      4, 5, 6}),
    std::tuple{
    (Eigen::Matrix<double, 3, 3>() <<
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2).finished(),
    (Eigen::Matrix<double, 2, 3>() <<
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5).finished()}));
  EXPECT_TRUE(is_near(split_horizontal(
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 2>> {1., 2, M_PI/6, M_PI/3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_horizontal<3, 3>(
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 6>> {
      1., 2, 3, 4, 5, 6,
      M_PI/6, M_PI/3, M_PI/4, M_PI/4, M_PI/3, M_PI/6}),
    std::tuple{(Eigen::Matrix<double, 3, 3>() <<
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2).finished(),
               (Eigen::Matrix<double, 3, 3>() <<
                 4, 5, 6,
                 std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                 std::sqrt(2)/2, std::sqrt(3)/2, 0.5).finished()}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 6>> {
      1., 2, 3, 4, 5, 6,
      M_PI/6, M_PI/3, M_PI/4, M_PI/4, M_PI/3, M_PI/6}),
    std::tuple{
    (Eigen::Matrix<double, 3, 3>() <<
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2).finished(),
    (Eigen::Matrix<double, 3, 2>() <<
      4, 5,
      std::sqrt(2)/2, 0.5,
      std::sqrt(2)/2, std::sqrt(3)/2).finished()}));
  EXPECT_TRUE(is_near(column(
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 3>> {1., 2, 3, M_PI/6, M_PI/3, M_PI/4}, 2),
    (Eigen::Matrix<double, 3, 1>() << 3, std::sqrt(2)/2, std::sqrt(2)/2).finished()));
  EXPECT_TRUE(is_near(column<1>(
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 3>> {1., 2, 3, M_PI/6, M_PI/3, M_PI/4}),
    (Eigen::Matrix<double, 3, 1>() << 2, 0.5, std::sqrt(3)/2).finished()));
  //
  auto b = ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 3>> {1., 2, 3, M_PI/6, M_PI/3, M_PI/4};
  EXPECT_TRUE(is_near(apply_columnwise(b,
    [](auto& col){ col *= 3; }),
    (Eigen::Matrix<double, 3, 3>() <<
      3, 6, 9,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2).finished()));
  b = ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 3>> {1., 2, 3, M_PI/6, M_PI/3, M_PI/4};
  EXPECT_TRUE(is_near(apply_columnwise(b,
    [](auto& col, std::size_t i){ col *= i + 1; }),
    (Eigen::Matrix<double, 3, 3>() <<
      1, 4, 9,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2).finished()));
  EXPECT_TRUE(is_near(apply_columnwise(
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 3>> {1., 2, 3, M_PI/6, M_PI/3, M_PI/4},
    [](const auto& col){ return col * 2; }),
    (Eigen::Matrix<double, 3, 3>() <<
      2, 4, 6,
      std::sqrt(3), 1, std::sqrt(2),
      1, std::sqrt(3), std::sqrt(2)).finished()));
  EXPECT_TRUE(is_near(apply_columnwise(
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 3>> {1., 2, 3, M_PI/6, M_PI/3, M_PI/4},
    [](const auto& col, std::size_t i){ return col * i; }),
    (Eigen::Matrix<double, 3, 3>() <<
      0, 2, 6,
      0, 0.5, std::sqrt(2),
      0, std::sqrt(3)/2, std::sqrt(2)).finished()));
  EXPECT_TRUE(is_near(apply_columnwise<3>(
    [](){ return ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 1>> {1., M_PI/6}; }),
    (Eigen::Matrix<double, 3, 3>() <<
      1, 1, 1,
      std::sqrt(3)/2, std::sqrt(3)/2, std::sqrt(3)/2,
      0.5, 0.5, 0.5).finished()));
  EXPECT_TRUE(is_near(apply_columnwise<3>(
    [](std::size_t i){ return ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 1>> {1., M_PI/6} * (i + 1); }),
    (Eigen::Matrix<double, 3, 3>() <<
      1, 2, 3,
      std::sqrt(3)/2, std::sqrt(3), std::sqrt(3)*3/2,
      0.5, 1, 1.5).finished()));
  //
  EXPECT_TRUE(is_near(apply_coefficientwise(
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 3>> {1., 2, 3, M_PI/6, M_PI/3, M_PI/4},
    [](const auto& x){ return x * 3; }),
    (Eigen::Matrix<double, 3, 3>() <<
      3, 6, 9,
      std::sqrt(3)*3/2, 1.5, std::sqrt(2)*3/2,
      1.5, std::sqrt(3)*3/2, std::sqrt(2)*3/2).finished()));
  EXPECT_TRUE(is_near(apply_coefficientwise(
    ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 3>> {1., 2, 3, M_PI/6, M_PI/3, M_PI/4},
    [](const auto& x, std::size_t i, std::size_t j){ return x * (j + 1); }),
    (Eigen::Matrix<double, 3, 3>() <<
      1, 4, 9,
      std::sqrt(3)/2, 1, std::sqrt(2)*3/2,
      0.5, std::sqrt(3), std::sqrt(2)*3/2).finished()));
}


TEST_F(matrix_tests, ToEuclideanExpr_arithmetic)
{
  EXPECT_TRUE(is_near(To3 {1, 2, M_PI/6, M_PI/3, 3, 4} + To3 {1, 2, M_PI/6, M_PI/3, 3, 4}, Mat4 {2, 4, std::sqrt(3), 1, 1, std::sqrt(3), 6, 8}));
  EXPECT_TRUE(is_near(To3 {1, 2, M_PI/6, M_PI/3, 3, 4} - To3 {1, 2, M_PI/6, M_PI/3, 3, 4}, Mat4::Zero()));
  EXPECT_TRUE(is_near(To3 {1, 2, M_PI/6, M_PI/3, 3, 4} * 2, Mat4 {2, 4, std::sqrt(3), 1, 1, std::sqrt(3), 6, 8}));
  EXPECT_TRUE(is_near(2 * To3 {1, 2, M_PI/6, M_PI/3, 3, 4}, Mat4 {2, 4, std::sqrt(3), 1, 1, std::sqrt(3), 6, 8}));
  EXPECT_TRUE(is_near(To3 {1, 2, M_PI/6, M_PI/3, 3, 4} / 2, Mat4 {0.5, 1, std::sqrt(3)/4, 0.25, 0.25, std::sqrt(3)/4, 1.5, 2}));
  EXPECT_TRUE(is_near(-To3 {1, 2, M_PI/6, M_PI/3, 3, 4}, Mat4 {-1, -2, -std::sqrt(3)/2, -0.5, -0.5, -std::sqrt(3)/2, -3, -4}));
  EXPECT_TRUE(is_near(To3 {1, 2, M_PI/6, M_PI/3, 3, 4} * EigenDiagonal {1., 2}, Mat4 {1, 4, std::sqrt(3)/2, 1, 0.5, std::sqrt(3), 3, 8}));
}


TEST_F(matrix_tests, ToEuclideanExpr_references)
{
  using C = Coefficients<Angle, Axis>;
  using M2 = Eigen::Matrix<double, 2, 2>;
  using M3 = Eigen::Matrix<double, 3, 2>;
  M2 m, n;
  m << M_PI/6, M_PI/4, 1, 2;
  n << M_PI/4, M_PI/3, 3, 4;
  M3 me, ne;
  me << std::sqrt(3)/2, std::sqrt(2)/2, 0.5, std::sqrt(2)/2, 1, 2;
  ne << std::sqrt(2)/2, 0.5, std::sqrt(2)/2, std::sqrt(3)/2, 3, 4;
  ToEuclideanExpr<C, M2> x = m;
  ToEuclideanExpr<C, M2&> x_lvalue = x;
  EXPECT_TRUE(is_near(x_lvalue, me));
  x = ToEuclideanExpr<C, M2>(n);
  EXPECT_TRUE(is_near(x_lvalue, ne));
  x_lvalue = ToEuclideanExpr<C, M2>(m);
  EXPECT_TRUE(is_near(x, me));
  ToEuclideanExpr<C, M2&&> x_rvalue = std::move(x);
  EXPECT_TRUE(is_near(x_rvalue, me));
  x_rvalue = ToEuclideanExpr<C, M2>(n);
  EXPECT_TRUE(is_near(x_rvalue, ne));
}
