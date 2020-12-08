/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen3.gtest.hpp"

using namespace OpenKalman;

using M2 = native_matrix_t<double, 2, 3>;
using M3 = native_matrix_t<double, 3, 2>;
using M4 = native_matrix_t<double, 4, 2>;
using C = Coefficients<Axis, angle::Radians, Axis>;
using To2 = ToEuclideanExpr<Coefficients<Axis, angle::Radians>, M2>;
using To3 = ToEuclideanExpr<C, M3>;
using ToFrom4 = ToEuclideanExpr<C, FromEuclideanExpr<C, M4>>;

template<typename...Args>
inline auto mat2(Args...args) { return MatrixTraits<M2>::make(args...); }

template<typename...Args>
inline auto mat3(Args...args) { return MatrixTraits<M3>::make(args...); }

template<typename...Args>
inline auto mat4(Args...args) { return MatrixTraits<M4>::make(args...); }


TEST_F(eigen3, ToEuclideanExpr_class)
{
  M4 m;
  m << 1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4;
  To3 d1;
  d1 << 1, 2, pi/6, pi/3, 3, 4;
  EXPECT_TRUE(is_near(d1.nested_matrix(), mat3(1, 2, pi/6, pi/3, 3, 4)));
  EXPECT_TRUE(is_near(d1, m));
  ToFrom4 d1b;
  d1b << 1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4;
  EXPECT_TRUE(is_near(d1b.nested_matrix(), mat3(1, 2, pi/6, pi/3, 3, 4)));
  EXPECT_TRUE(is_near(d1b, mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  //
  To3 d2 = (M3() << 1, 2, pi/6, pi/3, 3, 4).finished();
  EXPECT_TRUE(is_near(d2, m));
  To3 d3 = d2;
  EXPECT_TRUE(is_near(d3, m));
  To3 d4 = To3{1, 2, pi/6, pi/3, 3, 4};
  EXPECT_TRUE(is_near(d4, m));
  To3 d5 = MatrixTraits<M3>::zero();
  EXPECT_TRUE(is_near(d5, mat4(0, 0, 1, 1, 0, 0, 0, 0)));
  To3 d6 = ZeroMatrix<double, 3, 2>();
  EXPECT_TRUE(is_near(d6, mat4(0, 0, 1, 1, 0, 0, 0, 0)));
  To3 d7 = To3(ZeroMatrix<double, 3, 2>());
  EXPECT_TRUE(is_near(d7, mat4(0, 0, 1, 1, 0, 0, 0, 0)));
  To3 d8 {1, 2, pi/6, pi/3, 3, 4};
  EXPECT_TRUE(is_near(d8, m));
  EXPECT_TRUE(is_near(To3(ZeroMatrix<double, 3, 2>()), mat4(0, 0, 1, 1, 0, 0, 0, 0)));
  ToFrom4 d9 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4};
  EXPECT_TRUE(is_near(d9.nested_matrix(), mat3(1, 2, pi/6, pi/3, 3, 4)));
  EXPECT_TRUE(is_near(d9, mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  //
  d5 = d1;
  EXPECT_TRUE(is_near(d5, m));
  d6 = To3 {1, 2, pi/6, pi/3, 3, 4};
  EXPECT_TRUE(is_near(d5, m));
  d7 = m;
  EXPECT_TRUE(is_near(d7, m));
  d7 = M4::Zero();
  d7 = {1, 2, pi/6, pi/3, 3, 4};
  EXPECT_TRUE(is_near(d7, m));
  d9 = M4::Zero();
  d9 = {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4};
  EXPECT_TRUE(is_near(d9, m));
  //
  d1 += d2;
  EXPECT_TRUE(is_near(d1, mat4(2, 4, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 6, 8)));
  d1 -= To3 {1, 2, pi/6, pi/3, 3, 4};
  EXPECT_TRUE(is_near(d1, mat4(1, 2, 1, 1, 0, 0, 3, 4)));
  d1 *= 3;
  EXPECT_TRUE(is_near(d1, mat4(3, 6, 1, 1, 0, 0, 9, 12)));
  d1 /= 3;
  EXPECT_TRUE(is_near(d1, mat4(1, 2, 1, 1, 0, 0, 3, 4)));
  EXPECT_TRUE(is_near(d1.zero(), M4::Zero()));
  EXPECT_TRUE(is_near(ToEuclideanExpr<Axes<2>, native_matrix_t<double, 2, 2>>::identity(), native_matrix_t<double, 2, 2>::Identity()));
}


TEST_F(eigen3, ToEuclideanExpr_subscripts)
{
  EXPECT_NEAR(get_element(To3 {1, 2, pi/6, pi/3, 3, 4}, 1, 1), 0.5, 1e-8);
  EXPECT_NEAR(get_element(To3 {1, 2, pi/6, pi/3, 3, 4}, 1, 0), std::sqrt(3)/2, 1e-8);

  ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 1>> e1 = {3, pi/4};
  EXPECT_EQ(e1[0], 3);
  EXPECT_NEAR(e1(1), std::sqrt(2.)/2, 1e-6);
  EXPECT_NEAR(e1(2), std::sqrt(2.)/2, 1e-6);
  ToEuclideanExpr<Coefficients<Axes<2>>, native_matrix_t<double, 2, 2>> e2 = {1, 2, 3, 4};
  e2(0,0) = 5;
  EXPECT_EQ(e2(0, 0), 5);
  e2(0,1) = 6;
  EXPECT_EQ(e2(0, 1), 6);
  e2(1,0) = 7;
  EXPECT_EQ(e2(1, 0), 7);
  e2(1,1) = 8;
  EXPECT_EQ(e2(1, 1), 8);
  EXPECT_TRUE(is_near(e2, make_native_matrix<double, 2, 2>(5, 6, 7, 8)));
  EXPECT_NEAR((ToEuclideanExpr<C, native_matrix_t<double, 3, 1>>{1, pi/6, 3})(1), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((ToEuclideanExpr<C, native_matrix_t<double, 3, 1>>{1, pi/6, 3})(2), 0.5, 1e-6);
  EXPECT_NEAR((To3 {1, 2, pi/6, pi/3, 3, 4})(0, 0), 1, 1e-6);
  EXPECT_NEAR((To3 {1, 2, pi/6, pi/3, 3, 4})(1, 0), std::sqrt(3)/2, 1e-6);
  EXPECT_NEAR((To3 {1, 2, pi/6, pi/3, 3, 4})(1, 1), 0.5, 1e-6);
  EXPECT_NEAR((To3 {1, 2, pi/6, pi/3, 3, 4})(2, 0), 0.5, 1e-6);
}


TEST_F(eigen3, ToEuclideanExpr_traits)
{
  static_assert(to_euclidean_expr<decltype(To3 {1, 2, pi/6, pi/3, 3, 4})>);
  static_assert(typed_matrix_nestable<decltype(To3 {1, 2, pi/6, pi/3, 3, 4})>);
  static_assert(not from_euclidean_expr<decltype(To3 {1, 2, pi/6, pi/3, 3, 4})>);
  static_assert(not eigen_native<decltype(To3 {1, 2, pi/6, pi/3, 3, 4})>);
  static_assert(not eigen_matrix<decltype(To3 {1, 2, pi/6, pi/3, 3, 4})>);
  static_assert(not identity_matrix<decltype(To3 {1, 2, pi/6, pi/3, 3, 4})>);
  static_assert(not zero_matrix<decltype(To3 {1, 2, pi/6, pi/3, 3, 4})>);
  // MatrixTraits
  EXPECT_TRUE(is_near(MatrixTraits<To3>::make(make_native_matrix<double, 3, 2>(1, 2, pi/6, pi/3, 3, 4)),
    mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(MatrixTraits<To3>::make(1, 2, pi/6, pi/3, 3, 4),
    mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(MatrixTraits<To3>::zero(), native_matrix_t<double, 4, 2>::Zero()));
  EXPECT_TRUE(is_near(ToEuclideanExpr<Axes<2>, native_matrix_t<double, 2, 2>>::identity(), native_matrix_t<double, 2, 2>::Identity()));
}


TEST_F(eigen3, ToEuclideanExpr_overloads)
{
  EXPECT_TRUE(is_near(nested_matrix(To3 {1, 2, pi/6, pi/3, 3, 4}), mat3(1, 2, pi/6, pi/3, 3, 4)));
  EXPECT_TRUE(is_near(make_native_matrix(To3 {1, 2, pi/6, pi/3, 3, 4}), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(make_self_contained(To3 {1, 2, pi/6, pi/3, 3, 4}), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(from_euclidean(To3 {1, 2, 2*pi + pi/6, -4*pi + pi/3, 3, 4}), mat3(1, 2, pi/6, pi/3, 3, 4)));
  EXPECT_TRUE(is_near(from_euclidean<C>(To3 {1, 2, 2*pi + pi/6, -4*pi + pi/3, 3, 4}), mat3(1, 2, pi/6, pi/3, 3, 4)));
  EXPECT_TRUE(is_near(to_diagonal(ToEuclideanExpr<C, native_matrix_t<double, 3, 1>>{1, pi/6, 3}), DiagonalMatrix {1., std::sqrt(3)/2, 0.5, 3}));
  EXPECT_TRUE(is_near(transpose(To3 {1, 2, pi/6, pi/3, 3, 4}), make_native_matrix<double, 2, 4>(1, std::sqrt(3)/2, 0.5, 3, 2, 0.5, std::sqrt(3)/2, 4)));
  EXPECT_TRUE(is_near(adjoint(To3 {1, 2, pi/6, pi/3, 3, 4}), make_native_matrix<double, 2, 4>(1, std::sqrt(3)/2, 0.5, 3, 2, 0.5, std::sqrt(3)/2, 4)));
  EXPECT_NEAR(determinant(To2 {1, 2, 3, pi/6, pi/3, pi/4}), 0.25 * (3 * std::sqrt(2) - 3 * std::sqrt(6.) + 6), 1e-6);
  EXPECT_NEAR(trace(To2 {1, 2, 3, pi/6, pi/3, pi/4}), 1.5 + std::sqrt(2.)/2, 1e-6);
  EXPECT_TRUE(is_near(solve(
    To2 {1, 2, 3, pi/6, pi/3, pi/4},
    make_native_matrix<double, 3, 1>(14, std::sqrt(3)/2 + 1 + std::sqrt(2)*3/2, 0.5 + std::sqrt(3) + std::sqrt(2)*3/2)),
    make_native_matrix<double, 3, 1>(1, 2, 3)));
  EXPECT_TRUE(is_near(reduce_columns(To3 {1, 2, pi/6, pi/3, 3, 4}),
    make_native_matrix<double, 4, 1>(1.5, 0.5*(std::sqrt(3)/2 + 0.5), 0.5*(std::sqrt(3)/2 + 0.5), 3.5)));
  EXPECT_TRUE(is_near(LQ_decomposition(To2 {1, 2, 3, pi/6, pi/3, pi/4}),
    LQ_decomposition(make_native_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2))));
  EXPECT_TRUE(is_near(QR_decomposition(To2 {1, 2, 3, pi/6, pi/3, pi/4}),
    QR_decomposition(make_native_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2))));

  auto m = make_native_matrix(MatrixTraits<native_matrix_t<double, 3, 2>>::zero());
  for (int i=0; i<100; i++)
  {
    m = (m * i + from_euclidean(randomize<To3>(1.0, 0.3))) / (i + 1);
  }
  auto offset = native_matrix_t<double, 3, 2>::Constant(1);
  EXPECT_TRUE(is_near(m, offset, 0.1));
  EXPECT_FALSE(is_near(m, offset, 1e-6));
}


TEST_F(eigen3, ToEuclideanExpr_blocks)
{
  EXPECT_TRUE(is_near(concatenate_vertical(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4},
    ToEuclideanExpr<Coefficients<angle::Radians, Axis>, native_matrix_t<double, 2, 3>> {pi/4, pi/3, pi/6, 4, 5, 6}),
    make_native_matrix<double, 6, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2,
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
      4, 5, 6)));
  EXPECT_TRUE(is_near(concatenate_horizontal(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4},
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 3>> {4., 5, 6, pi/4, pi/3, pi/6}),
    make_native_matrix<double, 3, 6>(
      1, 2, 3, 4, 5, 6,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2, std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2, std::sqrt(2)/2, std::sqrt(3)/2, 0.5)));
  EXPECT_TRUE(is_near(split_vertical(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 2>> {1., 2, pi/6, pi/3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_vertical<3, 3>(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians, angle::Radians, Axis>, native_matrix_t<double, 4, 3>> {
      1., 2, 3,
      pi/6, pi/3, pi/4,
      pi/4, pi/3, pi/6,
      4, 5, 6}),
    std::tuple{make_native_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2),
               make_native_matrix<double, 3, 3>(
                 std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                 std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
                 4, 5, 6)}));
  auto a1 = ToEuclideanExpr<Coefficients<Polar<>, angle::Radians, Axis>, native_matrix_t<double, 4, 3>> {
    1., 2, 3,
    pi/6, pi/3, pi/4,
    pi/4, pi/3, pi/6,
    4, 5, 6};
  EXPECT_TRUE(is_near(split_vertical<3, 3>(a1),
    std::tuple{make_native_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2),
    make_native_matrix<double, 3, 3>(
                 std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                 std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
                 4, 5, 6)}));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians, angle::Radians, Axis>, native_matrix_t<double, 4, 3>> {
      1., 2, 3,
      pi/6, pi/3, pi/4,
      pi/4, pi/3, pi/6,
      4, 5, 6}),
    std::tuple{
      make_native_matrix<double, 3, 3>(
        1, 2, 3,
        std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
        0.5, std::sqrt(3)/2, std::sqrt(2)/2),
      make_native_matrix<double, 2, 3>(
        std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
        std::sqrt(2)/2, std::sqrt(3)/2, 0.5)}));
  EXPECT_TRUE(is_near(split_vertical<Coefficients<Axis, angle::Radians>, Coefficients<angle::Radians, Axis>>(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians, angle::Radians, Axis>, native_matrix_t<double, 4, 3>> {
      1., 2, 3,
      pi/6, pi/3, pi/4,
      pi/4, pi/3, pi/6,
      4, 5, 6}),
    std::tuple{make_native_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2),
               make_native_matrix<double, 3, 3>(
                 std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                 std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
                 4, 5, 6)}));
  EXPECT_TRUE(is_near(split_vertical<Coefficients<Axis, angle::Radians>, Coefficients<angle::Radians>>(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians, angle::Radians, Axis>, native_matrix_t<double, 4, 3>> {
      1., 2, 3,
      pi/6, pi/3, pi/4,
      pi/4, pi/3, pi/6,
      4, 5, 6}),
    std::tuple{
    make_native_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2),
    make_native_matrix<double, 2, 3>(
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5)}));

  EXPECT_TRUE(is_near(split_horizontal(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 2>> {1., 2, pi/6, pi/3}), std::tuple{}));
  EXPECT_TRUE(is_near(split_horizontal<3, 3>(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 6>> {
      1., 2, 3, 4, 5, 6,
      pi/6, pi/3, pi/4, pi/4, pi/3, pi/6}),
    std::tuple{make_native_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2),
               make_native_matrix<double, 3, 3>(
                 4, 5, 6,
                 std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                 std::sqrt(2)/2, std::sqrt(3)/2, 0.5)}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(
    ToEuclideanExpr<Polar<>, native_matrix_t<double, 2, 6>> {
      1., 2, 3, 4, 5, 6,
      pi/6, pi/3, pi/4, pi/4, pi/3, pi/6}),
    std::tuple{
    make_native_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2),
    make_native_matrix<double, 3, 2>(
      4, 5,
      std::sqrt(2)/2, 0.5,
      std::sqrt(2)/2, std::sqrt(3)/2)}));

  EXPECT_TRUE(is_near(split_diagonal<Axis, angle::Radians>(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 3>> {
      1., 2, 3,
      pi/6, pi/3, pi/4}),
    std::tuple{make_native_matrix<double, 1, 1>(1),
               make_native_matrix<double, 2, 2>(
                 0.5, std::sqrt(3)/2,
                 std::sqrt(3)/2, 0.5)}));
  EXPECT_TRUE(is_near(split_diagonal<1, 2>(
    ToEuclideanExpr<Polar<>, native_matrix_t<double, 2, 3>> {
      1., 2, 3,
      pi/6, pi/3, pi/4}),
    std::tuple{make_native_matrix<double, 1, 1>(1),
               make_native_matrix<double, 2, 2>(
                 0.5, std::sqrt(3)/2,
                 std::sqrt(3)/2, 0.5)}));

  EXPECT_TRUE(is_near(column(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4}, 2),
    make_native_matrix<double, 3, 1>(3, std::sqrt(2)/2, std::sqrt(2)/2)));
  EXPECT_TRUE(is_near(column<1>(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4}),
    make_native_matrix<double, 3, 1>(2, 0.5, std::sqrt(3)/2)));
  //
  auto b = ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4};
  EXPECT_TRUE(is_near(apply_columnwise(b,
    [](auto& col){ col *= 3; }),
    make_native_matrix<double, 3, 3>(
      3, 6, 9,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2)));
  b = ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4};
  EXPECT_TRUE(is_near(apply_columnwise(b,
    [](auto& col, std::size_t i){ col *= i + 1; }),
    make_native_matrix<double, 3, 3>(
      1, 4, 9,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2)));
  EXPECT_TRUE(is_near(apply_columnwise(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4},
    [](const auto& col){ return col * 2; }),
    make_native_matrix<double, 3, 3>(
      2, 4, 6,
      std::sqrt(3), 1, std::sqrt(2),
      1, std::sqrt(3), std::sqrt(2))));
  EXPECT_TRUE(is_near(apply_columnwise(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4},
    [](const auto& col, std::size_t i){ return col * i; }),
    make_native_matrix<double, 3, 3>(
      0, 2, 6,
      0, 0.5, std::sqrt(2),
      0, std::sqrt(3)/2, std::sqrt(2))));
  EXPECT_TRUE(is_near(apply_columnwise<3>(
    [](){ return ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 1>> {1., pi/6}; }),
    make_native_matrix<double, 3, 3>(
      1, 1, 1,
      std::sqrt(3)/2, std::sqrt(3)/2, std::sqrt(3)/2,
      0.5, 0.5, 0.5)));
  EXPECT_TRUE(is_near(apply_columnwise<3>(
    [](std::size_t i){ return ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 1>> {1., pi/6} * (i + 1); }),
    make_native_matrix<double, 3, 3>(
      1, 2, 3,
      std::sqrt(3)/2, std::sqrt(3), std::sqrt(3)*3/2,
      0.5, 1, 1.5)));
  //
  EXPECT_TRUE(is_near(apply_coefficientwise(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4},
    [](const auto& x){ return x * 3; }),
    make_native_matrix<double, 3, 3>(
      3, 6, 9,
      std::sqrt(3)*3/2, 1.5, std::sqrt(2)*3/2,
      1.5, std::sqrt(3)*3/2, std::sqrt(2)*3/2)));
  EXPECT_TRUE(is_near(apply_coefficientwise(
    ToEuclideanExpr<Coefficients<Axis, angle::Radians>, native_matrix_t<double, 2, 3>> {1., 2, 3, pi/6, pi/3, pi/4},
    [](const auto& x, std::size_t i, std::size_t j){ return x * (j + 1); }),
    make_native_matrix<double, 3, 3>(
      1, 4, 9,
      std::sqrt(3)/2, 1, std::sqrt(2)*3/2,
      0.5, std::sqrt(3), std::sqrt(2)*3/2)));
}


TEST_F(eigen3, ToEuclideanExpr_arithmetic)
{
  EXPECT_TRUE(is_near(To3 {1, 2, pi/6, pi/3, 3, 4} + To3 {1, 2, pi/6, pi/3, 3, 4}, mat4(2, 4, std::sqrt(3), 1, 1, std::sqrt(3), 6, 8)));
  EXPECT_TRUE(is_near(To3 {1, 2, pi/6, pi/3, 3, 4} - To3 {1, 2, pi/6, pi/3, 3, 4}, M4::Zero()));
  EXPECT_TRUE(is_near(To3 {1, 2, pi/6, pi/3, 3, 4} * 2, mat4(2, 4, std::sqrt(3), 1, 1, std::sqrt(3), 6, 8)));
  EXPECT_TRUE(is_near(2 * To3 {1, 2, pi/6, pi/3, 3, 4}, mat4(2, 4, std::sqrt(3), 1, 1, std::sqrt(3), 6, 8)));
  EXPECT_TRUE(is_near(To3 {1, 2, pi/6, pi/3, 3, 4} / 2, mat4(0.5, 1, std::sqrt(3)/4, 0.25, 0.25, std::sqrt(3)/4, 1.5, 2)));
  EXPECT_TRUE(is_near(-To3 {1, 2, pi/6, pi/3, 3, 4}, mat4(-1, -2, -std::sqrt(3)/2, -0.5, -0.5, -std::sqrt(3)/2, -3, -4)));
  EXPECT_TRUE(is_near(To3 {1, 2, pi/6, pi/3, 3, 4} * DiagonalMatrix {1., 2}, mat4(1, 4, std::sqrt(3)/2, 1, 0.5, std::sqrt(3), 3, 8)));
}


TEST_F(eigen3, ToEuclideanExpr_references)
{
  using C = Coefficients<angle::Radians, Axis>;
  using M2 = native_matrix_t<double, 2, 2>;
  using M3 = native_matrix_t<double, 3, 2>;
  M2 m, n;
  m << pi/6, pi/4, 1, 2;
  n << pi/4, pi/3, 3, 4;
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
