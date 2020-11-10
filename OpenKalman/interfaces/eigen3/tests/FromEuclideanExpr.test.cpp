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

using M1 = Eigen::Matrix<double, 1, 1>;
using M3 = Eigen::Matrix<double, 3, 2>;
using M4 = Eigen::Matrix<double, 4, 2>;
using C = Coefficients<Axis, Angle, Axis>;
using From3 = FromEuclideanExpr<Coefficients<Axis, Angle>, M3>;
using From4 = FromEuclideanExpr<C, M4>;
using FromTo3 = FromEuclideanExpr<C, ToEuclideanExpr<C, M3>>;

template<typename...Args>
inline auto mat3(Args...args) { return MatrixTraits<M3>::make(args...); }

template<typename...Args>
inline auto mat4(Args...args) { return MatrixTraits<M4>::make(args...); }

using GetCoeff = std::function<double(const Eigen::Index)>;
inline GetCoeff g(const double x, const double y = 0, const double z = 0)
{
  return [=](const Eigen::Index i) { if (i==0) return x; else if (i==1) return y; else return z;};
}


TEST_F(eigen3, fromEuclidean_Coefficients)
{
  EXPECT_NEAR((Axis::from_Euclidean_array<double, 0>[0](g(3.))), 3., 1e-6);
  EXPECT_NEAR((Angle::from_Euclidean_array<double, 0>[0](g(0.5, std::sqrt(3) / 2))), M_PI / 3, 1e-6);

  EXPECT_NEAR((from_Euclidean<Coefficients<Axis>, double>(0, g(3.))), 3, 1e-6);
  EXPECT_NEAR((from_Euclidean<Coefficients<Angle>, double>(0, g(0.5, std::sqrt(3) / 2))), M_PI / 3, 1e-6);
  EXPECT_NEAR((from_Euclidean<Coefficients<Axis, Angle>, double>(0, g(3, 0.5, std::sqrt(3) / 2))), 3, 1e-6);
  EXPECT_NEAR((from_Euclidean<Coefficients<Axis, Angle>, double>(1, g(3, 0.5, std::sqrt(3) / 2))), M_PI / 3, 1e-6);
  EXPECT_NEAR((from_Euclidean<Coefficients<Angle, Axis>, double>(0, g(0.5, std::sqrt(3) / 2, 3))), M_PI / 3, 1e-6);
  EXPECT_NEAR((from_Euclidean<Coefficients<Angle, Axis>, double>(1, g(0.5, std::sqrt(3) / 2, 3))), 3, 1e-6);
}


TEST_F(eigen3, FromEuclideanExpr_class)
{
  M3 m;
  m << 1, 2, M_PI/6, M_PI/3, 3, 4;
  From4 d1;
  d1 << 1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4;
  EXPECT_TRUE(is_near(d1.base_matrix(), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(d1, m));
  FromTo3 d1b;
  d1b << 1, 2, M_PI/6, M_PI/3, 3, 4;
  EXPECT_TRUE(is_near(d1b.base_matrix(), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(d1b, mat3(1, 2, M_PI/6, M_PI/3, 3, 4)));
  //
  From4 d2 = (M4() << 1, 2, std::sqrt(3.)/2, 0.5, 0.5, std::sqrt(3.)/2, 3, 4).finished();
  EXPECT_TRUE(is_near(d2, m));
  From4 d3 = d2;
  EXPECT_TRUE(is_near(d3, m));
  From4 d4 = From4{1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4};
  EXPECT_TRUE(is_near(d4, m));
  From4 d5 = MatrixTraits<M4>::zero();
  EXPECT_TRUE(is_near(d5, mat3(0, 0, 0, 0, 0, 0)));
  From4 d6 = ZeroMatrix<M4>();
  EXPECT_TRUE(is_near(d6, mat3(0, 0, 0, 0, 0, 0)));
  From4 d7 = From4(ZeroMatrix<M4>());
  EXPECT_TRUE(is_near(d7, mat3(0, 0, 0, 0, 0, 0)));
  From4 d8 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4};
  EXPECT_TRUE(is_near(d8, m));
  EXPECT_TRUE(is_near(From4(ZeroMatrix<M4>()), mat3(0, 0, 0, 0, 0, 0)));
  FromTo3 d9 {1, 2, M_PI/6, M_PI/3, 3, 4};
  EXPECT_TRUE(is_near(d9.base_matrix(), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(d9, mat3(1, 2, M_PI/6, M_PI/3, 3, 4)));
  //
  d5 = d1;
  EXPECT_TRUE(is_near(d5, m));
  d6 = From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4};
  EXPECT_TRUE(is_near(d5, m));
  d7 = m;
  EXPECT_TRUE(is_near(d7, m));
  d7 = M3::Zero();
  d7 = {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4};
  EXPECT_TRUE(is_near(d7, m));
  d9 = M3::Zero();
  d9 = {1, 2, M_PI/6, M_PI/3, 3, 4};
  EXPECT_TRUE(is_near(d9, m));
  //
  d1 += d2;
  EXPECT_TRUE(is_near(d1, m + m));
  d1 -= From4 {1, 2, std::sqrt(3.)/2, 0.5, 0.5, std::sqrt(3.)/2, 3, 4};
  EXPECT_TRUE(is_near(d1, m));
  d1 *= 3;
  EXPECT_TRUE(is_near(d1, m * 3));
  d1 /= 3;
  EXPECT_TRUE(is_near(d1, m));
  EXPECT_TRUE(is_near(d1.zero(), M3::Zero()));
  EXPECT_TRUE(is_near(FromEuclideanExpr<Axes<2>, Eigen::Matrix<double, 2, 2>>::identity(), Eigen::Matrix<double, 2, 2>::Identity()));
  FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 1>> e1 = {3, std::sqrt(2.)/2, std::sqrt(2.)/2};
  EXPECT_EQ(e1[0], 3);
  EXPECT_NEAR(e1(1), M_PI/4, 1e-6);
  EXPECT_EQ(d1(0, 1), 2);
  EXPECT_EQ(d1(2, 1), 4);
}


TEST_F(eigen3, FromEuclideanExpr_subscripts)
{
  auto el = FromTo3 {1, 2, M_PI/6, M_PI/3, 3, 4};
  set_element(el, M_PI/2, 1, 0);
  EXPECT_NEAR(get_element(el, 1, 0), M_PI/2, 1e-8);
  set_element(el, 3.1, 2, 0);
  EXPECT_NEAR(get_element(el, 2, 0), 3.1, 1e-8);
  EXPECT_NEAR(get_element(From3 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}, 1, 1), M_PI/3, 1e-8);

  FromEuclideanExpr<Coefficients<Axes<2>>, Eigen::Matrix<double, 2, 2>> e2 = {1, 2, 3, 4};
  e2(0,0) = 5;
  EXPECT_EQ(e2(0, 0), 5);
  e2(0,1) = 6;
  EXPECT_EQ(e2(0, 1), 6);
  e2(1,0) = 7;
  EXPECT_EQ(e2(1, 0), 7);
  e2(1,1) = 8;
  EXPECT_EQ(e2(1, 1), 8);
  EXPECT_TRUE(is_near(e2, (Eigen::Matrix<double, 2, 2>() << 5, 6, 7, 8).finished()));
  EXPECT_NEAR((FromEuclideanExpr<C, Eigen::Matrix<double, 4, 1>>{1., std::sqrt(3)/2, 0.5, 3})(1), M_PI/6, 1e-6);
  EXPECT_NEAR((FromEuclideanExpr<C, Eigen::Matrix<double, 4, 1>>{1., std::sqrt(3)/2, 0.5, 3})(2), 3, 1e-6);
  EXPECT_NEAR((From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})(0, 0), 1, 1e-6);
  EXPECT_NEAR((From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})(1, 0), M_PI/6, 1e-6);
  EXPECT_NEAR((From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})(1, 1), M_PI/3, 1e-6);
  EXPECT_NEAR((From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})(2, 0), 3, 1e-6);
}


TEST_F(eigen3, FromEuclideanExpr_traits)
{
  static_assert(from_euclidean_expr<decltype(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);
  static_assert(typed_matrix_base<decltype(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);
  static_assert(not to_euclidean_expr<decltype(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);
  static_assert(not eigen_native<decltype(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);
  static_assert(not eigen_matrix<decltype(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);
  static_assert(not identity_matrix<decltype(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);
  static_assert(not zero_matrix<decltype(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4})>);
  // MatrixTraits
  EXPECT_TRUE(is_near(MatrixTraits<From4>::make((Eigen::Matrix<double, 4, 2>() << 1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4).finished()),
    mat3(1, 2, M_PI/6, M_PI/3, 3, 4)));
  EXPECT_TRUE(is_near(MatrixTraits<From4>::make(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4),
    mat3(1, 2, M_PI/6, M_PI/3, 3, 4)));
  EXPECT_TRUE(is_near(MatrixTraits<From4>::zero(), Eigen::Matrix<double, 3, 2>::Zero()));
  EXPECT_TRUE(is_near(FromEuclideanExpr<Axes<2>, Eigen::Matrix<double, 2, 2>>::identity(), Eigen::Matrix<double, 2, 2>::Identity()));
}


TEST_F(eigen3, FromEuclideanExpr_overloads)
{
  EXPECT_TRUE(is_near(base_matrix(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(make_native_matrix(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), mat3(1, 2, M_PI/6, M_PI/3, 3, 4)));
  EXPECT_TRUE(is_near(make_self_contained(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), mat3(1, 2, M_PI/6, M_PI/3, 3, 4)));
  EXPECT_TRUE(is_near(to_Euclidean(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(to_Euclidean<C>(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), mat4(1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4)));
  EXPECT_TRUE(is_near(to_diagonal(FromEuclideanExpr<C, Eigen::Matrix<double, 4, 1>>{1., std::sqrt(3)/2, 0.5, 3}), DiagonalMatrix {1, M_PI/6, 3}));
  EXPECT_TRUE(is_near(transpose(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), (Eigen::Matrix<double, 2, 3>() << 1, M_PI/6, 3, 2, M_PI/3, 4).finished()));
  EXPECT_TRUE(is_near(adjoint(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), (Eigen::Matrix<double, 2, 3>() << 1, M_PI/6, 3, 2, M_PI/3, 4).finished()));
  EXPECT_NEAR(determinant(From3 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}), 0.0, 1e-6);
  EXPECT_NEAR(trace(From3 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}), 1 + M_PI/3, 1e-6);
  EXPECT_TRUE(is_near(solve(
    From3 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2},
    (Eigen::Matrix<double, 2, 1>() << 5, M_PI*5/6).finished()),
    (Eigen::Matrix<double, 2, 1>() << 1, 2).finished()));
  EXPECT_TRUE(is_near(reduce_columns(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}), (Eigen::Matrix<double, 3, 1>() << 1.5, M_PI/4, 3.5).finished()));
  EXPECT_TRUE(is_near(LQ_decomposition(From3 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}),
    LQ_decomposition((Eigen::Matrix<double, 2, 2>() << 1, 2, M_PI/6, M_PI/3).finished())));
  EXPECT_TRUE(is_near(QR_decomposition(From3 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}),
    QR_decomposition((Eigen::Matrix<double, 2, 2>() << 1, 2, M_PI/6, M_PI/3).finished())));

  using N = std::normal_distribution<double>::param_type;
  auto m = make_native_matrix(MatrixTraits<Eigen::Matrix<double, 4, 2>>::zero());
  for (int i=0; i<100; i++)
  {
    m = (m * i + to_Euclidean(randomize<From4>(1.0, 0.3))) / (i + 1);
  }
  auto offset = Eigen::Matrix<double, 4, 2>::Constant(1);
  EXPECT_TRUE(is_near(m, offset, 0.1));
  EXPECT_FALSE(is_near(m, offset, 1e-6));

  for (int i=0; i<100; i++)
  {
    m = (m * i + to_Euclidean(randomize<From4>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3}))) / (i + 1);
  }
  auto offset2 = to_Euclidean(From4 {1., 1., 2., 2., 3., 3., 4., 4.});
  EXPECT_TRUE(is_near(m, offset2, 0.1));
  EXPECT_FALSE(is_near(m, offset2, 1e-6));

  for (int i=0; i<100; i++)
  {
    m = (m * i + to_Euclidean(randomize<From4>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3},
      N {5.0, 0.3}, 6.0, N {7.0, 0.3}, N {8.0, 0.3}))) / (i + 1);
  }
  auto offset3 = to_Euclidean(From4 {1., 2., 3., 4., 5., 6., 7., 8.});
  EXPECT_TRUE(is_near(m, offset3, 0.1));
  EXPECT_FALSE(is_near(m, offset3, 1e-6));
}


TEST_F(eigen3, FromEuclideanExpr_blocks)
{
  EXPECT_TRUE(is_near(concatenate_vertical(
    FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 3>> {1, 2, 3,
                                                                               std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                                               0.5, std::sqrt(3)/2, std::sqrt(2)/2},
    FromEuclideanExpr<Coefficients<Angle, Axis>, Eigen::Matrix<double, 3, 3>> {std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                                                                               std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
                                                                               4, 5, 6}),
    (Eigen::Matrix<double, 4, 3>() <<
      1., 2, 3,
      M_PI/6, M_PI/3, M_PI/4,
      M_PI/4, M_PI/3, M_PI/6,
      4, 5, 6).finished()));
  EXPECT_TRUE(is_near(concatenate_horizontal(
    FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 3>> {1, 2, 3,
                                                                               std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                                               0.5, std::sqrt(3)/2, std::sqrt(2)/2},
    FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 3>> {4, 5, 6,
                                                                               std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
                                                                               std::sqrt(2)/2, std::sqrt(3)/2, 0.5}),
    (Eigen::Matrix<double, 2, 6>() <<
      1, 2, 3, 4, 5, 6,
      M_PI/6, M_PI/3, M_PI/4, M_PI/4, M_PI/3, M_PI/6).finished()));
  EXPECT_TRUE(is_near(split_vertical(FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 2>> {
      1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}), std::tuple{}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(
    FromEuclideanExpr<Coefficients<Axis, Angle, Angle, Axis>, Eigen::Matrix<double, 6, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2,
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
      4, 5, 6}),
    std::tuple{(Eigen::Matrix<double, 2, 3>() << 1., 2, 3, M_PI/6, M_PI/3, M_PI/4).finished(),
               (Eigen::Matrix<double, 2, 3>() << M_PI/4, M_PI/3, M_PI/6, 4, 5, 6).finished()}));
  EXPECT_TRUE(is_near(split_vertical<2, 1>(
    FromEuclideanExpr<Coefficients<Axis, Angle, Angle, Axis>, Eigen::Matrix<double, 6, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2,
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
      4, 5, 6}),
    std::tuple{(Eigen::Matrix<double, 2, 3>() << 1., 2, 3, M_PI/6, M_PI/3, M_PI/4).finished(),
               (Eigen::Matrix<double, 1, 3>() << M_PI/4, M_PI/3, M_PI/6).finished()}));
  EXPECT_TRUE(is_near(split_vertical<Coefficients<Axis, Angle>, Coefficients<Angle, Axis>>(
    FromEuclideanExpr<Coefficients<Axis, Angle, Angle, Axis>, Eigen::Matrix<double, 6, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2,
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
      4, 5, 6}),
    std::tuple{(Eigen::Matrix<double, 2, 3>() << 1., 2, 3, M_PI/6, M_PI/3, M_PI/4).finished(),
               (Eigen::Matrix<double, 2, 3>() << M_PI/4, M_PI/3, M_PI/6, 4, 5, 6).finished()}));
  EXPECT_TRUE(is_near(
    split_vertical<Coefficients<Axis, Angle>, Coefficients<Angle, Axis>>(
      from_Euclidean<Coefficients<Axis, Angle, Angle, Axis>>(
        to_Euclidean<Coefficients<Axis, Angle, Angle, Axis>>(
          (Eigen::Matrix<double, 4, 3>() << 1., 2, 3, M_PI/6, M_PI/3, M_PI/4, M_PI/4, M_PI/3, M_PI/6, 4, 5, 6).finished()
      ))),
    std::tuple{
      (Eigen::Matrix<double, 2, 3>() << 1., 2, 3, M_PI/6, M_PI/3, M_PI/4).finished(),
      (Eigen::Matrix<double, 2, 3>() << M_PI/4, M_PI/3, M_PI/6, 4, 5, 6).finished()
      }));
  EXPECT_TRUE(is_near(split_vertical<Coefficients<Axis, Angle>, Coefficients<Angle>>(
    FromEuclideanExpr<Coefficients<Axis, Angle, Angle, Axis>, Eigen::Matrix<double, 6, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2,
      std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      std::sqrt(2)/2, std::sqrt(3)/2, 0.5,
      4, 5, 6}),
    std::tuple{(Eigen::Matrix<double, 2, 3>() << 1., 2, 3, M_PI/6, M_PI/3, M_PI/4).finished(),
               (Eigen::Matrix<double, 1, 3>() << M_PI/4, M_PI/3, M_PI/6).finished()}));
  EXPECT_TRUE(is_near(split_horizontal(FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 2>> {
    1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2}), std::tuple{}));
  EXPECT_TRUE(is_near(split_horizontal<3, 3>(
    FromEuclideanExpr<Polar<>, const Eigen::Matrix<double, 3, 6>> {
      1, 2, 3, 4, 5, 6,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2, std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2, std::sqrt(2)/2, std::sqrt(3)/2, 0.5}),
    std::tuple{(Eigen::Matrix<double, 2, 3>() << 1., 2, 3, M_PI/6, M_PI/3, M_PI/4).finished(),
               (Eigen::Matrix<double, 2, 3>() << 4, 5, 6, M_PI/4, M_PI/3, M_PI/6).finished()}));
  auto a1 = FromEuclideanExpr<Polar<>, const Eigen::Matrix<double, 3, 6>> {
    1, 2, 3, 4, 5, 6,
    std::sqrt(3)/2, 0.5, std::sqrt(2)/2, std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
    0.5, std::sqrt(3)/2, std::sqrt(2)/2, std::sqrt(2)/2, std::sqrt(3)/2, 0.5};
  EXPECT_TRUE(is_near(split_horizontal<3, 3>(a1),
    std::tuple{(Eigen::Matrix<double, 2, 3>() << 1., 2, 3, M_PI/6, M_PI/3, M_PI/4).finished(),
               (Eigen::Matrix<double, 2, 3>() << 4, 5, 6, M_PI/4, M_PI/3, M_PI/6).finished()}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(
    FromEuclideanExpr<Polar<>, Eigen::Matrix<double, 3, 6>> {
      1, 2, 3, 4, 5, 6,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2, std::sqrt(2)/2, 0.5, std::sqrt(3)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2, std::sqrt(2)/2, std::sqrt(3)/2, 0.5}),
    std::tuple{(Eigen::Matrix<double, 2, 3>() << 1., 2, 3, M_PI/6, M_PI/3, M_PI/4).finished(),
               (Eigen::Matrix<double, 2, 2>() << 4, 5, M_PI/4, M_PI/3).finished()}));

  EXPECT_TRUE(is_near(split_diagonal<Axis, Angle>(
    FromEuclideanExpr<Coefficients<Axis, Angle>, const Eigen::Matrix<double, 3, 2>> {
      1, 2,
      std::sqrt(3)/2, 0.5,
      0.5, std::sqrt(3)/2}),
    std::tuple{(Eigen::Matrix<double, 1, 1>() << 1).finished(),
               (Eigen::Matrix<double, 1, 1>() << M_PI/6).finished()}));
  EXPECT_TRUE(is_near(split_diagonal<1, 1>(
    FromEuclideanExpr<Polar<>, const Eigen::Matrix<double, 3, 2>> {
      1, 2,
      std::sqrt(3)/2, 0.5,
      0.5, std::sqrt(3)/2}),
    std::tuple{(Eigen::Matrix<double, 1, 1>() << 1).finished(),
               (Eigen::Matrix<double, 1, 1>() << M_PI/6).finished()}));

  EXPECT_TRUE(is_near(column(
    FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 3>> {1, 2, 3,
                                                                               std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                                               0.5, std::sqrt(3)/2, std::sqrt(2)/2}, 2),
    (Eigen::Matrix<double, 2, 1>() << 3, M_PI/4).finished()));
  EXPECT_TRUE(is_near(column<1>(
    FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 3>> {1, 2, 3,
                                                                               std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                                               0.5, std::sqrt(3)/2, std::sqrt(2)/2}),
    (Eigen::Matrix<double, 2, 1>() << 2, M_PI/3).finished()));
  //
  auto b = FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 3>> {1, 2, 3,
                                                                                      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                                                      0.5, std::sqrt(3)/2, std::sqrt(2)/2};
  EXPECT_TRUE(is_near(b, (Eigen::Matrix<double, 2, 3>() << 1, 2, 3, M_PI/6, M_PI/3, M_PI/4).finished()));
  EXPECT_TRUE(is_near(apply_columnwise(b,
    [](auto& col){ col *= 3; }),
    (Eigen::Matrix<double, 2, 3>() <<
      3, 6, 9,
      M_PI/2, M_PI, M_PI*3/4).finished()));

  b = FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 3>> {1, 2, 3,
                                                                                 std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                                                 0.5, std::sqrt(3)/2, std::sqrt(2)/2};
  EXPECT_TRUE(is_near(apply_columnwise(b,
    [](auto& col, std::size_t i){ col *= i + 1; }),
    (Eigen::Matrix<double, 2, 3>() <<
      1, 4, 9,
      M_PI/6, M_PI*2/3, M_PI*3/4).finished()));

  auto f2 = [](const auto& col){ return col + col; };
  EXPECT_TRUE(is_near(apply_columnwise(
    FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 3>> {
      1, 2, 3,
      std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
      0.5, std::sqrt(3)/2, std::sqrt(2)/2},
    f2),
    (Eigen::Matrix<double, 2, 3>() <<
      2, 4, 6,
      M_PI/3, M_PI*2/3, M_PI/2).finished()));
  EXPECT_TRUE(is_near(apply_columnwise(
    FromEuclideanExpr<Coefficients<Axis, Angle>, ToEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 2, 3>>>
      {1., 2, 3, M_PI/6, M_PI/3, M_PI/4}, f2),
    (Eigen::Matrix<double, 2, 3>() << 2., 4, 6, M_PI/3, M_PI*2/3, M_PI/2).finished()));

  EXPECT_TRUE(is_near(apply_columnwise(
    FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 3>> {1, 2, 3,
                                                                               std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                                               0.5, std::sqrt(3)/2, std::sqrt(2)/2},
    [](const auto& col, std::size_t i){ return col * i; }),
    (Eigen::Matrix<double, 2, 3>() <<
      0, 2, 6,
      0, M_PI/3, M_PI/2).finished()));
  EXPECT_TRUE(is_near(apply_columnwise<3>(
    [](){ return FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 1>> {1., std::sqrt(3)/2, 0.5}; }),
    (Eigen::Matrix<double, 2, 3>() <<
      1, 1, 1,
      M_PI/6, M_PI/6, M_PI/6).finished()));
  EXPECT_TRUE(is_near(apply_columnwise<3>(
    [](std::size_t i){ return FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 1>> {1., std::sqrt(3)/2, 0.5} * (i + 1); }),
    (Eigen::Matrix<double, 2, 3>() <<
      1, 2, 3,
      M_PI/6, M_PI/3, M_PI/2).finished()));
  //
  EXPECT_TRUE(is_near(apply_coefficientwise(
    FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 3>> {1, 2, 3,
                                                                               std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                                               0.5, std::sqrt(3)/2, std::sqrt(2)/2},
    [](const auto& x){ return x * 3; }),
    (Eigen::Matrix<double, 2, 3>() <<
      3, 6, 9,
      M_PI/2, M_PI, M_PI*3/4).finished()));
  EXPECT_TRUE(is_near(apply_coefficientwise(
    FromEuclideanExpr<Coefficients<Axis, Angle>, Eigen::Matrix<double, 3, 3>> {1, 2, 3,
                                                                               std::sqrt(3)/2, 0.5, std::sqrt(2)/2,
                                                                               0.5, std::sqrt(3)/2, std::sqrt(2)/2},
    [](const auto& x, std::size_t i, std::size_t j){ return x * (j + 1); }),
    (Eigen::Matrix<double, 2, 3>() <<
      1, 4, 9,
      M_PI/6, M_PI*2/3, M_PI*3/4).finished()));
}


TEST_F(eigen3, FromEuclideanExpr_arithmetic)
{
  EXPECT_TRUE(is_near(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4} + From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}, mat3(2, 4, M_PI/3, M_PI*2/3, 6, 8)));
  EXPECT_TRUE(is_near(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4} - From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}, M3::Zero()));
  EXPECT_TRUE(is_near(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4} * 2, mat3(2, 4, M_PI/3, M_PI*2/3, 6, 8)));
  EXPECT_TRUE(is_near(2 * From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}, mat3(2, 4, M_PI/3, M_PI*2/3, 6, 8)));
  EXPECT_TRUE(is_near(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4} / 2, mat3(0.5, 1, M_PI/12, M_PI/6, 1.5, 2)));
  EXPECT_TRUE(is_near(-From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4}, mat3(-1, -2, -M_PI/6, -M_PI/3, -3, -4)));
  EXPECT_TRUE(is_near(From4 {1, 2, std::sqrt(3)/2, 0.5, 0.5, std::sqrt(3)/2, 3, 4} * DiagonalMatrix {1., 2}, mat3(1, 4, M_PI/6, M_PI*2/3, 3, 8)));
  using To3 = ToEuclideanExpr<C, M3>;
  using FromTo3 = FromEuclideanExpr<C, To3>;
  EXPECT_TRUE(is_near(FromTo3(To3 {1, 2, M_PI/6 + 2*M_PI, M_PI/3 - 6*M_PI, 3, 4}) + FromTo3(To3 {1, 2, M_PI/6, M_PI/3, 3, 4}), mat3(2, 4, M_PI/3, M_PI*2/3, 6, 8)));
  EXPECT_TRUE(is_near(FromTo3(To3 {2, 4, M_PI/3, M_PI*2/3, 6, 8}) - FromTo3(To3 {1, 2, M_PI/6 + 2*M_PI, M_PI/3 - 6*M_PI, 3, 4}), mat3(1, 2, M_PI/6, M_PI/3, 3, 4)));
  EXPECT_TRUE(is_near(-FromTo3(To3 {1, 2, M_PI/6 + 2*M_PI, M_PI/3 - 6*M_PI, 3, 4}), mat3(-1, -2, -M_PI/6, -M_PI/3, -3, -4)));
}


TEST_F(eigen3, FromEuclideanExpr_references)
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
  FromEuclideanExpr<C, M3> x = me;
  FromEuclideanExpr<C, M3&> x_lvalue = x;
  EXPECT_TRUE(is_near(x_lvalue, m));
  x = FromEuclideanExpr<C, M3>(ne);
  EXPECT_TRUE(is_near(x_lvalue, n));
  x_lvalue = FromEuclideanExpr<C, M3>(me);
  EXPECT_TRUE(is_near(x, m));
  FromEuclideanExpr<C, M3&&> x_rvalue = std::move(x);
  EXPECT_TRUE(is_near(x_rvalue, m));
  x_rvalue = FromEuclideanExpr<C, M3>(ne);
  EXPECT_TRUE(is_near(x_rvalue, n));
}


TEST_F(eigen3, Wrap_angle)
{
  using R = FromEuclideanExpr<Angle, ToEuclideanExpr<Angle, M1>>;
  R x0 {M_PI_4};
  EXPECT_NEAR(get_element(x0, 0, 0), M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x0, 0), M_PI_4, 1e-6);
  set_element(x0, 5*M_PI_4, 0, 0);
  EXPECT_NEAR(get_element(x0, 0), -3*M_PI_4, 1e-6);
  set_element(x0, -7*M_PI/6, 0);
  EXPECT_NEAR(get_element(x0, 0), 5*M_PI/6, 1e-6);
}


TEST_F(eigen3, Wrap_distance)
{
  using R = FromEuclideanExpr<Distance, ToEuclideanExpr<Distance, M1>>;
  R x0 {-5};
  EXPECT_TRUE(is_near(x0 + R {1.2}, Eigen::Matrix<double, 1, 1> {6.2}));
  EXPECT_TRUE(is_near(R {R {1.1} - 3. * R {1}}, R {1.9}));
  EXPECT_TRUE(is_near(R {1.2} + R {-3}, R {4.2}));
  EXPECT_NEAR(get_element(x0, 0, 0), 5, 1e-6);
  EXPECT_NEAR(get_element(x0, 0), 5., 1e-6);
  set_element(x0, 4, 0);
  EXPECT_NEAR(get_element(x0, 0), 4., 1e-6);
  set_element(x0, -3, 0);
  EXPECT_NEAR(get_element(x0, 0), 3., 1e-6);
}


TEST_F(eigen3, Wrap_inclination)
{
  using R = FromEuclideanExpr<InclinationAngle, ToEuclideanExpr<InclinationAngle, M1>>;
  R x0 {M_PI_2};
  EXPECT_NEAR(get_element(x0, 0, 0), M_PI_2, 1e-6);
  EXPECT_NEAR(get_element(x0, 0), M_PI_2, 1e-6);
  set_element(x0, M_PI_4, 0, 0);
  EXPECT_NEAR(get_element(x0, 0, 0), M_PI_4, 1e-6);
  set_element(x0, 3*M_PI_4, 0, 0);
  EXPECT_NEAR(get_element(x0, 0, 0), M_PI_4, 1e-6);
}


TEST_F(eigen3, Wrap_polar)
{
  using C1 = Polar<Distance, Angle>;
  using P = FromEuclideanExpr<C1, ToEuclideanExpr<C1, Eigen::Matrix<double, 2, 1>>>;
  P x0 {2, M_PI_4};
  EXPECT_NEAR(get_element(x0, 0, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), M_PI_4, 1e-6);
  set_element(x0, -1.5, 0);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), -3*M_PI_4, 1e-6);
  set_element(x0, 7*M_PI/6, 1);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), -5*M_PI/6, 1e-6);

  using C2 = Polar<Angle, Distance>;
  using Q = FromEuclideanExpr<C2, ToEuclideanExpr<C2, Eigen::Matrix<double, 2, 1>>>;
  Q x1 {M_PI_4, 2};
  EXPECT_NEAR(get_element(x1, 1, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), M_PI_4, 1e-6);
  set_element(x1, -1.5, 1);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), -3*M_PI_4, 1e-6);
  set_element(x1, 7*M_PI/6, 0);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), -5*M_PI/6, 1e-6);
}


TEST_F(eigen3, Wrap_spherical)
{
  using C1 = Spherical<Distance, Angle, InclinationAngle>;
  using S = FromEuclideanExpr<C1, ToEuclideanExpr<C1, Eigen::Matrix<double, 3, 1>>>;
  S x0 {2, M_PI_4, -M_PI_4};
  EXPECT_NEAR(get_element(x0, 0, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), -M_PI_4, 1e-6);
  set_element(x0, -1.5, 0);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), -3*M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), M_PI_4, 1e-6);
  set_element(x0, 7*M_PI/6, 1);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), -5*M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), M_PI_4, 1e-6);
  set_element(x0, 3*M_PI_4, 2);
  EXPECT_NEAR(get_element(x0, 0), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x0, 1), M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x0, 2), M_PI_4, 1e-6);

  using C2 = Spherical<Angle, Distance, InclinationAngle>;
  using T = FromEuclideanExpr<C2, ToEuclideanExpr<C2, Eigen::Matrix<double, 3, 1>>>;
  T x1 {M_PI_4, 2, -M_PI_4};
  EXPECT_NEAR(get_element(x1, 1, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x1, 2), -M_PI_4, 1e-6);
  set_element(x1, -1.5, 1);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), -3*M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x1, 2), M_PI_4, 1e-6);
  set_element(x1, 7*M_PI/6, 0);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), -5*M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x1, 2), M_PI_4, 1e-6);
  set_element(x1, 3*M_PI_4, 2);
  EXPECT_NEAR(get_element(x1, 1), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x1, 0), M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x1, 2), M_PI_4, 1e-6);

  using C3 = Spherical<Angle, InclinationAngle, Distance>;
  using U = FromEuclideanExpr<C3, ToEuclideanExpr<C3, Eigen::Matrix<double, 3, 1>>>;
  U x2 {M_PI_4, -M_PI_4, 2};
  EXPECT_NEAR(get_element(x2, 2, 0), 2, 1e-6);
  EXPECT_NEAR(get_element(x2, 0), M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x2, 1), -M_PI_4, 1e-6);
  set_element(x2, -1.5, 2);
  EXPECT_NEAR(get_element(x2, 2), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x2, 0), -3*M_PI_4, 1e-6);
  EXPECT_NEAR(get_element(x2, 1), M_PI_4, 1e-6);
  set_element(x2, 7*M_PI/6, 0);
  EXPECT_NEAR(get_element(x2, 2), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x2, 0), -5*M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x2, 1), M_PI_4, 1e-6);
  set_element(x2, 3*M_PI_4, 1);
  EXPECT_NEAR(get_element(x2, 2), 1.5, 1e-6);
  EXPECT_NEAR(get_element(x2, 0), M_PI/6, 1e-6);
  EXPECT_NEAR(get_element(x2, 1), M_PI_4, 1e-6);
}
