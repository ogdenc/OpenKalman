/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "covariance_tests.h"

using namespace OpenKalman;

using Mat2 = Eigen::Matrix<double, 2, 2>;
using Mat3 = Eigen::Matrix<double, 3, 3>;

TEST_F(covariance_tests, Distribution_construction_angle)
{
  using C = Coefficients<Angle, Axis, Axis>;
  Mean<C> x_mean {M_PI / 3, 10, 5};
  Covariance<C> c;
  c << 9, 6, 3,
       6, 29, 4.5,
       3, 4.5, 65.25;
  auto c_chol = make_Covariance<C, TriangleType::lower>(Cholesky_factor(base_matrix(c)));
  GaussianDistribution dist {x_mean, c};
  GaussianDistribution dist_chol {x_mean, c_chol};
  auto x = strict_matrix(x_mean);
  auto m = strict_matrix(c);
  auto m_sqrt = strict_matrix(square_root(c_chol));
  EXPECT_TRUE(is_near(dist, dist_chol));
  EXPECT_TRUE(is_near(mean(dist), x));
  EXPECT_TRUE(is_near(Mat3 {covariance(dist)}, m));
  EXPECT_TRUE(is_near(Mat3 {covariance(dist_chol)}, m));
  EXPECT_TRUE(is_near(Mat3 {square_root(covariance(dist))}, m_sqrt));
  EXPECT_TRUE(is_near(Mat3 {square_root(covariance(dist_chol))}, m_sqrt));
  EXPECT_TRUE(is_near(Mat3 {to_Cholesky(covariance(dist))}, m));
  EXPECT_TRUE(is_near(Mat3 {from_Cholesky(covariance(dist_chol))}, m));
}

TEST_F(covariance_tests, Distribution_scale_angle)
{
  using C = Coefficients<Angle, Axis>;
  auto a = GaussianDistribution(make_Mean<C>(20., 30), make_Covariance<C>(8., 2, 2, 6));
  auto a_chol = GaussianDistribution(make_Mean<C>(20., 30), make_Covariance<C, TriangleType::lower>(8., 2, 2, 6));
  Eigen::Matrix<double, 3, 2> f_mat; f_mat << 1, 2, 3, 4, 5, 6;
  auto f_vector = make_Mean<Coefficients<Axis, Angle, Angle>>(f_mat);
  auto f_matrix = make_Matrix<Coefficients<Axis, Angle, Angle>, C>(f_mat);
  Eigen::Matrix<double, 3, 1> fa_mean; fa_mean << 80, 180, 280;
  Eigen::Matrix<double, 3, 1> fa_mean_wrapped; fa_mean_wrapped << 20-6*M_PI + 60, 180 - M_PI*58, 280 - M_PI*90;
  Eigen::Matrix<double, 3, 3> faf_cov; faf_cov << 40, 92, 144, 92, 216, 340, 144, 340, 536;
  auto a_scaled3 = f_matrix * a;
  static_assert(std::is_same_v<typename decltype(a_scaled3)::Coefficients, typename decltype(f_matrix)::RowCoefficients>);
  EXPECT_TRUE(is_near(mean(a_scaled3), fa_mean_wrapped));
  EXPECT_TRUE(is_near(covariance(a_scaled3), faf_cov));
  auto a_chol_scaled3 = f_matrix * a_chol;
  static_assert(std::is_same_v<typename decltype(a_chol_scaled3)::Coefficients, typename decltype(f_matrix)::RowCoefficients>);
  EXPECT_TRUE(is_near(mean(a_chol_scaled3), fa_mean_wrapped));
  EXPECT_TRUE(is_near(covariance(a_chol_scaled3), faf_cov));
}


TEST_F(covariance_tests, Distribution_Gaussian_random_angle)
{
  using Coeffs = Coefficients<Angle, Axis, Axis>;
  using V = Mean<Coeffs>;
  Mat3 d;
  d << 0.9, 0.1, 0.3,
       0.1, 1.4, 0.45,
       0.3, 0.45, 1.1;
  const V true_x {M_PI * 99/100, 10, 5};
  GaussianDistribution dist {true_x, make_Covariance<Coeffs>(d)};
  const V x1 {dist()};
  const V x2 {dist()};
  EXPECT_NE(x1, x2);
  using EV = EuclideanMean<Coeffs>;
  EV mean_x = EV::zero();
  for (int i = 0; i < 1000; i++)
  {
    V x {dist()};
    mean_x = (mean_x * i + to_Euclidean(x)) / (i + 1);
  }
  EXPECT_NE(from_Euclidean(mean_x), true_x);
  EXPECT_TRUE(is_near(from_Euclidean(mean_x) - true_x, V::zero(), MatrixTraits<V>::BaseMatrix::Constant(0.1)));
}

TEST_F(covariance_tests, Distribution_Gaussian_Cholesky_random_angle)
{
  using Coeffs = Coefficients<Angle, Axis, Axis>;
  using V = Mean<Coeffs>;
  Mat3 d;
  d << 0.9, 0.1, 0.3,
       0.1, 1.4, 0.45,
       0.3, 0.45, 1.1;
  const V true_x {M_PI * 99/100, 10, 5};
  GaussianDistribution dist {true_x, make_Covariance<Coeffs, TriangleType::lower>(d)};
  const V x1 {dist()};
  const V x2 {dist()};
  EXPECT_NE(x1, x2);
  using EV = EuclideanMean<Coeffs>;
  EV mean_x = EV::zero();
  for (int i = 0; i < 1000; i++)
  {
    V x {dist()};
    mean_x = (mean_x * i + to_Euclidean(x)) / (i + 1);
  }
  EXPECT_NE(from_Euclidean(mean_x), true_x);
  EXPECT_TRUE(is_near(from_Euclidean(mean_x) - true_x, V::zero(), MatrixTraits<V>::BaseMatrix::Constant(0.1)));
}

TEST_F(covariance_tests, Distribution_concatenate_axis)
{
using C2 = Coefficients<Axis, Axis>;
Mean<C2> x_mean {20, 30};
Covariance<C2> x_cov;
x_cov << 9, 3, 3, 8;
GaussianDistribution distx {x_mean, x_cov};
GaussianDistribution distx_sqrt {x_mean, to_Cholesky(x_cov)};
Mean<C2> y_mean {11, 23};
Covariance<C2> y_cov;
y_cov << 7, 1, 1, 3;
GaussianDistribution disty {y_mean, y_cov};
GaussianDistribution disty_sqrt {y_mean, to_Cholesky(y_cov)};
Mean<Coefficients<Axis, Axis, Axis, Axis>> z_mean {20, 30, 11, 23};
Eigen::Matrix<double, 4, 4> z_cov;
z_cov <<
9, 3, 0, 0,
3, 8, 0, 0,
0, 0, 7, 1,
0, 0, 1, 3;
GaussianDistribution distz {z_mean, Covariance(z_cov)};
GaussianDistribution distz_sqrt {z_mean, to_Cholesky(Covariance(z_cov))};
EXPECT_TRUE(is_near(concatenate(distx, disty), distz));
EXPECT_TRUE(is_near(concatenate(distx_sqrt, disty_sqrt), distz_sqrt));
EXPECT_TRUE(is_near(split<C2, C2>(distz), std::tuple(distx, disty)));
EXPECT_TRUE(is_near(split<C2, C2>(distz_sqrt), std::tuple(distx_sqrt, disty_sqrt)));
}


TEST_F(covariance_tests, Distribution_construction_axis)
{
Mean<Coefficients<Axis, Axis>> x_mean {20, 30};
Mat2 d, d2;
d << 3, 0,
1, 3;
d2 << 9, 3,
3, 10;
GaussianDistribution dist {x_mean, EigenSelfAdjointMatrix(d2)};
EXPECT_TRUE(is_near(mean(dist), x_mean));
EXPECT_TRUE(is_near(covariance(dist), d2));
EXPECT_TRUE(is_near(square_root(covariance(dist)), d));
GaussianDistribution dist_chol {x_mean, EigenTriangularMatrix(d)};
EXPECT_TRUE(is_near(mean(dist_chol), x_mean));
EXPECT_TRUE(is_near(covariance(dist_chol), d2));
EXPECT_TRUE(is_near(square_root(covariance(dist_chol)), d));
}


TEST_F(covariance_tests, Distribution_addition_subtraction_axis)
{
Mean<Coefficients<Axis, Axis>> x_mean {20, 30};
Mat2 d;
d << 9, 3,
3, 8;
GaussianDistribution dist1 {x_mean, EigenSelfAdjointMatrix(d)};
Mean<Coefficients<Axis, Axis>> y_mean {11, 23};
Mat2 e;
e << 7, 1,
1, 3;
GaussianDistribution dist2 {y_mean, EigenSelfAdjointMatrix(e)};
auto sum1 = dist1 + dist2;
EXPECT_TRUE(is_near(mean(sum1), Mean {31., 53}));
EXPECT_TRUE(is_near(covariance(sum1), Covariance {16., 4, 4, 11}));
auto diff1 = dist1 - dist2;
EXPECT_TRUE(is_near(mean(diff1), Mean {9., 7.}));
EXPECT_TRUE(is_near(covariance(diff1), Covariance {2., 2, 2, 5}));
GaussianDistribution dist1_chol {x_mean, EigenTriangularMatrix(EigenSelfAdjointMatrix(d))};
GaussianDistribution dist2_chol {y_mean, EigenTriangularMatrix(EigenSelfAdjointMatrix(e))};
auto sum2 = dist1_chol + dist2_chol;
EXPECT_TRUE(is_near(mean(sum2), Mean {31., 53}));
EXPECT_TRUE(is_near(covariance(sum2), Covariance {16., 4, 4, 11}));
auto diff2 = dist1_chol - dist2_chol;
EXPECT_TRUE(is_near(mean(diff2), Mean {9., 7.}));
EXPECT_TRUE(is_near(covariance(diff2), Covariance {2., 2, 2, 5}));
auto sum3 = dist1 + dist2_chol;
EXPECT_TRUE(is_near(mean(sum3), Mean {31., 53}));
EXPECT_TRUE(is_near(covariance(sum3), Covariance {16., 4, 4, 11}));
auto diff3 = dist1 - dist2_chol;
EXPECT_TRUE(is_near(mean(diff3), Mean {9., 7.}));
EXPECT_TRUE(is_near(covariance(diff3), Covariance {2., 2, 2, 5}));
auto sum4 = dist1_chol + dist2;
EXPECT_TRUE(is_near(mean(sum4), Mean {31., 53}));
EXPECT_TRUE(is_near(covariance(sum4), Covariance {16., 4, 4, 11}));
auto diff4 = dist1_chol - dist2;
EXPECT_TRUE(is_near(mean(diff4), Mean {9., 7.}));
EXPECT_TRUE(is_near(covariance(diff4), Covariance {2., 2, 2, 5}));
}


TEST_F(covariance_tests, Distribution_scale_axis)
{
auto a = GaussianDistribution(Mean{20., 30}, Covariance{8., 2, 2, 6});
auto a_chol = GaussianDistribution(Mean{20., 30}, to_Cholesky(covariance(a)));
Eigen::Matrix<double, 2, 1> mean_mat {20, 30};
Eigen::Matrix<double, 2, 2> cov_mat; cov_mat << 8, 2, 2, 6;
decltype(a) a_scaled = a * 2;
EXPECT_TRUE(is_near(mean(a_scaled), mean_mat * 2));
EXPECT_TRUE(is_near(covariance(a_scaled), cov_mat * 4));
decltype(a_chol) a_chol_scaled = a_chol * 2;
EXPECT_TRUE(is_near(mean(a_chol_scaled), mean_mat * 2));
EXPECT_TRUE(is_near(covariance(a_chol_scaled), cov_mat * 4));
a_scaled = 2 * a;
EXPECT_TRUE(is_near(mean(a_scaled), 2 * mean_mat));
EXPECT_TRUE(is_near(covariance(a_scaled), 4 * cov_mat));
a_chol_scaled = 2 * a_chol;
EXPECT_TRUE(is_near(mean(a_chol_scaled), 2 * mean_mat));
EXPECT_TRUE(is_near(covariance(a_chol_scaled), 4 * cov_mat));
a_scaled = a / 2;
EXPECT_TRUE(is_near(mean(a_scaled), mean_mat / 2));
EXPECT_TRUE(is_near(covariance(a_scaled), cov_mat / 4));
a_chol_scaled = a_chol / 2;
EXPECT_TRUE(is_near(mean(a_chol_scaled), mean_mat / 2));
EXPECT_TRUE(is_near(covariance(a_chol_scaled), cov_mat / 4));
}


TEST_F(covariance_tests, Distribution_Gaussian_random_axis)
{
using Mat = Mean<Coefficients<Axis, Axis>>;
const Mat true_x {20, 30};
GaussianDistribution dist {true_x, Covariance(9., 3, 3, 10)};
const Mat x1 {dist()};
const Mat x2 {dist()};
EXPECT_NE(x1, x2);
using EMat = EuclideanMean<Coefficients<Axis, Axis>>;
EMat mean_x = EMat::zero();
for (int i = 0; i < 100; i++)
{
Mat x {dist()};
mean_x = (mean_x * i + to_Euclidean(x)) / (i + 1);
}
EXPECT_NE(from_Euclidean(mean_x), true_x);
EXPECT_TRUE(is_near(from_Euclidean(mean_x), true_x, MatrixTraits<Mat>::BaseMatrix::Constant(0.5)));
}


TEST_F(covariance_tests, Distribution_Gaussian_Cholesky_random_axis)
{
using Mat = Mean<Coefficients<Axis, Axis>>;
Mat2 d;
d << 3, 0,
1, 3;
const Mat true_x {20, 30};
GaussianDistribution dist {true_x, EigenTriangularMatrix(d)};
const Mat x1 {dist()};
const Mat x2 {dist()};
EXPECT_NE(x1, x2);
using EMat = EuclideanMean<Coefficients<Axis, Axis>>;
EMat mean_x = EMat::zero();
for (int i = 0; i < 100; i++)
{
Mat x {dist()};
mean_x = (mean_x * i + to_Euclidean(x)) / (i + 1);
}
EXPECT_NE(from_Euclidean(mean_x), true_x);
EXPECT_TRUE(is_near(from_Euclidean(mean_x), true_x, MatrixTraits<Mat>::BaseMatrix::Constant(0.5)));
}
