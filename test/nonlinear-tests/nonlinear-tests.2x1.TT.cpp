/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "nonlinear-tests.h"

/*
 * Test data from Gustafsson & Hendeby. Some Relations Between Extended and Unscented Kalman Filters.
 * IEEE Transactions on Signal Processing, (60), 2, 545-555. 2012.
 */

template<std::size_t n>
using M = Eigen::Matrix<double, n, 1>;

template<std::size_t n>
using SA = EigenSelfAdjointMatrix<Eigen::Matrix<double, n, n>>;

template<std::size_t n>
using TR = EigenTriangularMatrix<Eigen::Matrix<double, n, n>>;

template<std::size_t n>
using G = GaussianDistribution<Axes<n>, M<n>, SA<n>>;

template<std::size_t n>
using GT = GaussianDistribution<Axes<n>, M<n>, TR<n>>;

TEST_F(nonlinear_tests, TT1SumOfSquares2SelfAdjoint)
{
  constexpr std::size_t n = 2;
  auto t = make_LinearizedTransform(sum_of_squares<n>);
  auto in = G<n>::normal();
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 0., 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 0., 1e-6);
}

TEST_F(nonlinear_tests, TT1SumOfSquares2Triangular)
{
  constexpr std::size_t n = 2;
  auto t = make_LinearizedTransform(sum_of_squares<n>);
  auto in = GT<n>::normal();
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 0., 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 0., 1e-6);
}

TEST_F(nonlinear_tests, TT1SumOfSquares5SelfAdjoint)
{
  constexpr std::size_t n = 5;
  auto t = make_LinearizedTransform(sum_of_squares<n>);
  auto in = G<n>::normal();
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 0., 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 0., 1e-6);
}

TEST_F(nonlinear_tests, TT1SumOfSquares5Triangular)
{
  constexpr std::size_t n = 5;
  auto t = make_LinearizedTransform(sum_of_squares<n>);
  auto in = GT<n>::normal();
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 0., 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 0., 1e-6);
}

TEST_F(nonlinear_tests, TT2SumOfSquares2SelfAdjoint)
{
  constexpr std::size_t n = 2;
  auto t = make_LinearizedTransform<2>(sum_of_squares<n>);
  auto in = G<n>::normal();
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 2. * n, 1e-6);
}

TEST_F(nonlinear_tests, TT2SumOfSquares2Triangular)
{
  constexpr std::size_t n = 2;
  auto t = make_LinearizedTransform<2>(sum_of_squares<n>);
  auto in = GT<n>::normal();
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 2. * n, 1e-6);
}

TEST_F(nonlinear_tests, TT2SumOfSquares5SelfAdjoint)
{
  constexpr std::size_t n = 5;
  auto t = make_LinearizedTransform<2>(sum_of_squares<n>);
  auto in = G<n>::normal();
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 2. * n, 1e-6);
}

TEST_F(nonlinear_tests, TT2SumOfSquares5Triangular)
{
  constexpr std::size_t n = 5;
  auto t = make_LinearizedTransform<2>(sum_of_squares<n>);
  auto in = GT<n>::normal();
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 2. * n, 1e-6);
}

TEST_F(nonlinear_tests, TT1TOA2SelfAdjoint)
{
  constexpr std::size_t n = 2;
  auto t = make_LinearizedTransform(time_of_arrival<n>);
  auto in = G<n> {{3., 0}, {1., 0, 0, 10}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 3., 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 1., 1e-6);
}

TEST_F(nonlinear_tests, TT1TOA2Triangular)
{
  constexpr std::size_t n = 2;
  auto t = make_LinearizedTransform(time_of_arrival<n>);
  auto in = GT<n> {{3., 0}, {1., 0, 0, 10}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 3., 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 1., 1e-6);
}

TEST_F(nonlinear_tests, TT2TOA2SelfAdjoint)
{
  constexpr std::size_t n = 2;
  auto t = make_LinearizedTransform<2>(time_of_arrival<n>);
  auto in = G<n> {{3., 0}, {1., 0, 0, 10}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 4.67, 1e-2);
  EXPECT_NEAR(covariance(out)(0,0), 6.56, 1e-2);
}

TEST_F(nonlinear_tests, TT2TOA2Triangular)
{
  constexpr std::size_t n = 2;
  auto t = make_LinearizedTransform<2>(time_of_arrival<n>);
  auto in = GT<n> {{3., 0}, {1., 0, 0, 10}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 4.67, 1e-2);
  EXPECT_NEAR(covariance(out)(0,0), 6.56, 1e-2);
}

TEST_F(nonlinear_tests, TT1TOA3SelfAdjoint)
{
  constexpr std::size_t n = 3;
  auto t = make_LinearizedTransform(time_of_arrival<n>);
  auto in = G<n> {{3., 0, 0}, {1., 0, 0, 0, 10, 0, 0, 0, 10}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 3., 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 1., 1e-6);
}

TEST_F(nonlinear_tests, TT1TOA3Triangular)
{
  constexpr std::size_t n = 3;
  auto t = make_LinearizedTransform(time_of_arrival<n>);
  auto in = GT<n> {{3., 0, 0}, {1., 0, 0, 0, 10, 0, 0, 0, 10}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 3., 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 1., 1e-6);
}

TEST_F(nonlinear_tests, TT2TOA3SelfAdjoint)
{
  constexpr std::size_t n = 3;
  auto t = make_LinearizedTransform<2>(time_of_arrival<n>);
  auto in = G<n> {{3., 0, 0}, {1., 0, 0, 0, 10, 0, 0, 0, 10}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 6.33, 1e-2);
  EXPECT_NEAR(covariance(out)(0,0), 12.1, 1e-1);
}

TEST_F(nonlinear_tests, TT2TOA3Triangular)
{
  constexpr std::size_t n = 3;
  auto t = make_LinearizedTransform<2>(time_of_arrival<n>);
  auto in = GT<n> {{3., 0, 0}, {1., 0, 0, 0, 10, 0, 0, 0, 10}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 6.33, 1e-2);
  EXPECT_NEAR(covariance(out)(0,0), 12.1, 1e-1);
}

