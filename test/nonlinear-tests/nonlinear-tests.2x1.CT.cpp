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
#include <transforms/classes/SamplePointsTransform.h>
#include <transforms/sample-points/CubaturePoints.h>

using namespace OpenKalman;

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

TEST_F(nonlinear_tests, CTSumOfSquares2SelfAdjoint)
{
  constexpr std::size_t n = 2;
  auto t = make_SamplePointsTransform<CubaturePoints>(sum_of_squares<n>);
  auto in = G<n>::normal();
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 0., 1e-6);
}

TEST_F(nonlinear_tests, CTSumOfSquares2Triangular)
{
  constexpr std::size_t n = 2;
  auto t = make_SamplePointsTransform<CubaturePoints>(sum_of_squares<n>);
  auto in = GT<n>::normal();
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 0., 1e-6);
}

TEST_F(nonlinear_tests, CTSumOfSquares5SelfAdjoint)
{
  constexpr std::size_t n = 5;
  auto t = make_SamplePointsTransform<CubaturePoints>(sum_of_squares<n>);
  auto in = G<n>::normal();
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 0., 1e-6);
}

TEST_F(nonlinear_tests, CTSumOfSquares5Triangular)
{
  constexpr std::size_t n = 5;
  auto t = make_SamplePointsTransform<CubaturePoints>(sum_of_squares<n>);
  auto in = GT<n>::normal();
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance(out)(0,0), 0., 1e-6);
}

TEST_F(nonlinear_tests, CTTOA2SelfAdjoint)
{
  constexpr std::size_t n = 2;
  auto t = make_SamplePointsTransform<CubaturePoints>(time_of_arrival<n>);
  auto in = G<n> {{3., 0}, {1., 0, 0, 10}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 4.19, 1e-2);
  EXPECT_NEAR(covariance(out)(0,0), 2.42, 1e-2);
}

TEST_F(nonlinear_tests, CTTOA2Triangular)
{
  constexpr std::size_t n = 2;
  auto t = make_SamplePointsTransform<CubaturePoints>(time_of_arrival<n>);
  auto in = GT<n> {{3., 0}, {1., 0, 0, 10}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 4.19, 1e-2);
  EXPECT_NEAR(covariance(out)(0,0), 2.42, 1e-2);
}

TEST_F(nonlinear_tests, CTTOA3SelfAdjoint)
{
  constexpr std::size_t n = 3;
  auto t = make_SamplePointsTransform<CubaturePoints>(time_of_arrival<n>);
  auto in = G<n> {{3., 0, 0}, {1., 0, 0, 0, 10, 0, 0, 0, 10}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 5.16, 1e-2);
  EXPECT_NEAR(covariance(out)(0,0), 3.34, 1e-2);
}

TEST_F(nonlinear_tests, CTTOA3Triangular)
{
  constexpr std::size_t n = 3;
  auto t = make_SamplePointsTransform<CubaturePoints>(time_of_arrival<n>);
  auto in = GT<n> {{3., 0, 0}, {1., 0, 0, 0, 10, 0, 0, 0, 10}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 5.16, 1e-2);
  EXPECT_NEAR(covariance(out)(0,0), 3.34, 1e-2);
}

