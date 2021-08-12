/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "nonlinear.gtest.hpp"

/*
 * Test data from Gustafsson & Hendeby. Some Relations Between Extended and Unscented Kalman Filters.
 * IEEE Transactions on Signal Processing, (60), 2, 545-555. 2012.
 */

template<std::size_t n>
using M = eigen_matrix_t<double, n, 1>;

template<std::size_t n>
using SA = SelfAdjointMatrix<eigen_matrix_t<double, n, n>>;

template<std::size_t n>
using TR = TriangularMatrix<eigen_matrix_t<double, n, n>>;

template<std::size_t n>
using G = GaussianDistribution<Axes<n>, M<n>, SA<n>>;

template<std::size_t n>
using GT = GaussianDistribution<Axes<n>, M<n>, TR<n>>;

struct UT1p
{
  static constexpr double alpha = 1;
  static constexpr double beta = 0;
  template<int dim> static constexpr double kappa = 3 - dim;
};
using UT1 = Unscented<UT1p>;

using UT2 = UnscentedSigmaPointsStateEstimation; // alpha = 1e-3, beta = 2, kappa = 0;


TEST_F(nonlinear, UT1SumOfSquares2SelfAdjoint)
{
  constexpr std::size_t n = 2;
  auto g = sum_of_squares<n>;
  SamplePointsTransform<UT1> t;
  auto in = G<n>::normal();
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance_of(out)(0,0), (3. - n) * n, 1e-6);
}

TEST_F(nonlinear, UT1SumOfSquares2Triangular)
{
  constexpr std::size_t n = 2;
  auto g = sum_of_squares<n>;
  SamplePointsTransform<UT1> t;
  auto in = GT<n>::normal();
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance_of(out)(0,0), (3. - n) * n, 1e-6);
}

TEST_F(nonlinear, UT1SumOfSquares5SelfAdjoint)
{
  constexpr std::size_t n = 5;
  auto g = sum_of_squares<n>;
  SamplePointsTransform<UT1> t;
  auto in = G<n>::normal();
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance_of(out)(0,0), (3. - n) * n, 1e-6);
}

TEST_F(nonlinear, UT1SumOfSquares5Triangular)
{
  constexpr std::size_t n = 5;
  auto g = sum_of_squares<n>;
  SamplePointsTransform<UT1> t;
  auto in = GT<n>::normal();
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance_of(out)(0,0), (3. - n) * n, 1e-6);
}

TEST_F(nonlinear, UT2SumOfSquares2SelfAdjoint)
{
  constexpr std::size_t n = 2;
  auto g = sum_of_squares<n>;
  SamplePointsTransform<UT2> t;
  auto in = G<n>::normal();
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance_of(out)(0,0), 2. * n * n, 1e-6);
}

TEST_F(nonlinear, UT2SumOfSquares2Triangular)
{
  constexpr std::size_t n = 2;
  auto g = sum_of_squares<n>;
  SamplePointsTransform<UT2> t;
  auto in = GT<n>::normal();
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance_of(out)(0,0), 2. * n * n, 1e-6);
}

TEST_F(nonlinear, UT2SumOfSquares5SelfAdjoint)
{
  constexpr std::size_t n = 5;
  auto g = sum_of_squares<n>;
  SamplePointsTransform<UT2> t;
  auto in = G<n>::normal();
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance_of(out)(0,0), 2. * n * n, 1e-6);
}

TEST_F(nonlinear, UT2SumOfSquares5Triangular)
{
  constexpr std::size_t n = 5;
  auto g = sum_of_squares<n>;
  SamplePointsTransform<UT2> t;
  auto in = GT<n>::normal();
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), (double) n, 1e-6);
  EXPECT_NEAR(covariance_of(out)(0,0), 2. * n * n, 1e-6);
}

TEST_F(nonlinear, UT1TOA2SelfAdjoint)
{
  constexpr std::size_t n = 2;
  auto g = time_of_arrival<n>;
  SamplePointsTransform<UT1> t;
  auto in = G<n> {{3., 0}, {1., 0, 0, 10}};
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), 4.08, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.34, 1e-2);
}

TEST_F(nonlinear, UT1TOA2Triangular)
{
  constexpr std::size_t n = 2;
  auto g = time_of_arrival<n>;
  SamplePointsTransform<UT1> t;
  auto in = GT<n> {{3., 0}, {1., 0, 0, 10}};
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), 4.08, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.34, 1e-2);
}

TEST_F(nonlinear, UT1TOA3SelfAdjoint)
{
  constexpr std::size_t n = 3;
  auto g = time_of_arrival<n>;
  SamplePointsTransform<UT1> t;
  auto in = G<n> {{3., 0, 0}, {1., 0, 0, 0, 10, 0, 0, 0, 10}};
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), 5.16, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.34, 1e-2);
}

TEST_F(nonlinear, UT1TOA3Triangular)
{
  constexpr std::size_t n = 3;
  auto g = time_of_arrival<n>;
  SamplePointsTransform<UT1> t;
  auto in = GT<n> {{3., 0, 0}, {1., 0, 0, 0, 10, 0, 0, 0, 10}};
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), 5.16, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.34, 1e-2);
}

TEST_F(nonlinear, UT2TOA2SelfAdjoint)
{
  constexpr std::size_t n = 2;
  auto g = time_of_arrival<n>;
  SamplePointsTransform<UT2> t;
  auto in = G<n> {{3., 0}, {1., 0, 0, 10}};
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), 4.67, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,0), 6.56, 1e-2);
}

TEST_F(nonlinear, UT2TOA2Triangular)
{
  constexpr std::size_t n = 2;
  auto g = time_of_arrival<n>;
  SamplePointsTransform<UT2> t;
  auto in = GT<n> {{3., 0}, {1., 0, 0, 10}};
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), 4.67, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,0), 6.56, 1e-2);
}

TEST_F(nonlinear, UT2TOA3SelfAdjoint)
{
  constexpr std::size_t n = 3;
  auto g = time_of_arrival<n>;
  SamplePointsTransform<UT2> t;
  auto in = G<n> {{3., 0, 0}, {1., 0, 0, 0, 10, 0, 0, 0, 10}};
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), 6.33, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,0), 23.2, 1e-1);
}

TEST_F(nonlinear, UT2TOA3Triangular)
{
  constexpr std::size_t n = 3;
  auto g = time_of_arrival<n>;
  SamplePointsTransform<UT2> t;
  auto in = GT<n> {{3., 0, 0}, {1., 0, 0, 0, 10, 0, 0, 0, 10}};
  auto out = t(in, g);
  EXPECT_NEAR(mean_of(out)(0), 6.33, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,0), 23.2, 1e-1);
}

