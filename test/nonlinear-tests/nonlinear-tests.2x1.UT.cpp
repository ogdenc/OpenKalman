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
#include <transforms/sample-points/SigmaPointsTypes/Unscented.h>

using namespace OpenKalman;

/*
 * Test data from Gustafsson & Hendeby. Some Relations Between Extended and Unscented Kalman Filters.
 * IEEE Transactions on Signal Processing, (60), 2, 545-555. 2012.
 */

Unscented<GaussianDistribution, double, Axes<2>> UT1_2 {1, 0, 3 - 2};
Unscented<GaussianDistribution, double, Axes<3>> UT1_3 {1, 0, 3 - 3};
Unscented<GaussianDistribution, double, Axes<5>> UT1_5 {1, 0, 3 - 5};
Unscented<GaussianDistribution, double, Axes<2>> UT2_2 {1e-3, 2, 0};
Unscented<GaussianDistribution, double, Axes<3>> UT2_3 {1e-3, 2, 0};
Unscented<GaussianDistribution, double, Axes<5>> UT2_5 {1e-3, 2, 0};

TEST_F(nonlinear_tests, UT1SumOfSquares2)
{
  constexpr int n = 2;
  SamplePointsTransform t {UT1_2, sum_of_squares<double, n>};
  const Eigen::Matrix<double, n, 1> mu = Eigen::Matrix<double, n, 1>::Zero();
  const Eigen::Matrix<double, n, n> P = Eigen::Matrix<double, n, n>::Identity();
  doReduction(t, mu, P, (double) n, (double) (3 - n) * n, 1e-6, 1e-6);
}

TEST_F(nonlinear_tests, UT1SumOfSquares5)
{
  constexpr int n = 5;
  SamplePointsTransform t {UT1_5, sum_of_squares<double, n>};
  const Eigen::Matrix<double, n, 1> mu = Eigen::Matrix<double, n, 1>::Zero();
  const Eigen::Matrix<double, n, n> P = Eigen::Matrix<double, n, n>::Identity();
  doReduction(t, mu, P, (double) n, (double) (3 - n) * n, 1e-6, 1e-6);
}

TEST_F(nonlinear_tests, UT2SumOfSquares2)
{
  constexpr int n = 2;
  SamplePointsTransform t {UT2_2, sum_of_squares<double, n>};
  const Eigen::Matrix<double, n, 1> mu = Eigen::Matrix<double, n, 1>::Zero();
  const Eigen::Matrix<double, n, n> P = Eigen::Matrix<double, n, n>::Identity();
  doReduction(t, mu, P, (double) n, (double) 2 * n * n, 1e-6, 1e-6);
}


TEST_F(nonlinear_tests, UT2SumOfSquares5)
{
  constexpr int n = 5;
  SamplePointsTransform t {UT2_5, sum_of_squares<double, n>};
  const Eigen::Matrix<double, n, 1> mu = Eigen::Matrix<double, n, 1>::Zero();
  const Eigen::Matrix<double, n, n> P = Eigen::Matrix<double, n, n>::Identity();
  doReduction(t, mu, P, (double) n, (double) 2 * n * n, 1e-6, 1e-6);
}

TEST_F(nonlinear_tests, UT1TOA2)
{
  constexpr int n = 2;
  SamplePointsTransform t {UT1_2, time_of_arrival<double, n>};
  Eigen::Matrix<double, n, 1> mu = Eigen::Matrix<double, n, 1>::Zero();
  mu(0) = 3;
  Eigen::Matrix<double, n, n> P = Eigen::Matrix<double, n, n>::Identity();
  P *= 10;
  P(0, 0) = 1;
  doReduction(t, mu, P, 4.08, 3.34, 1e-2, 1e-2);
}

TEST_F(nonlinear_tests, UT1TOA3)
{
  constexpr int n = 3;
  SamplePointsTransform t {UT1_3, time_of_arrival<double, n>};
  Eigen::Matrix<double, n, 1> mu = Eigen::Matrix<double, n, 1>::Zero();
  mu(0) = 3;
  Eigen::Matrix<double, n, n> P = Eigen::Matrix<double, n, n>::Identity();
  P *= 10;
  P(0, 0) = 1;
  doReduction(t, mu, P, 5.16, 3.34, 1e-2, 1e-2);
}

TEST_F(nonlinear_tests, UT2TOA2)
{
  constexpr int n = 2;
  SamplePointsTransform t {UT2_2, time_of_arrival<double, n>};
  Eigen::Matrix<double, n, 1> mu = Eigen::Matrix<double, n, 1>::Zero();
  mu(0) = 3;
  Eigen::Matrix<double, n, n> P = Eigen::Matrix<double, n, n>::Identity();
  P *= 10;
  P(0, 0) = 1;
  doReduction(t, mu, P, 4.67, 6.56, 1e-2, 1e-2);
}

TEST_F(nonlinear_tests, UT2TOA3)
{
  constexpr int n = 3;
  SamplePointsTransform t {UT2_3, time_of_arrival<double, n>};
  Eigen::Matrix<double, n, 1> mu = Eigen::Matrix<double, n, 1>::Zero();
  mu(0) = 3;
  Eigen::Matrix<double, n, n> P = Eigen::Matrix<double, n, n>::Identity();
  P *= 10;
  P(0, 0) = 1;
  doReduction(t, mu, P, 6.33, 23.2, 1e-2, 1e-1);
}

