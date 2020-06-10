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
#include "transforms/classes/LinearizedTransform.h"
#include "distributions/DistributionTraits.h"


/*
 * Test data from Gustafsson & Hendeby. Some Relations Between Extended and Unscented Kalman Filters.
 * IEEE Transactions on Signal Processing, (60), 2, 545-555. 2012.
 */

TEST_F(nonlinear_tests, TT1SumOfSquares2)
{
  constexpr int n = 2;
  LinearizedTransform t {LinearizedTransformation<double, Axes<n>, Coefficients<Axis>, NoiseType::none, 1> {
      sum_of_squares<double, n>}};
  const Eigen::Matrix<double, n, 1> mu = Eigen::Matrix<double, n, 1>::Zero();
  const Eigen::Matrix<double, n, n> P = Eigen::Matrix<double, n, n>::Identity();
  doReduction(t, mu, P, 0., 0., 1e-6, 1e-6);
}

TEST_F(nonlinear_tests, TT1SumOfSquares5)
{
  constexpr int n = 5;
  LinearizedTransform t {LinearizedTransformation<double, Axes<n>, Coefficients<Axis>, NoiseType::none, 1> {
      sum_of_squares<double, n>}};
  const Eigen::Matrix<double, n, 1> mu = Eigen::Matrix<double, n, 1>::Zero();
  const Eigen::Matrix<double, n, n> P = Eigen::Matrix<double, n, n>::Identity();
  doReduction(t, mu, P, 0., 0., 1e-6, 1e-6);
}

TEST_F(nonlinear_tests, TT2SumOfSquares2)
{
  constexpr int n = 2;
  LinearizedTransform t {sum_of_squares<double, n>};
  const Eigen::Matrix<double, n, 1> mu = Eigen::Matrix<double, n, 1>::Zero();
  const Eigen::Matrix<double, n, n> P = Eigen::Matrix<double, n, n>::Identity();
  doReduction(t, mu, P, (double) n, (double) 2 * n, 1e-6, 1e-6);
}

TEST_F(nonlinear_tests, TT2SumOfSquares5)
{
  constexpr int n = 5;
  LinearizedTransform t {sum_of_squares<double, n>};
  const Eigen::Matrix<double, n, 1> mu = Eigen::Matrix<double, n, 1>::Zero();
  const Eigen::Matrix<double, n, n> P = Eigen::Matrix<double, n, n>::Identity();
  doReduction(t, mu, P, (double) n, (double) 2 * n, 1e-6, 1e-6);
}


TEST_F(nonlinear_tests, TT1TOA2)
{
  constexpr int n = 2;
  LinearizedTransform t {LinearizedTransformation<double, Axes<n>, Coefficients<Axis>, NoiseType::none, 1> {
      time_of_arrival<double, n>}};
  Eigen::Matrix<double, n, 1> mu_x = Eigen::Matrix<double, n, 1>::Zero();
  mu_x(0) = 3;
  Eigen::Matrix<double, n, n> P_xx = Eigen::Matrix<double, n, n>::Identity();
  P_xx *= 10;
  P_xx(0, 0) = 1;
  doReduction(t, mu_x, P_xx, 3., 1., 1e-6, 1e-6);
}

TEST_F(nonlinear_tests, TT2TOA2)
{
  constexpr int n = 2;
  LinearizedTransform t {time_of_arrival<double, n>};
  Eigen::Matrix<double, n, 1> mu_x = Eigen::Matrix<double, n, 1>::Zero();
  mu_x(0) = 3;
  Eigen::Matrix<double, n, n> P_xx = Eigen::Matrix<double, n, n>::Identity();
  P_xx *= 10;
  P_xx(0, 0) = 1;
  doReduction(t, mu_x, P_xx, 4.67, 6.56, 1e-2, 1e-2);
}

TEST_F(nonlinear_tests, TT1TOA3)
{
  constexpr int n = 3;
  LinearizedTransform t {LinearizedTransformation<double, Axes<n>, Coefficients<Axis>, NoiseType::none, 1> {
      time_of_arrival<double, n>}};
  Eigen::Matrix<double, n, 1> mu_x = Eigen::Matrix<double, n, 1>::Zero();
  mu_x(0) = 3;
  Eigen::Matrix<double, n, n> P_xx = Eigen::Matrix<double, n, n>::Identity();
  P_xx *= 10;
  P_xx(0, 0) = 1;
  doReduction(t, mu_x, P_xx, 3., 1., 1e-6, 1e-6);
}

TEST_F(nonlinear_tests, TT2TOA3)
{
  constexpr int n = 3;
  LinearizedTransform t {time_of_arrival<double, n>};
  Eigen::Matrix<double, n, 1> mu_x = Eigen::Matrix<double, n, 1>::Zero();
  mu_x(0) = 3;
  Eigen::Matrix<double, n, n> P_xx = Eigen::Matrix<double, n, n>::Identity();
  P_xx *= 10;
  P_xx(0, 0) = 1;
  doReduction(t, mu_x, P_xx, 6.33, 12.1, 1e-2, 1e-1);
}
