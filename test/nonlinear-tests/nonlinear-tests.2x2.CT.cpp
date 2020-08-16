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

using M2 = Eigen::Matrix<double, 2, 1>;
using SA = EigenSelfAdjointMatrix<Eigen::Matrix<double, 2, 2>>;
using TR = EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>>;
using G2 = GaussianDistribution<Axes<2>, M2, SA>;
using G2T = GaussianDistribution<Axes<2>, M2, TR>;

TEST_F(nonlinear_tests, CTRadar1SelfAdjoint)
{
  auto t = make_SamplePointsTransform<CubaturePoints>(radar);
  auto in = G2 {{3.0, 0.0}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.73, 1e-2);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 2.6, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 4.39, 1e-2);
}

TEST_F(nonlinear_tests, CTRadar1Triangular)
{
  auto t = make_SamplePointsTransform<CubaturePoints>(radar);
  auto in = G2T {{3.0, 0.0}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.73, 1e-2);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 2.6, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 4.39, 1e-2);
}

TEST_F(nonlinear_tests, CTRadar2SelfAdjoint)
{
  auto t = make_SamplePointsTransform<CubaturePoints>(radar);
  auto in = G2 {{3.0, 0.5}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.52, 1e-2);
  EXPECT_NEAR(mean(out)(1), 0.83, 1e-2);
  EXPECT_NEAR(covariance(out)(0,0), 3.01, 1e-2);
  EXPECT_NEAR(covariance(out)(0,1), -0.75, 1e-2);
  EXPECT_NEAR(covariance(out)(1,0), -0.75, 1e-2);
  EXPECT_NEAR(covariance(out)(1,1), 3.98, 1e-2);
}

TEST_F(nonlinear_tests, CTRadar2Triangular)
{
  auto t = make_SamplePointsTransform<CubaturePoints>(radar);
  auto in = G2T {{3.0, 0.5}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.52, 1e-2);
  EXPECT_NEAR(mean(out)(1), 0.83, 1e-2);
  EXPECT_NEAR(covariance(out)(0,0), 3.01, 1e-2);
  EXPECT_NEAR(covariance(out)(0,1), -0.75, 1e-2);
  EXPECT_NEAR(covariance(out)(1,0), -0.75, 1e-2);
  EXPECT_NEAR(covariance(out)(1,1), 3.98, 1e-2);
}

TEST_F(nonlinear_tests, CTRadar3SelfAdjoint)
{
  auto t = make_SamplePointsTransform<CubaturePoints>(radar);
  auto in = G2 {{3.0, 0.8}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.21, 1e-2);
  EXPECT_NEAR(mean(out)(1), 1.24, 1e-2);
  EXPECT_NEAR(covariance(out)(0,0), 3.52, 1e-2);
  EXPECT_NEAR(covariance(out)(0,1), -0.893, 1e-3);
  EXPECT_NEAR(covariance(out)(1,0), -0.893, 1e-3);
  EXPECT_NEAR(covariance(out)(1,1), 3.47, 1e-2);
}

TEST_F(nonlinear_tests, CTRadar3Triangular)
{
  auto t = make_SamplePointsTransform<CubaturePoints>(radar);
  auto in = G2T {{3.0, 0.8}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.21, 1e-2);
  EXPECT_NEAR(mean(out)(1), 1.24, 1e-2);
  EXPECT_NEAR(covariance(out)(0,0), 3.52, 1e-2);
  EXPECT_NEAR(covariance(out)(0,1), -0.893, 1e-3);
  EXPECT_NEAR(covariance(out)(1,0), -0.893, 1e-3);
  EXPECT_NEAR(covariance(out)(1,1), 3.47, 1e-2);
}

