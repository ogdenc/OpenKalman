/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \details Test data from Gustafsson & Hendeby. Some Relations Between Extended and Unscented Kalman Filters.
 * IEEE Transactions on Signal Processing, (60), 2, 545-555. 2012.
 */

#include "nonlinear.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;

inline namespace
{
  using M2 = eigen_matrix_t<double, 2, 1>;
  using SA = HermitianAdapter<eigen_matrix_t<double, 2, 2>>;
  using TR = TriangularAdapter<eigen_matrix_t<double, 2, 2>>;
  using G2 = GaussianDistribution<Polar<>, M2, SA>;
  using G2T = GaussianDistribution<Polar<>, M2, TR>;
}


TEST(nonlinear, CTRadar1SelfAdjoint)
{
  SamplePointsTransform<CubaturePoints> t;
  auto in = G2 {{3.0, 0.0}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.73, 1e-2);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 2.6, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 4.39, 1e-2);
}

TEST(nonlinear, CTRadar1Triangular)
{
  SamplePointsTransform<CubaturePoints> t;
  auto in = G2T {{3.0, 0.0}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.73, 1e-2);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 2.6, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 4.39, 1e-2);
}

TEST(nonlinear, CTRadar2SelfAdjoint)
{
  SamplePointsTransform<CubaturePoints> t;
  auto in = G2 {{3.0, 0.5}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.52, 1e-2);
  EXPECT_NEAR(mean_of(out)(1), 0.83, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.01, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,1), -0.75, 1e-2);
  EXPECT_NEAR(covariance_of(out)(1,0), -0.75, 1e-2);
  EXPECT_NEAR(covariance_of(out)(1,1), 3.98, 1e-2);
}

TEST(nonlinear, CTRadar2Triangular)
{
  SamplePointsTransform<CubaturePoints> t;
  auto in = G2T {{3.0, 0.5}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.52, 1e-2);
  EXPECT_NEAR(mean_of(out)(1), 0.83, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.01, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,1), -0.75, 1e-2);
  EXPECT_NEAR(covariance_of(out)(1,0), -0.75, 1e-2);
  EXPECT_NEAR(covariance_of(out)(1,1), 3.98, 1e-2);
}

TEST(nonlinear, CTRadar3SelfAdjoint)
{
  SamplePointsTransform<CubaturePoints> t;
  auto in = G2 {{3.0, 0.8}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.21, 1e-2);
  EXPECT_NEAR(mean_of(out)(1), 1.24, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.52, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,1), -0.893, 1e-3);
  EXPECT_NEAR(covariance_of(out)(1,0), -0.893, 1e-3);
  EXPECT_NEAR(covariance_of(out)(1,1), 3.47, 1e-2);
}

TEST(nonlinear, CTRadar3Triangular)
{
  SamplePointsTransform<CubaturePoints> t;
  auto in = G2T {{3.0, 0.8}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.21, 1e-2);
  EXPECT_NEAR(mean_of(out)(1), 1.24, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.52, 1e-2);
  EXPECT_NEAR(covariance_of(out)(0,1), -0.893, 1e-3);
  EXPECT_NEAR(covariance_of(out)(1,0), -0.893, 1e-3);
  EXPECT_NEAR(covariance_of(out)(1,1), 3.47, 1e-2);
}

