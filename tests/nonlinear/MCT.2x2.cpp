/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "nonlinear.hpp"

/*
 * Test data from Gustafsson & Hendeby. Some Relations Between Extended and Unscented Kalman Filters.
 * IEEE Transactions on Signal Processing, (60), 2, 545-555. 2012.
 */

using M2 = native_matrix_t<double, 2, 1>;
using SA = SelfAdjointMatrix<native_matrix_t<double, 2, 2>>;
using TR = TriangularMatrix<native_matrix_t<double, 2, 2>>;
using G2 = GaussianDistribution<Polar<>, M2, SA>;
using G2T = GaussianDistribution<Polar<>, M2, TR>;

TEST_F(nonlinear, MCTRadarA1SelfAdjoint)
{
  MonteCarloTransform t;
  auto in = G2 {{3.0, 0.0}, SA::identity()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.8, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 2.4, 1e-1); // Original paper said 2.5. This is probably wrong.
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 4.4, 2e-1);
}

TEST_F(nonlinear, MCTRadarA1Triangular)
{
  MonteCarloTransform t;
  auto in = G2T {{3.0, 0.0}, SA::identity()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.8, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 2.4, 1e-1); // Original paper said 2.5. This is probably wrong.
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 4.4, 2e-1);
}

TEST_F(nonlinear, MCTRadarA2SelfAdjoint)
{
  MonteCarloTransform t;
  auto in = G2 {{3.0, pi/6}, SA::identity()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.6, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.9, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 2.9, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -0.8, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -0.8, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 3.9, 1e-1);
}

TEST_F(nonlinear, MCTRadarA2Triangular)
{
  MonteCarloTransform t;
  auto in = G2T {{3.0, pi/6}, SA::identity()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.6, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.9, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 2.9, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -0.8, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -0.8, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 3.9, 1e-1);
}

TEST_F(nonlinear, MCTRadarA3SelfAdjoint)
{
  MonteCarloTransform t;
  auto in = G2 {{3.0, pi/4}, SA::identity()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.3, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 1.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -1.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -1.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 3.4, 1e-1);
}

TEST_F(nonlinear, MCTRadarA3Triangular)
{
  MonteCarloTransform t;
  auto in = G2T {{3.0, pi/4}, SA::identity()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.3, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 1.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -1.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -1.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 3.4, 1e-1);
}


/*
 * Test data from Hendeby & Gustafsson, On Nonlinear Transformations of Stochastic Variables and its Application
 * to Nonlinear Filtering. International Conference on Acoustics, Speech, and Signal Processing (ICASSP). 2007:
 */

TEST_F(nonlinear, MCTRadarB1SelfAdjoint)
{
  MonteCarloTransform t;
  auto in = G2 {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 19.0, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1); // Reference says -0.1.
  EXPECT_NEAR(covariance_of(out)(0,0), 2.7, 2e-1); // Reference says 2.9
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 2e-1); // Reference says 0.3
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 2e-1); // Reference says 0.3
  EXPECT_NEAR(covariance_of(out)(1,1), 36.3, 4e-1); // Reference says 36.6
}

TEST_F(nonlinear, MCTRadarB1Triangular)
{
  MonteCarloTransform t;
  auto in = G2T {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 19.0, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1); // Reference says -0.1
  EXPECT_NEAR(covariance_of(out)(0,0), 2.7, 2e-1); // Reference says 2.9
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 2e-1); // Reference says 0.3
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 2e-1); // Reference says 0.3
  EXPECT_NEAR(covariance_of(out)(1,1), 36.3, 4e-1); // Reference says 36.6
}

TEST_F(nonlinear, MCTRadarB2SelfAdjoint)
{
  MonteCarloTransform t;
  auto in = G2 {{20.0, pi/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 16.5, 1e-1); // Reference says 16.3
  EXPECT_NEAR(mean_of(out)(1), 9.5, 1e-1); // Reference says 9.8
  EXPECT_NEAR(covariance_of(out)(0,0), 11.2, 3e-1); // Reference says 12.2
  EXPECT_NEAR(covariance_of(out)(0,1), -14.6, 3e-1); // Reference says -15.4
  EXPECT_NEAR(covariance_of(out)(1,0), -14.6, 3e-1); // Reference says -15.4
  EXPECT_NEAR(covariance_of(out)(1,1), 27.9, 3e-1); // Reference says 27.9
}

TEST_F(nonlinear, MCTRadarB2Triangular)
{
  MonteCarloTransform t;
  auto in = G2T {{20.0, pi/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 16.5, 1e-1); // Reference says 16.3
  EXPECT_NEAR(mean_of(out)(1), 9.5, 1e-1); // Reference says 9.8
  EXPECT_NEAR(covariance_of(out)(0,0), 11.2, 3e-1); // Reference says 12.2
  EXPECT_NEAR(covariance_of(out)(0,1), -14.6, 3e-1); // Reference says -15.4
  EXPECT_NEAR(covariance_of(out)(1,0), -14.6, 3e-1); // Reference says -15.4
  EXPECT_NEAR(covariance_of(out)(1,1), 27.9, 3e-1); // Reference says 27.9
}

TEST_F(nonlinear, MCTRadarB3SelfAdjoint)
{
  MonteCarloTransform t;
  auto in = G2 {{20.0, pi/4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 13.45, 1e-1); // Reference says 13.3
  EXPECT_NEAR(mean_of(out)(1), 13.45, 1e-1); // Reference says 13.6
  EXPECT_NEAR(covariance_of(out)(0,0), 19.5, 4e-1); // Reference says 20.3
  EXPECT_NEAR(covariance_of(out)(0,1), -16.8, 2e-1); // Reference says -17.1
  EXPECT_NEAR(covariance_of(out)(1,0), -16.8, 2e-1); // Reference says -17.1
  EXPECT_NEAR(covariance_of(out)(1,1), 19.45, 3e-1); // Reference says 20.0
}

TEST_F(nonlinear, MCTRadarB3Triangular)
{
  MonteCarloTransform t;
  auto in = G2T {{20.0, pi/4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 13.45, 1e-1); // Reference says 13.3
  EXPECT_NEAR(mean_of(out)(1), 13.45, 1e-1); // Reference says 13.6
  EXPECT_NEAR(covariance_of(out)(0,0), 19.5, 3e-1); // Reference says 20.3
  EXPECT_NEAR(covariance_of(out)(0,1), -16.8, 2e-1); // Reference says -17.1
  EXPECT_NEAR(covariance_of(out)(1,0), -16.8, 2e-1); // Reference says -17.1
  EXPECT_NEAR(covariance_of(out)(1,1), 19.45, 3e-1); // Reference says 20.0
}

