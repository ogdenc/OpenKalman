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
#include "transforms/LinearizedTransform.hpp"

using std::numbers::pi;

/*
 * Test data from Gustafsson & Hendeby. Some Relations Between Extended and Unscented Kalman Filters.
 * IEEE Transactions on Signal Processing, (60), 2, 545-555. 2012 (TT2 data appears to be in error, and is omitted).
 */

using M2 = eigen_matrix_t<double, 2, 1>;
using SA = SelfAdjointMatrix<eigen_matrix_t<double, 2, 2>>;
using TR = TriangularMatrix<eigen_matrix_t<double, 2, 2>>;
using G2 = GaussianDistribution<Polar<>, M2, SA>;
using G2T = GaussianDistribution<Polar<>, M2, TR>;

TEST_F(nonlinear, TT1RadarA1SelfAdjoint)
{
  LinearizedTransform t;
  auto in = G2 {{3.0, 0.0}, SA::identity()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 3.0, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 1.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 9.0, 1e-1);
}

TEST_F(nonlinear, TT1RadarA1Triangular)
{
  LinearizedTransform t;
  auto in = G2T {{3.0, 0.0}, SA::identity()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 3.0, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 1.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 9.0, 1e-1);
}

TEST_F(nonlinear, TT1RadarA2SelfAdjoint)
{
  LinearizedTransform t;
  auto in = G2 {{3.0, pi/6}, SA::identity()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 2.6, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 1.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -3.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -3.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 7.0, 1e-1);
}

TEST_F(nonlinear, TT1RadarA2Triangular)
{
  LinearizedTransform t;
  auto in = G2T {{3.0, pi/6}, SA::identity()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 2.6, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 1.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -3.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -3.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 7.0, 1e-1);
}

TEST_F(nonlinear, TT1RadarA3SelfAdjoint)
{
  LinearizedTransform t;
  auto in = G2 {{3.0, pi/4}, SA::identity()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 2.1, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 2.1, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 5.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -4.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -4.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 5.0, 1e-1);
}

TEST_F(nonlinear, TT1RadarA3Triangular)
{
  LinearizedTransform t;
  auto in = G2T {{3.0, pi/4}, SA::identity()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 2.1, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 2.1, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 5.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -4.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -4.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 5.0, 1e-1);
}


/*
 * Test data from Hendeby & Gustafsson, On Nonlinear Transformations of Stochastic Variables and its Application
 * to Nonlinear Filtering. International Conference on Acoustics, Speech, and Signal Processing (ICASSP). 2007:
 */

TEST_F(nonlinear, TT1RadarB1SelfAdjoint)
{
  LinearizedTransform t;
  auto in = G2 {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 20.0, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 1.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 40.0, 1e-1);
}

TEST_F(nonlinear, TT1RadarB1Triangular)
{
  LinearizedTransform t;
  auto in = G2T {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 20.0, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 1.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 40.0, 1e-1);
}

TEST_F(nonlinear, TT1RadarB2SelfAdjoint)
{
  LinearizedTransform t;
  auto in = G2 {{20.0, pi/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 17.3, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 10.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 10.7, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -16.9, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -16.9, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 30.3, 1e-1);
}

TEST_F(nonlinear, TT1RadarB2Triangular)
{
  LinearizedTransform t;
  auto in = G2T {{20.0, pi/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 17.3, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 10.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 10.7, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -16.9, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -16.9, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 30.3, 1e-1);
}

TEST_F(nonlinear, TT1RadarB3SelfAdjoint)
{
  LinearizedTransform t;
  auto in = G2 {{20.0, pi/4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 14.1, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 14.1, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 20.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -19.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -19.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 20.5, 1e-1);
}

TEST_F(nonlinear, TT1RadarB3Triangular)
{
  LinearizedTransform t;
  auto in = G2T {{20.0, pi/4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 14.1, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 14.1, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 20.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -19.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -19.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 20.5, 1e-1);
}

TEST_F(nonlinear, TT2RadarB1SelfAdjoint)
{
  LinearizedTransform<2> t;
  auto in = G2 {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 19.0, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 40.1, 1e-1);
}

TEST_F(nonlinear, TT2RadarB1Triangular)
{
  LinearizedTransform<2> t;
  auto in = G2T {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 19.0, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 40.1, 1e-1);
}

TEST_F(nonlinear, TT2RadarB2SelfAdjoint)
{
  LinearizedTransform<2> t;
  auto in = G2 {{20.0, pi/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 16.5, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 9.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 12.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -16.1, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -16.1, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 30.8, 1e-1);
}

TEST_F(nonlinear, TT2RadarB2Triangular)
{
  LinearizedTransform<2> t;
  auto in = G2T {{20.0, pi/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 16.5, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 9.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 12.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -16.1, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -16.1, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 30.8, 1e-1);
}

TEST_F(nonlinear, TT2RadarB3SelfAdjoint)
{
  LinearizedTransform<2> t;
  auto in = G2 {{20.0, pi/4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 13.4, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 13.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 21.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -18.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -18.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 21.6, 1e-1);
}

TEST_F(nonlinear, TT2RadarB3Triangular)
{
  LinearizedTransform<2> t;
  auto in = G2 {{20.0, pi/4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 13.4, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 13.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 21.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -18.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -18.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 21.6, 1e-1);
}

