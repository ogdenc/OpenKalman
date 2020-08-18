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

/*
 * Test data from Gustafsson & Hendeby. Some Relations Between Extended and Unscented Kalman Filters.
 * IEEE Transactions on Signal Processing, (60), 2, 545-555. 2012 (TT2 data appears to be in error, and is omitted).
 */

using M2 = Eigen::Matrix<double, 2, 1>;
using SA = EigenSelfAdjointMatrix<Eigen::Matrix<double, 2, 2>>;
using TR = EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>>;
using G2 = GaussianDistribution<Axes<2>, M2, SA>;
using G2T = GaussianDistribution<Axes<2>, M2, TR>;

TEST_F(nonlinear_tests, TT1RadarA1SelfAdjoint)
{
  auto t = make_LinearizedTransform(radar);
  auto in = G2 {{3.0, 0.0}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 3.0, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 1.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 9.0, 1e-1);
}

TEST_F(nonlinear_tests, TT1RadarA1Triangular)
{
  auto t = make_LinearizedTransform(radar);
  auto in = G2T {{3.0, 0.0}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 3.0, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 1.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 9.0, 1e-1);
}

TEST_F(nonlinear_tests, TT1RadarA2SelfAdjoint)
{
  auto t = make_LinearizedTransform(radar);
  auto in = G2 {{3.0, M_PI/6}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 2.6, 1e-1);
  EXPECT_NEAR(mean(out)(1), 1.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 3.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -3.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -3.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 7.0, 1e-1);
}

TEST_F(nonlinear_tests, TT1RadarA2Triangular)
{
  auto t = make_LinearizedTransform(radar);
  auto in = G2T {{3.0, M_PI/6}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 2.6, 1e-1);
  EXPECT_NEAR(mean(out)(1), 1.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 3.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -3.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -3.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 7.0, 1e-1);
}

TEST_F(nonlinear_tests, TT1RadarA3SelfAdjoint)
{
  auto t = make_LinearizedTransform(radar);
  auto in = G2 {{3.0, M_PI_4}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 2.1, 1e-1);
  EXPECT_NEAR(mean(out)(1), 2.1, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 5.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -4.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -4.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 5.0, 1e-1);
}

TEST_F(nonlinear_tests, TT1RadarA3Triangular)
{
  auto t = make_LinearizedTransform(radar);
  auto in = G2T {{3.0, M_PI_4}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 2.1, 1e-1);
  EXPECT_NEAR(mean(out)(1), 2.1, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 5.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -4.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -4.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 5.0, 1e-1);
}


/*
 * Test data from Hendeby & Gustafsson, On Nonlinear Transformations of Stochastic Variables and its Application
 * to Nonlinear Filtering. International Conference on Acoustics, Speech, and Signal Processing (ICASSP). 2007:
 */

TEST_F(nonlinear_tests, TT1RadarB1SelfAdjoint)
{
  auto t = make_LinearizedTransform(radar);
  auto in = G2 {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 20.0, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 1.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 40.0, 1e-1);
}

TEST_F(nonlinear_tests, TT1RadarB1Triangular)
{
  auto t = make_LinearizedTransform(radar);
  auto in = G2T {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 20.0, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 1.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 40.0, 1e-1);
}

TEST_F(nonlinear_tests, TT1RadarB2SelfAdjoint)
{
  auto t = make_LinearizedTransform(radar);
  auto in = G2 {{20.0, M_PI/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 17.3, 1e-1);
  EXPECT_NEAR(mean(out)(1), 10.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 10.7, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -16.9, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -16.9, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 30.3, 1e-1);
}

TEST_F(nonlinear_tests, TT1RadarB2Triangular)
{
  auto t = make_LinearizedTransform(radar);
  auto in = G2T {{20.0, M_PI/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 17.3, 1e-1);
  EXPECT_NEAR(mean(out)(1), 10.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 10.7, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -16.9, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -16.9, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 30.3, 1e-1);
}

TEST_F(nonlinear_tests, TT1RadarB3SelfAdjoint)
{
  auto t = make_LinearizedTransform(radar);
  auto in = G2 {{20.0, M_PI_4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 14.1, 1e-1);
  EXPECT_NEAR(mean(out)(1), 14.1, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 20.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -19.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -19.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 20.5, 1e-1);
}

TEST_F(nonlinear_tests, TT1RadarB3Triangular)
{
  auto t = make_LinearizedTransform(radar);
  auto in = G2T {{20.0, M_PI_4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 14.1, 1e-1);
  EXPECT_NEAR(mean(out)(1), 14.1, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 20.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -19.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -19.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 20.5, 1e-1);
}

TEST_F(nonlinear_tests, TT2RadarB1SelfAdjoint)
{
  auto t = make_LinearizedTransform<2>(radar);
  auto in = G2 {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 19.0, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 3.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 40.1, 1e-1);
}

TEST_F(nonlinear_tests, TT2RadarB1Triangular)
{
  auto t = make_LinearizedTransform<2>(radar);
  auto in = G2T {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 19.0, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 3.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 40.1, 1e-1);
}

TEST_F(nonlinear_tests, TT2RadarB2SelfAdjoint)
{
  auto t = make_LinearizedTransform<2>(radar);
  auto in = G2 {{20.0, M_PI/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 16.5, 1e-1);
  EXPECT_NEAR(mean(out)(1), 9.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 12.3, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -16.1, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -16.1, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 30.8, 1e-1);
}

TEST_F(nonlinear_tests, TT2RadarB2Triangular)
{
  auto t = make_LinearizedTransform<2>(radar);
  auto in = G2T {{20.0, M_PI/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 16.5, 1e-1);
  EXPECT_NEAR(mean(out)(1), 9.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 12.3, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -16.1, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -16.1, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 30.8, 1e-1);
}

TEST_F(nonlinear_tests, TT2RadarB3SelfAdjoint)
{
  auto t = make_LinearizedTransform<2>(radar);
  auto in = G2 {{20.0, M_PI_4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 13.4, 1e-1);
  EXPECT_NEAR(mean(out)(1), 13.4, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 21.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -18.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -18.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 21.6, 1e-1);
}

TEST_F(nonlinear_tests, TT2RadarB3Triangular)
{
  auto t = make_LinearizedTransform<2>(radar);
  auto in = G2 {{20.0, M_PI_4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 13.4, 1e-1);
  EXPECT_NEAR(mean(out)(1), 13.4, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 21.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -18.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -18.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 21.6, 1e-1);
}

