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

using M2 = Eigen::Matrix<double, 2, 1>;
using SA = EigenSelfAdjointMatrix<Eigen::Matrix<double, 2, 2>>;
using TR = EigenTriangularMatrix<Eigen::Matrix<double, 2, 2>>;
using G2 = GaussianDistribution<Axes<2>, M2, SA>;
using G2T = GaussianDistribution<Axes<2>, M2, TR>;

struct UT1p
{
  static constexpr double alpha = 1;
  static constexpr double beta = 0;
  template<int dim> static constexpr double kappa = 3 - dim;
};
using UT1 = SigmaPoints<Unscented<UT1p>>;

using UT2 = UnscentedSigmaPointsStateEstimation; // alpha = 1e-3, beta = 2, kappa = 0;


TEST_F(nonlinear_tests, UT1Radar1ASelfAdjoint)
{
  auto t = make_SamplePointsTransform<UT1>(radar);
  auto in = G2 {{3.0, 0.0}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.8, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 3.7, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 2.9, 1e-1);
}

TEST_F(nonlinear_tests, UT1Radar1ATriangular)
{
  auto t = make_SamplePointsTransform<UT1>(radar);
  auto in = G2T {{3.0, 0.0}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.8, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 3.7, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 2.9, 1e-1);
}

TEST_F(nonlinear_tests, UT1Radar2ASelfAdjoint)
{
  auto t = make_SamplePointsTransform<UT1>(radar);
  auto in = G2 {{3.0, M_PI/6}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.6, 1e-1); //2
  EXPECT_NEAR(mean(out)(1), 0.9, 1e-1); //4
  EXPECT_NEAR(covariance(out)(0,0), 3.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.3, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.3, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 3.1, 1e-1);
}

TEST_F(nonlinear_tests, UT1Radar2ATriangular)
{
  auto t = make_SamplePointsTransform<UT1>(radar);
  auto in = G2T {{3.0, M_PI/6}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.6, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.9, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 3.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.3, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.3, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 3.1, 1e-1);
}

TEST_F(nonlinear_tests, UT1Radar3ASelfAdjoint)
{
  auto t = make_SamplePointsTransform<UT1>(radar);
  auto in = G2 {{3.0, M_PI_4}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.3, 1e-1);
  EXPECT_NEAR(mean(out)(1), 1.3, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 3.3, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.4, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.4, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 3.3, 1e-1);
}

TEST_F(nonlinear_tests, UT1Radar3ATriangular)
{
  auto t = make_SamplePointsTransform<UT1>(radar);
  auto in = G2T {{3.0, M_PI_4}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.3, 1e-1);
  EXPECT_NEAR(mean(out)(1), 1.3, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 3.3, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.4, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.4, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 3.3, 1e-1);
}

TEST_F(nonlinear_tests, UT2Radar1ASelfAdjoint)
{
  auto t = make_SamplePointsTransform<UT2>(radar);
  auto in = G2 {{3.0, 0.0}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.5, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 5.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 9.0, 1e-1);
}

TEST_F(nonlinear_tests, UT2Radar1ATriangular)
{
  auto t = make_SamplePointsTransform<UT2>(radar);
  auto in = G2T {{3.0, 0.0}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.5, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 5.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 9.0, 1e-1);
}

TEST_F(nonlinear_tests, UT2Radar2ASelfAdjoint)
{
  auto t = make_SamplePointsTransform<UT2>(radar);
  auto in = G2 {{3.0, M_PI/6}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.3, 1e-1); //2
  EXPECT_NEAR(mean(out)(1), 0.8, 1e-1); //4
  EXPECT_NEAR(covariance(out)(0,0), 6.4, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -1.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -1.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 8.1, 1e-1);
}

TEST_F(nonlinear_tests, UT2Radar2ATriangular)
{
  auto t = make_SamplePointsTransform<UT2>(radar);
  auto in = G2T {{3.0, M_PI/6}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.3, 1e-1); //2
  EXPECT_NEAR(mean(out)(1), 0.8, 1e-1); //4
  EXPECT_NEAR(covariance(out)(0,0), 6.4, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -1.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -1.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 8.1, 1e-1);
}

TEST_F(nonlinear_tests, UT2Radar3ASelfAdjoint)
{
  auto t = make_SamplePointsTransform<UT2>(radar);
  auto in = G2 {{3.0, M_PI_4}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.1, 1e-1);
  EXPECT_NEAR(mean(out)(1), 1.1, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 7.2, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -1.7, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -1.7, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 7.2, 1e-1);
}

TEST_F(nonlinear_tests, UT2Radar3ATriangular)
{
  auto t = make_SamplePointsTransform<UT2>(radar);
  auto in = G2T {{3.0, M_PI_4}, SA::identity()};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 1.1, 1e-1);
  EXPECT_NEAR(mean(out)(1), 1.1, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 7.2, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -1.7, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -1.7, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 7.2, 1e-1);
}


/*
 * Test data from Hendeby & Gustafsson, On Nonlinear Transformations of Stochastic Variables and its Application
 * to Nonlinear Filtering. International Conference on Acoustics, Speech, and Signal Processing (ICASSP). 2007:
 */

/* These do not appear to be accurate in the paper:
TEST_F(nonlinear_tests, UT1Radar1BSelfAdjoint)
{
  auto t = make_SamplePointsTransform<UT1>(radar);
  auto in = G2 {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 20.0, 1e-1); // should be 19.0
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 1.0, 1e-1); // should be 2.9
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 40.0, 1e-1); // should be 36.2
}

TEST_F(nonlinear_tests, UT1Radar1BTriangular)
{
  auto t = make_SamplePointsTransform<UT1>(radar);
  auto in = G2T {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 20.0, 1e-1); // should be 19.0
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 1.0, 1e-1); // should be 2.9
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 40.0, 1e-1); // should be 36.2
}*/

TEST_F(nonlinear_tests, UT1Radar2BSelfAdjoint)
{
  auto t = make_SamplePointsTransform<UT1>(radar);
  auto in = G2 {{20.0, M_PI/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 16.5, 1e-1);
  EXPECT_NEAR(mean(out)(1), 9.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 11.2, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -14.4, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -14.4, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 27.8, 1e-1);
}

TEST_F(nonlinear_tests, UT1Radar2BTriangular)
{
  auto t = make_SamplePointsTransform<UT1>(radar);
  auto in = G2T {{20.0, M_PI/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 16.5, 1e-1);
  EXPECT_NEAR(mean(out)(1), 9.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 11.2, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -14.4, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -14.4, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 27.8, 1e-1);
}

TEST_F(nonlinear_tests, UT1Radar3BSelfAdjoint)
{
  auto t = make_SamplePointsTransform<UT1>(radar);
  auto in = G2 {{20.0, M_PI_4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 13.5, 1e-1);
  EXPECT_NEAR(mean(out)(1), 13.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 19.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -16.6, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -16.6, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 19.5, 1e-1);
}

TEST_F(nonlinear_tests, UT1Radar3BTriangular)
{
  auto t = make_SamplePointsTransform<UT1>(radar);
  auto in = G2T {{20.0, M_PI_4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 13.5, 1e-1);
  EXPECT_NEAR(mean(out)(1), 13.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 19.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -16.6, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -16.6, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 19.5, 1e-1);
}

TEST_F(nonlinear_tests, UT2Radar1BSelfAdjoint)
{
  auto t = make_SamplePointsTransform<UT2>(radar);
  auto in = G2 {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 19.0, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 3.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 40.1, 2e-1);
}

TEST_F(nonlinear_tests, UT2Radar1BTriangular)
{
  auto t = make_SamplePointsTransform<UT2>(radar);
  auto in = G2T {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 19.0, 1e-1);
  EXPECT_NEAR(mean(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 3.0, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 40.1, 2e-1);
}

TEST_F(nonlinear_tests, UT2Radar2BSelfAdjoint)
{
  auto t = make_SamplePointsTransform<UT2>(radar);
  auto in = G2 {{20.0, M_PI/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 16.5, 1e-1);
  EXPECT_NEAR(mean(out)(1), 9.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 12.3, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -16.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -16.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 30.7, 1e-1);
}

TEST_F(nonlinear_tests, UT2Radar2BTriangular)
{
  auto t = make_SamplePointsTransform<UT2>(radar);
  auto in = G2T {{20.0, M_PI/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 16.5, 1e-1);
  EXPECT_NEAR(mean(out)(1), 9.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 12.3, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -16.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -16.0, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 30.7, 1e-1);
}

TEST_F(nonlinear_tests, UT2Radar3BSelfAdjoint)
{
  auto t = make_SamplePointsTransform<UT2>(radar);
  auto in = G2 {{20.0, M_PI_4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 13.4, 1e-1);
  EXPECT_NEAR(mean(out)(1), 13.4, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 21.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -18.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -18.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 21.5, 1e-1);
}

TEST_F(nonlinear_tests, UT2Radar3BTriangular)
{
  auto t = make_SamplePointsTransform<UT2>(radar);
  auto in = G2T {{20.0, M_PI_4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = std::get<0>(t(in));
  EXPECT_NEAR(mean(out)(0), 13.4, 1e-1);
  EXPECT_NEAR(mean(out)(1), 13.4, 1e-1);
  EXPECT_NEAR(covariance(out)(0,0), 21.5, 1e-1);
  EXPECT_NEAR(covariance(out)(0,1), -18.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,0), -18.5, 1e-1);
  EXPECT_NEAR(covariance(out)(1,1), 21.5, 1e-1);
}

