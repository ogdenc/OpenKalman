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

using stdcompat::numbers::pi;


inline namespace
{
  using M2 = eigen_matrix_t<double, 2, 1>;
  using SA = HermitianAdapter<eigen_matrix_t<double, 2, 2>>;
  using TR = TriangularAdapter<eigen_matrix_t<double, 2, 2>>;
  using G2 = GaussianDistribution<Polar<>, M2, SA>;
  using G2T = GaussianDistribution<Polar<>, M2, TR>;

  struct UT1p
  {
    static constexpr double alpha = 1;
    static constexpr double beta = 0;
    template<int dim> static constexpr double kappa = 3 - dim;
  };

  using UT1 = Unscented<UT1p>;

  using UT2 = UnscentedSigmaPointsStateEstimation; // alpha = 1e-3, beta = 2, kappa = 0;
}


TEST(nonlinear, UT1Radar1ASelfAdjoint)
{
  SamplePointsTransform<UT1> t;
  auto in = G2 {{3.0, 0.0}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.8, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.7, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 2.9, 1e-1);
}

TEST(nonlinear, UT1Radar1ATriangular)
{
  SamplePointsTransform<UT1> t;
  auto in = G2T {{3.0, 0.0}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.8, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.7, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 2.9, 1e-1);
}

TEST(nonlinear, UT1Radar2ASelfAdjoint)
{
  SamplePointsTransform<UT1> t;
  auto in = G2 {{3.0, pi/6}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.6, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.9, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 3.1, 1e-1);
}

TEST(nonlinear, UT1Radar2ATriangular)
{
  SamplePointsTransform<UT1> t;
  auto in = G2T {{3.0, pi/6}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.6, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.9, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 3.1, 1e-1);
}

TEST(nonlinear, UT1Radar3ASelfAdjoint)
{
  SamplePointsTransform<UT1> t;
  auto in = G2 {{3.0, pi/4}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.3, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 1.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 3.3, 1e-1);
}

TEST(nonlinear, UT1Radar3ATriangular)
{
  SamplePointsTransform<UT1> t;
  auto in = G2T {{3.0, pi/4}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.3, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 1.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 3.3, 1e-1);
}

TEST(nonlinear, UT2Radar1ASelfAdjoint)
{
  SamplePointsTransform<UT2> t;
  auto in = G2 {{3.0, 0.0}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.5, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 5.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 9.0, 1e-1);
}

TEST(nonlinear, UT2Radar1ATriangular)
{
  SamplePointsTransform<UT2> t;
  auto in = G2T {{3.0, 0.0}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.5, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 5.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 9.0, 1e-1);
}

TEST(nonlinear, UT2Radar2ASelfAdjoint)
{
  SamplePointsTransform<UT2> t;
  auto in = G2 {{3.0, pi/6}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.3, 1e-1); //2
  EXPECT_NEAR(mean_of(out)(1), 0.8, 1e-1); //4
  EXPECT_NEAR(covariance_of(out)(0,0), 6.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -1.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -1.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 8.1, 1e-1);
}

TEST(nonlinear, UT2Radar2ATriangular)
{
  SamplePointsTransform<UT2> t;
  auto in = G2T {{3.0, pi/6}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.3, 1e-1); //2
  EXPECT_NEAR(mean_of(out)(1), 0.8, 1e-1); //4
  EXPECT_NEAR(covariance_of(out)(0,0), 6.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -1.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -1.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 8.1, 1e-1);
}

TEST(nonlinear, UT2Radar3ASelfAdjoint)
{
  SamplePointsTransform<UT2> t;
  auto in = G2 {{3.0, pi/4}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.1, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 1.1, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 7.2, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -1.7, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -1.7, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 7.2, 1e-1);
}

TEST(nonlinear, UT2Radar3ATriangular)
{
  SamplePointsTransform<UT2> t;
  auto in = G2T {{3.0, pi/4}, make_identity_matrix_like<SA>()};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 1.1, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 1.1, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 7.2, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -1.7, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -1.7, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 7.2, 1e-1);
}


/*
 * Test data from Hendeby & Gustafsson, On Nonlinear Transformations of Stochastic Variables and its Application
 * to Nonlinear Filtering. International Conference on Acoustics, Speech, and Signal Processing (ICASSP). 2007:
 */

/* These do not appear to be accurate in the paper:
TEST(nonlinear_tests, UT1Radar1BSelfAdjoint)
{
  SamplePointsTransform<UT1> t;
  auto in = G2 {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radar);
  EXPECT_NEAR(mean_of(out)(0), 20.0, 1e-1); // should be 19.0
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 1.0, 1e-1); // should be 2.9
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 40.0, 1e-1); // should be 36.2
}

TEST(nonlinear_tests, UT1Radar1BTriangular)
{
  SamplePointsTransform<UT1> t;
  auto in = G2T {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radar);
  EXPECT_NEAR(mean_of(out)(0), 20.0, 1e-1); // should be 19.0
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 1.0, 1e-1); // should be 2.9
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 40.0, 1e-1); // should be 36.2
}*/

TEST(nonlinear, UT1Radar2BSelfAdjoint)
{
  SamplePointsTransform<UT1> t;
  auto in = G2 {{20.0, pi/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 16.5, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 9.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 11.2, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -14.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -14.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 27.8, 1e-1);
}

TEST(nonlinear, UT1Radar2BTriangular)
{
  SamplePointsTransform<UT1> t;
  auto in = G2T {{20.0, pi/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 16.5, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 9.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 11.2, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -14.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -14.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 27.8, 1e-1);
}

TEST(nonlinear, UT1Radar3BSelfAdjoint)
{
  SamplePointsTransform<UT1> t;
  auto in = G2 {{20.0, pi/4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 13.5, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 13.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 19.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -16.6, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -16.6, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 19.5, 1e-1);
}

TEST(nonlinear, UT1Radar3BTriangular)
{
  SamplePointsTransform<UT1> t;
  auto in = G2T {{20.0, pi/4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 13.5, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 13.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 19.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -16.6, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -16.6, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 19.5, 1e-1);
}

TEST(nonlinear, UT2Radar1BSelfAdjoint)
{
  SamplePointsTransform<UT2> t;
  auto in = G2 {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 19.0, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 40.1, 2e-1);
}

TEST(nonlinear, UT2Radar1BTriangular)
{
  SamplePointsTransform<UT2> t;
  auto in = G2T {{20.0, 0.0}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 19.0, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 3.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), 0.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 40.1, 2e-1);
}

TEST(nonlinear, UT2Radar2BSelfAdjoint)
{
  SamplePointsTransform<UT2> t;
  auto in = G2 {{20.0, pi/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 16.5, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 9.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 12.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -16.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -16.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 30.7, 1e-1);
}

TEST(nonlinear, UT2Radar2BTriangular)
{
  SamplePointsTransform<UT2> t;
  auto in = G2T {{20.0, pi/6}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 16.5, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 9.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 12.3, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -16.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -16.0, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 30.7, 1e-1);
}

TEST(nonlinear, UT2Radar3BSelfAdjoint)
{
  SamplePointsTransform<UT2> t;
  auto in = G2 {{20.0, pi/4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 13.4, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 13.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 21.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -18.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -18.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 21.5, 1e-1);
}

TEST(nonlinear, UT2Radar3BTriangular)
{
  SamplePointsTransform<UT2> t;
  auto in = G2T {{20.0, pi/4}, {1.0, 0.0, 0.0, 0.1}};
  auto out = t(in, radarP);
  EXPECT_NEAR(mean_of(out)(0), 13.4, 1e-1);
  EXPECT_NEAR(mean_of(out)(1), 13.4, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,0), 21.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(0,1), -18.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,0), -18.5, 1e-1);
  EXPECT_NEAR(covariance_of(out)(1,1), 21.5, 1e-1);
}

