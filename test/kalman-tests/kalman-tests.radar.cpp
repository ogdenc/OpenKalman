/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "kalman-tests.h"

using M22 = Eigen::Matrix<double, 2, 2>;

/*TEST_F(kalman_tests, artillery_2d)
{
  using SA = EigenSelfAdjointMatrix<M22>;
  auto true_state = randomize<Radar2, std::uniform_real_distribution>(-M_PI, M_PI);
  auto x = GaussianDistribution<Polar<>, M2, SA> {Radar2::zero(), SA::identity()};
  auto meas_cov = SA {0.1, 0, 0, 0.1};
  auto r = GaussianDistribution<Axes<2>, M2, SA> {Loc2::zero(), meas_cov};
  parameter_test(CubatureTransform(), radarP, x, true_state, r, 0.1, 100);
}*/

/// Locates a stationary object in 2D space based on a radar ping (a Polar vector).
TEST_F(kalman_tests, radar_2d_linearized1_SA)
{
  radar_2D<EigenSelfAdjointMatrix<M22>>(LinearizedTransform<1>());
}

