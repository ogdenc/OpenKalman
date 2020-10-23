/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "kalman-tests.hpp"

template<typename Cov, typename Trans>
void kalman_tests::radar_2D(const Trans& transform)
{
  using M2 = Eigen::Matrix<double, 2, 1>;
  using Loc2 = Mean<Axes<2>, M2>;
  using Polar2 = Mean<Polar<>, M2>;
  for (int i = 0; i < 5; i++)
  {
    auto true_state = randomize<Loc2, std::uniform_real_distribution>(5.0, 10.0);
    auto x = GaussianDistribution<Axes<2>, M2, Cov> {Loc2 {7.5, 7.5}, Cov::identity()};
    auto meas_cov = Cov {0.01, 0, 0, M_PI/360};
    auto r = GaussianDistribution<Polar<>, M2, Cov> {Polar2::zero(), meas_cov};
    parameter_test(transform, Cartesian2polar, x, true_state, r, 0.2, 100);
  }
}

using M22 = Eigen::Matrix<double, 2, 2>;

/// Locates a stationary object in 2D space based on a radar ping (a Polar vector).

TEST_F(kalman_tests, radar_2d_linearized1_SA)
{
  radar_2D<EigenSelfAdjointMatrix<M22>>(LinearizedTransform<1>());
}

TEST_F(kalman_tests, radar_2d_linearized1_T)
{
  radar_2D<EigenTriangularMatrix<M22>>(LinearizedTransform<1>());
}

TEST_F(kalman_tests, radar_2d_linearized2_SA)
{
  radar_2D<EigenSelfAdjointMatrix<M22>>(LinearizedTransform<2>());
}

TEST_F(kalman_tests, radar_2d_cubature_SA)
{
  radar_2D<EigenSelfAdjointMatrix<M22>>(CubatureTransform());
}

TEST_F(kalman_tests, radar_2d_cubature_T)
{
  radar_2D<EigenTriangularMatrix<M22>>(CubatureTransform());
}

TEST_F(kalman_tests, radar_2d_unscented_SA)
{
  radar_2D<EigenSelfAdjointMatrix<M22>>(UnscentedTransform());
}

TEST_F(kalman_tests, radar_2d_unscented_T)
{
  radar_2D<EigenTriangularMatrix<M22>>(UnscentedTransform());
}

TEST_F(kalman_tests, radar_2d_simplex_SA)
{
  radar_2D<EigenSelfAdjointMatrix<M22>>(SamplePointsTransform<SphericalSimplexSigmaPoints>());
}

TEST_F(kalman_tests, radar_2d_simplex_T)
{
  radar_2D<EigenTriangularMatrix<M22>>(SamplePointsTransform<SphericalSimplexSigmaPoints>());
}

