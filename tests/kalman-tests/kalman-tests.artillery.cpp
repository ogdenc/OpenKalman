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
void kalman_tests::artillery_2D(const Trans& transform)
{
  using M2 = Eigen::Matrix<double, 2, 1>;
  using Loc2 = Mean<Axes<2>, M2>;
  using Polar2 = Mean<Polar<>, M2>;
  for (int i = 0; i < 5; i++)
  {
    using U = std::uniform_real_distribution<double>::param_type;
    auto true_state = randomize<Polar2, std::uniform_real_distribution>(U {5, 10}, U {-M_PI, M_PI});
    auto x = GaussianDistribution<Polar<>, M2, Cov> {Polar2 {7.5, 0}, Cov::identity()};
    auto meas_cov = Cov {0.0025, 0, 0, 0.0025};
    auto r = GaussianDistribution<Axes<2>, M2, Cov> {Loc2::zero(), meas_cov};
    parameter_test(transform, radarP, x, true_state, r, 0.5, 100);
  }
}

using M22 = Eigen::Matrix<double, 2, 2>;

/// Locates a bearing and distance (in polar coordinates) based on Cartesian map coordinates.

TEST_F(kalman_tests, artillery_2d_linearized1_SA)
{
  artillery_2D<EigenSelfAdjointMatrix<M22>>(LinearizedTransform<1>());
}

TEST_F(kalman_tests, artillery_2d_linearized1_T)
{
  artillery_2D<EigenTriangularMatrix<M22>>(LinearizedTransform<1>());
}

TEST_F(kalman_tests, artillery_2d_linearized2_SA)
{
  artillery_2D<EigenSelfAdjointMatrix<M22>>(LinearizedTransform<2>());
}

TEST_F(kalman_tests, artillery_2d_cubature_SA)
{
  artillery_2D<EigenSelfAdjointMatrix<M22>>(CubatureTransform());
}

TEST_F(kalman_tests, artillery_2d_cubature_T)
{
  artillery_2D<EigenTriangularMatrix<M22>>(CubatureTransform());
}

TEST_F(kalman_tests, artillery_2d_unscented_SA)
{
  artillery_2D<EigenSelfAdjointMatrix<M22>>(UnscentedTransformParameterEstimation());
}

TEST_F(kalman_tests, artillery_2d_unscented_T)
{
  artillery_2D<EigenTriangularMatrix<M22>>(UnscentedTransformParameterEstimation());
}

TEST_F(kalman_tests, artillery_2d_simplex_SA)
{
  artillery_2D<EigenSelfAdjointMatrix<M22>>(SamplePointsTransform<SphericalSimplexSigmaPoints>());
}

TEST_F(kalman_tests, artillery_2d_simplex_T)
{
  artillery_2D<EigenTriangularMatrix<M22>>(SamplePointsTransform<SphericalSimplexSigmaPoints>());
}

