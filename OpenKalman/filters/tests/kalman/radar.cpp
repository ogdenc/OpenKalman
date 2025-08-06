/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \details Locates a stationary object in 2D space based on a radar ping (a Polar vector).
 */

#include "kalman.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;

using stdcompat::numbers::pi;

namespace
{
  template<typename Cov, typename Trans>
  void radar_2D(const Trans& transform)
  {
    using M2 = eigen_matrix_t<double, 2, 1>;
    using Loc2 = Mean<Dimensions<2>, M2>;
    using Polar2 = Mean<Polar<>, M2>;
    for (int i = 0; i < 5; i++)
    {
      auto true_state = randomize<Loc2>(std::uniform_real_distribution {5.0, 10.0});
      auto x = GaussianDistribution<Dimensions<2>, M2, Cov> { Loc2 {7.5, 7.5}, make_identity_matrix_like<Cov>() };
      auto meas_cov = Cov {0.01, 0, 0, stdcompat::numbers::pi / 360};
      auto r = GaussianDistribution<Polar<>, M2, Cov> { make_zero<Polar2>(), meas_cov };
      parameter_test(transform, Cartesian2polar, x, true_state, r, 0.3, 100);
    }
  }

  using M22 = eigen_matrix_t<double, 2, 2>;

}


TEST(kalman, radar_2d_linearized1_SA)
{
  radar_2D<HermitianAdapter<M22>>(LinearizedTransform<1>());
}

TEST(kalman, radar_2d_linearized1_T)
{
  radar_2D<TriangularAdapter<M22>>(LinearizedTransform<1>());
}

TEST(kalman, radar_2d_linearized2_SA)
{
  radar_2D<HermitianAdapter<M22>>(LinearizedTransform<2>());
}

TEST(kalman, radar_2d_cubature_SA)
{
  radar_2D<HermitianAdapter<M22>>(CubatureTransform());
}

TEST(kalman, radar_2d_cubature_T)
{
  radar_2D<TriangularAdapter<M22>>(CubatureTransform());
}

TEST(kalman, radar_2d_unscented_SA)
{
  radar_2D<HermitianAdapter<M22>>(UnscentedTransform());
}

TEST(kalman, radar_2d_unscented_T)
{
  radar_2D<TriangularAdapter<M22>>(UnscentedTransform());
}

TEST(kalman, radar_2d_simplex_SA)
{
  radar_2D<HermitianAdapter<M22>>(SamplePointsTransform<SphericalSimplexSigmaPoints>());
}

TEST(kalman, radar_2d_simplex_T)
{
  radar_2D<TriangularAdapter<M22>>(SamplePointsTransform<SphericalSimplexSigmaPoints>());
}

