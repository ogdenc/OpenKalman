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
 * \details Locates a bearing and distance (in polar coordinates) based on Cartesian map coordinates.
 */


#include "kalman.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;

using numbers::pi;

namespace
{
  template<typename Cov, typename Trans>
  void artillery_2D(const Trans& transform)
  {
    using M2 = eigen_matrix_t<double, 2, 1>;
    using Loc2 = Mean<Dimensions<2>, M2>;
    using Polar2 = Mean<Polar<>, M2>;
    for (int i = 0; i < 5; i++)
    {
      using U = std::uniform_real_distribution<double>;
      auto true_state = randomize<Polar2>(U {5, 10}, U {-pi, pi});
      auto x = GaussianDistribution<Polar<>, M2, Cov> { Polar2 {7.5, 0}, make_identity_matrix_like<Cov>() };
      auto meas_cov = Cov {0.0025, 0, 0, 0.0025};
      auto r = GaussianDistribution<Dimensions<2>, M2, Cov> { make_zero<Loc2>(), meas_cov };
      parameter_test(transform, radarP, x, true_state, r, 0.7, 100);
    }
  }

  using M22 = eigen_matrix_t<double, 2, 2>;
}

TEST(kalman, artillery_2d_linearized1_SA)
{
  artillery_2D<SelfAdjointMatrix<M22>>(LinearizedTransform<1>());
}

TEST(kalman, artillery_2d_linearized1_T)
{
  artillery_2D<TriangularMatrix<M22>>(LinearizedTransform<1>());
}

TEST(kalman, artillery_2d_linearized2_SA)
{
  artillery_2D<SelfAdjointMatrix<M22>>(LinearizedTransform<2>());
}

TEST(kalman, artillery_2d_cubature_SA)
{
  artillery_2D<SelfAdjointMatrix<M22>>(CubatureTransform());
}

TEST(kalman, artillery_2d_cubature_T)
{
  artillery_2D<TriangularMatrix<M22>>(CubatureTransform());
}

TEST(kalman, artillery_2d_unscented_SA)
{
  artillery_2D<SelfAdjointMatrix<M22>>(UnscentedTransformParameterEstimation());
}

TEST(kalman, artillery_2d_unscented_T)
{
  artillery_2D<TriangularMatrix<M22>>(UnscentedTransformParameterEstimation());
}

TEST(kalman, artillery_2d_simplex_SA)
{
  artillery_2D<SelfAdjointMatrix<M22>>(SamplePointsTransform<SphericalSimplexSigmaPoints>());
}

TEST(kalman, artillery_2d_simplex_T)
{
  artillery_2D<TriangularMatrix<M22>>(SamplePointsTransform<SphericalSimplexSigmaPoints>());
}

