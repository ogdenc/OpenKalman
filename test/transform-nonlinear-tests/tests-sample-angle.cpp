/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transform-nonlinear-tests.h"

inline namespace
{
  using C2 = Coefficients<Axis, Axis>;
  using P2 = Coefficients<Polar<>>;

  const GaussianDistribution angle_input {Mean<P2>(1, 0.95 * M_PI), Covariance<P2>(0.01, 0, 0, M_PI * M_PI / 9)};
  const GaussianDistribution angle_input_rot {Mean<P2>(1, 0.95 * M_PI - M_PI), Covariance<P2>(0.01, 0, 0, M_PI * M_PI / 9)};
  const GaussianDistribution angle_noise {Mean<P2>::zero(), Covariance<P2>(0.01, 0, 0, M_PI * M_PI / 81)};
  const GaussianDistribution cart_noise {Mean<C2>::zero(), Covariance<C2>(0.01, 0, 0, 0.01)};
}


TEST_F(transform_nonlinear_tests, Transform_angle_cubature)
{
  EXPECT_TRUE(rotational_invariance_test<CubaturePoints>(
    radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test<CubaturePoints>(
    radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test<CubaturePoints>(
    radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test<CubaturePoints>(
    radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot), to_Cholesky(cart_noise)));
}

TEST_F(transform_nonlinear_tests, Transform_angle_unscented)
{
  EXPECT_TRUE(rotational_invariance_test<UnscentedSigmaPoints>(
    radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test<UnscentedSigmaPoints>(
    radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test<UnscentedSigmaPoints>(
    radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test<SphericalSimplexSigmaPoints>(
    radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
}

TEST_F(transform_nonlinear_tests, Transform_angle_spherical_simplex)
{
  EXPECT_TRUE(rotational_invariance_test<SphericalSimplexSigmaPoints>(
    radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test<SphericalSimplexSigmaPoints>(
    radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test<UnscentedSigmaPoints>(
    radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot), to_Cholesky(cart_noise)));
  EXPECT_TRUE(rotational_invariance_test<SphericalSimplexSigmaPoints>(
    radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot), to_Cholesky(cart_noise)));
}

