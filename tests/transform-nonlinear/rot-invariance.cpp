/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transform-nonlinear.hpp"

inline namespace
{
  using C2 = Coefficients<Axis, Axis>;
  using P2 = Coefficients<Polar<>>;

  const GaussianDistribution angle_input {Mean<P2>(1, 0.9999 * M_PI), Covariance<P2>(0.25, 0, 0, M_PI * M_PI / 9)};
  const GaussianDistribution angle_input_rot {Mean<P2>(1, 0.9999 * M_PI - M_PI), Covariance<P2>(0.25, 0, 0, M_PI * M_PI / 9)};
  const GaussianDistribution cart_noise {Mean<C2>::zero(), Covariance<C2>(0.0625, 0, 0, 0.0625)};
}


TEST_F(transform_nonlinear, Transform_rotation_cubature)
{
  SamplePointsTransform<CubaturePoints> t;
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot), to_Cholesky(cart_noise)));
}

TEST_F(transform_nonlinear, Transform_rotation_unscented)
{
  SamplePointsTransform<UnscentedSigmaPoints> t;
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot), to_Cholesky(cart_noise)));
}

TEST_F(transform_nonlinear, Transform_rotation_spherical_simplex)
{
  SamplePointsTransform<SphericalSimplexSigmaPoints> t;
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot), to_Cholesky(cart_noise)));
}

TEST_F(transform_nonlinear, Transform_rotation_linearized1)
{
  LinearizedTransform t;
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot), to_Cholesky(cart_noise)));
}

TEST_F(transform_nonlinear, Transform_rotation_linearized2)
{
  LinearizedTransform<2> t;
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot), to_Cholesky(cart_noise)));
}

