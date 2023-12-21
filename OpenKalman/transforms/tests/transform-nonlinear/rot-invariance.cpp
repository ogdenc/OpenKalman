/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transform-nonlinear.gtest.hpp"
#include <cmath>

using namespace OpenKalman;
using namespace OpenKalman::test;

using numbers::pi;

inline namespace
{
  using C2 = TypedIndex<Axis, Axis>;
  using P2 = TypedIndex<Polar<>>;

  const GaussianDistribution angle_input {Mean<P2>(1, 0.9999 * pi), Covariance<P2>(0.25, 0, 0, pi * pi / 9)};
  const GaussianDistribution angle_input_tri {Mean<P2>(1, 0.9999 * pi), make_covariance<P2, TriangleType::lower>(0.25, 0, 0, pi * pi / 9)};
  const GaussianDistribution angle_input_rot {Mean<P2>(1, 0.9999 * pi - pi), Covariance<P2>(0.25, 0, 0, pi * pi / 9)};
  const GaussianDistribution angle_input_rot_tri {Mean<P2>(1, 0.9999 * pi - pi), make_covariance<P2, TriangleType::lower>(0.25, 0, 0, pi * pi / 9)};
  const GaussianDistribution cart_noise {make_zero<Mean<C2>>(), Covariance<C2>(0.0625, 0, 0, 0.0625)};
  const GaussianDistribution cart_noise_tri {make_zero<Mean<C2>>(), make_covariance<C2, TriangleType::lower>(0.0625, 0, 0, 0.0625)};
}


TEST(transform_nonlinear, Transform_rotation_cubature)
{
  SamplePointsTransform<CubaturePoints> t;
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input_tri, angle_input_rot_tri));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input_tri, angle_input_rot_tri, cart_noise_tri));
}

TEST(transform_nonlinear, Transform_rotation_unscented)
{
  SamplePointsTransform<UnscentedSigmaPoints> t;
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input_tri, angle_input_rot_tri));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input_tri, angle_input_rot_tri, cart_noise_tri));
}

TEST(transform_nonlinear, Transform_rotation_spherical_simplex)
{
  SamplePointsTransform<SphericalSimplexSigmaPoints> t;
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input_tri, angle_input_rot_tri));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input_tri, angle_input_rot_tri, cart_noise_tri));
}

TEST(transform_nonlinear, Transform_rotation_linearized1)
{
  LinearizedTransform t;
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input_tri, angle_input_rot_tri));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input_tri, angle_input_rot_tri, cart_noise_tri));
}

TEST(transform_nonlinear, Transform_rotation_linearized2)
{
  LinearizedTransform<2> t;
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input_tri, angle_input_rot_tri));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test(
    t, radarP, Cartesian2polar, angle_input_tri, angle_input_rot_tri, cart_noise_tri));
}

