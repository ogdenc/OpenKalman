/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
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

  const GaussianDistribution input1 {Mean<C2>(std::cos(0.9999 * pi), std::sin(0.9999 * pi)), Covariance<C2>(0.25, 0, 0, 0.25)};
  const GaussianDistribution input1_tri {Mean<C2>(std::cos(0.9999 * pi), std::sin(0.9999 * pi)), make_covariance<C2, TriangleType::lower>(0.25, 0, 0, 0.25)};
  const GaussianDistribution input1_rot {Mean<C2>(std::cos(-0.0001 * pi), std::sin(-0.0001 * pi)), Covariance<C2>(0.25, 0, 0, 0.25)};
  const GaussianDistribution input1_rot_tri {Mean<C2>(std::cos(-0.0001 * pi), std::sin(-0.0001 * pi)), make_covariance<C2, TriangleType::lower>(0.25, 0, 0, 0.25)};
  const GaussianDistribution input2 {Mean<C2>(std::cos(-0.9999 * pi), std::sin(-0.9999 * pi)), Covariance<C2>(0.25, 0, 0, 0.25)};
  const GaussianDistribution input2_tri {Mean<C2>(std::cos(-0.9999 * pi), std::sin(-0.9999 * pi)), make_covariance<C2, TriangleType::lower>(0.25, 0, 0, 0.25)};
  const GaussianDistribution input2_rot {Mean<C2>(std::cos(0.0001 * pi), std::sin(0.0001 * pi)), Covariance<C2>(0.25, 0, 0, 0.25)};
  const GaussianDistribution input2_rot_tri {Mean<C2>(std::cos(0.0001 * pi), std::sin(0.0001 * pi)), make_covariance<C2, TriangleType::lower>(0.25, 0, 0, 0.25)};
  const GaussianDistribution noise {Mean<P2>::zero(), Covariance<P2>(0.0625, 0, 0, pi * pi / 81)};
  const GaussianDistribution noise_tri {Mean<P2>::zero(), make_covariance<C2, TriangleType::lower>(0.0625, 0, 0, pi * pi / 81)};
}


TEST_F(transform_nonlinear, Transform_rev_rotation_cubature1)
{
  CubatureTransform t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1_tri, input1_rot_tri));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1_tri, input1_rot_tri, noise_tri));
}

TEST_F(transform_nonlinear, Transform_rev_rotation_cubature2)
{
  CubatureTransform t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2_tri, input2_rot_tri));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2_tri, input2_rot_tri, noise_tri));
}

TEST_F(transform_nonlinear, Transform_rev_rotation_unscented1)
{
  UnscentedTransformParameterEstimation t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1_tri, input1_rot_tri));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1_tri, input1_rot_tri, noise_tri));
}

TEST_F(transform_nonlinear, Transform_rev_rotation_unscented2)
{
  UnscentedTransform t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2_tri, input2_rot_tri));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2_tri, input2_rot_tri, noise_tri));
}

TEST_F(transform_nonlinear, Transform_rev_rotation_spherical_simplex1)
{
  SamplePointsTransform<SphericalSimplexSigmaPoints> t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1_tri, input1_rot_tri));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1_tri, input1_rot_tri, noise_tri));
}

TEST_F(transform_nonlinear, Transform_rev_rotation_spherical_simplex2)
{
  SamplePointsTransform<SphericalSimplexSigmaPoints> t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2_tri, input2_rot_tri));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2_tri, input2_rot_tri, noise_tri));
}

TEST_F(transform_nonlinear, Transform_rev_rotation_linearized1_1)
{
  LinearizedTransform<1> t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1_tri, input1_rot_tri));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1_tri, input1_rot_tri, noise_tri));
}

TEST_F(transform_nonlinear, Transform_rev_rotation_linearized1_2)
{
  LinearizedTransform<1> t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2_tri, input2_rot_tri));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2_tri, input2_rot_tri, noise_tri));
}

TEST_F(transform_nonlinear, Transform_rev_rotation_linearized2_1)
{
  LinearizedTransform<2> t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1_tri, input1_rot_tri));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1_tri, input1_rot_tri, noise_tri));
}

TEST_F(transform_nonlinear, Transform_rev_rotation_linearized2_2)
{
  LinearizedTransform<2> t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2_tri, input2_rot_tri));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2_tri, input2_rot_tri, noise_tri));
}

