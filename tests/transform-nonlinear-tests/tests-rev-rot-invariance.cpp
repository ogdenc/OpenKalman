/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transform-nonlinear-tests.hpp"

inline namespace
{
  using C2 = Coefficients<Axis, Axis>;
  using P2 = Coefficients<Polar<>>;

  const GaussianDistribution input1 {Mean<C2>(std::cos(0.9999 * M_PI), std::sin(0.9999 * M_PI)), Covariance<C2>(0.25, 0, 0, 0.25)};
  const GaussianDistribution input1_rot {Mean<C2>(std::cos(-0.0001 * M_PI), std::sin(-0.0001 * M_PI)), Covariance<C2>(0.25, 0, 0, 0.25)};
  const GaussianDistribution input2 {Mean<C2>(std::cos(-0.9999 * M_PI), std::sin(-0.9999 * M_PI)), Covariance<C2>(0.25, 0, 0, 0.25)};
  const GaussianDistribution input2_rot {Mean<C2>(std::cos(0.0001 * M_PI), std::sin(0.0001 * M_PI)), Covariance<C2>(0.25, 0, 0, 0.25)};
  const GaussianDistribution noise {Mean<P2>::zero(), Covariance<P2>(0.0625, 0, 0, M_PI * M_PI / 81)};
}


TEST_F(transform_nonlinear_tests, Transform_rev_rotation_cubature1)
{
  CubatureTransform t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input1), to_Cholesky(input1_rot)));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input1), to_Cholesky(input1_rot), to_Cholesky(noise)));
}

TEST_F(transform_nonlinear_tests, Transform_rev_rotation_cubature2)
{
  CubatureTransform t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input2), to_Cholesky(input2_rot)));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input2), to_Cholesky(input2_rot), to_Cholesky(noise)));
}

TEST_F(transform_nonlinear_tests, Transform_rev_rotation_unscented1)
{
  UnscentedTransformParameterEstimation t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input1), to_Cholesky(input1_rot)));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input1), to_Cholesky(input1_rot), to_Cholesky(noise)));
}

TEST_F(transform_nonlinear_tests, Transform_rev_rotation_unscented2)
{
  UnscentedTransform t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input2), to_Cholesky(input2_rot)));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input2), to_Cholesky(input2_rot), to_Cholesky(noise)));
}

TEST_F(transform_nonlinear_tests, Transform_rev_rotation_spherical_simplex1)
{
  SamplePointsTransform<SphericalSimplexSigmaPoints> t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input1), to_Cholesky(input1_rot)));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input1), to_Cholesky(input1_rot), to_Cholesky(noise)));
}

TEST_F(transform_nonlinear_tests, Transform_rev_rotation_spherical_simplex2)
{
  SamplePointsTransform<SphericalSimplexSigmaPoints> t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input2), to_Cholesky(input2_rot)));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input2), to_Cholesky(input2_rot), to_Cholesky(noise)));
}

TEST_F(transform_nonlinear_tests, Transform_rev_rotation_linearized1_1)
{
  LinearizedTransform<1> t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input1), to_Cholesky(input1_rot)));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input1), to_Cholesky(input1_rot), to_Cholesky(noise)));
}

TEST_F(transform_nonlinear_tests, Transform_rev_rotation_linearized1_2)
{
  LinearizedTransform<1> t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input2), to_Cholesky(input2_rot)));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input2), to_Cholesky(input2_rot), to_Cholesky(noise)));
}

TEST_F(transform_nonlinear_tests, Transform_rev_rotation_linearized2_1)
{
  LinearizedTransform<2> t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input1), to_Cholesky(input1_rot)));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input1, input1_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input1), to_Cholesky(input1_rot), to_Cholesky(noise)));
}

TEST_F(transform_nonlinear_tests, Transform_rev_rotation_linearized2_2)
{
  LinearizedTransform<2> t;
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input2), to_Cholesky(input2_rot)));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, input2, input2_rot, noise));
  EXPECT_TRUE(reverse_rotational_invariance_test(
    t, Cartesian2polar, radarP, to_Cholesky(input2), to_Cholesky(input2_rot), to_Cholesky(noise)));
}

