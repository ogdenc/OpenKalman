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

  static auto polar2Cartesian = make_Transformation<P2, C2>(
    [](const Mean<P2>& x, const auto&...noise) -> Mean<C2>
      {
        return {((x(0)*cos(x(1))) + ... + noise(0)), ((x(0)*sin(x(1))) + ... + noise(1))};
      });

  static auto Cartesian2polar = make_Transformation<C2, P2>(
    [](const Mean<C2>& a, const auto&...noise) -> Mean<P2>
      {
        return {((std::hypot(a(1), a(0))) + ... + noise(0)), ((std::atan2(a(1), a(0))) + ... + noise(1))};
      });

  static auto polar2polar = make_Transformation<P2, P2>(
    [](const Mean<P2> a, const auto&...noise) -> Mean<P2>
      {
        return ((a + Mean<P2>(0, M_PI)) + ... + noise);
      });

  const GaussianDistribution angle_input {Mean<P2>(1, 0.95 * M_PI), Covariance<P2>(0.01, 0, 0, M_PI * M_PI / 9)};
  const GaussianDistribution angle_input_rot {Mean<P2>(1, 0.95 * M_PI - M_PI), Covariance<P2>(0.01, 0, 0, M_PI * M_PI / 9)};
  const GaussianDistribution angle_noise {Mean<P2>::zero(), Covariance<P2>(0.01, 0, 0, M_PI * M_PI / 81)};
  const GaussianDistribution cart_noise {Mean<C2>::zero(), Covariance<C2>(0.01, 0, 0, 0.01)};
}


TEST_F(transform_nonlinear_tests, Transform_angle_cubature)
{
  EXPECT_TRUE(rotational_invariance_test<CubaturePoints>(
    polar2Cartesian, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test<CubaturePoints>(
    polar2Cartesian, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test<CubaturePoints>(
    polar2Cartesian, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test<CubaturePoints>(
    polar2Cartesian, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot), to_Cholesky(cart_noise)));
}

TEST_F(transform_nonlinear_tests, Transform_angle_unscented)
{
  EXPECT_TRUE(rotational_invariance_test<UnscentedSigmaPoints>(
    polar2Cartesian, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test<UnscentedSigmaPoints>(
    polar2Cartesian, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test<UnscentedSigmaPoints>(
    polar2Cartesian, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
  EXPECT_TRUE(rotational_invariance_test<SphericalSimplexSigmaPoints>(
    polar2Cartesian, Cartesian2polar, angle_input, angle_input_rot, cart_noise));
}

TEST_F(transform_nonlinear_tests, Transform_angle_spherical_simplex)
{
  EXPECT_TRUE(rotational_invariance_test<SphericalSimplexSigmaPoints>(
    polar2Cartesian, Cartesian2polar, angle_input, angle_input_rot));
  EXPECT_TRUE(rotational_invariance_test<SphericalSimplexSigmaPoints>(
    polar2Cartesian, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot)));
  EXPECT_TRUE(rotational_invariance_test<UnscentedSigmaPoints>(
    polar2Cartesian, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot), to_Cholesky(cart_noise)));
  EXPECT_TRUE(rotational_invariance_test<SphericalSimplexSigmaPoints>(
    polar2Cartesian, Cartesian2polar, to_Cholesky(angle_input), to_Cholesky(angle_input_rot), to_Cholesky(cart_noise)));
}

