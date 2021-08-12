/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transform-nonlinear.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;

using std::numbers::pi;

inline namespace
{
  using C2 = Coefficients<Axis, Axis>;
  using M2 = Mean<C2>;
  using Mat2 = Matrix<C2,C2>;
  Mat2 a {1, 2,
          3, 4};
  Mat2 n {0.1, 0.2,
          0.3, 0.4};
  LinearTransformation g {a, n};
  GaussianDistribution input {M2(1, 2), Mat2 {1, 0.1, 0.1, 1}};
  GaussianDistribution input_chol {M2(1, 2), make_covariance<TriangleType::lower>(Mat2 {1, 0.1, 0.1, 1})};
  GaussianDistribution noise {M2::zero(), Mat2 {1, 0.1, 0.1, 1}};
  Covariance<C2> P_output = {5.454, 12.12,
                             12.12, 27.674};
  Covariance<C2> P_output2 = {10.854, 24.12,
                             24.12, 55.074};
  Mat2 cross_output {1.2, 3.4,
                     2.1, 4.3};
  Mat2 cross_output2 {2.4, 6.8,
                      4.2, 8.6};
  GaussianDistribution output {M2(5, 11), P_output};
  GaussianDistribution output2 {M2(5, 11), P_output2};
  auto full_output = std::tuple {output, cross_output};
  auto full_output2 = std::tuple {output2, cross_output2};

  /// Unscented with alpha==1 and kappa = 0 (to avoid negative first weight).
  struct Params
  {
    static constexpr double alpha = 1;
    static constexpr double beta = 2.0;
    template<int dim> static constexpr double kappa = 3 - dim;
  };

  using UnscentedSigmaPoints2 = Unscented<Params>;
}

TEST(transform_nonlinear, Basic_linear_unscented_aug)
{
  SamplePointsTransform<UnscentedSigmaPoints> unscented;
  EXPECT_TRUE(is_near(unscented.transform_with_cross_covariance(input, g, noise), full_output));
  EXPECT_TRUE(is_near(unscented.transform_with_cross_covariance(input_chol, g, noise), full_output));
  static_assert(not cholesky_form<decltype(unscented(input, g))>);
  static_assert(cholesky_form<decltype(unscented(input_chol, g))>);
}

TEST(transform_nonlinear, Basic_linear_unscented2_aug)
{
  SamplePointsTransform<UnscentedSigmaPoints2> unscented2;
  EXPECT_TRUE(is_near(unscented2.transform_with_cross_covariance(input, g, noise), full_output));
  EXPECT_TRUE(is_near(unscented2.transform_with_cross_covariance(input_chol, g, noise), full_output));
  static_assert(not cholesky_form<decltype(unscented2(input, g))>);
  static_assert(cholesky_form<decltype(unscented2(input_chol, g))>);
}

TEST(transform_nonlinear, Basic_linear_spherical_simplex_aug)
{
  SamplePointsTransform<SphericalSimplexSigmaPoints> spherical_simplex;
  EXPECT_TRUE(is_near(spherical_simplex.transform_with_cross_covariance(input, g, noise), full_output));
  EXPECT_TRUE(is_near(spherical_simplex.transform_with_cross_covariance(input_chol, g, noise), full_output));
  static_assert(not cholesky_form<decltype(spherical_simplex(input, g))>);
  static_assert(cholesky_form<decltype(spherical_simplex(input_chol, g))>);
}

TEST(transform_nonlinear, Basic_linear_cubature_aug)
{
  SamplePointsTransform<CubaturePoints> cubature;
  EXPECT_TRUE(is_near(cubature.transform_with_cross_covariance(input, g, noise), full_output));
  EXPECT_TRUE(is_near(cubature.transform_with_cross_covariance(input_chol, g, noise), full_output));
  static_assert(not cholesky_form<decltype(cubature(input, g))>);
  static_assert(cholesky_form<decltype(cubature(input_chol, g))>);
}

TEST(transform_nonlinear, Basic_linear_identity_unscented_aug)
{
  SamplePointsTransform<UnscentedSigmaPoints> t;
  auto out1 = t.transform_with_cross_covariance(input, std::tuple {IdentityTransformation(), noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(out1, full_output2));
  auto out2 = t.transform_with_cross_covariance(input_chol, std::tuple {IdentityTransformation(), noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(out2, full_output2));
}

TEST(transform_nonlinear, Basic_linear_identity_unscented2_aug)
{
  SamplePointsTransform<UnscentedSigmaPoints2> t;
  auto out1 = t.transform_with_cross_covariance(input, std::tuple {IdentityTransformation(), noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(out1, full_output2));
  auto out2 = t.transform_with_cross_covariance(input_chol, std::tuple {IdentityTransformation(), noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(out2, full_output2));
}

TEST(transform_nonlinear, Basic_linear_identity_spherical_simplex_aug)
{
  SamplePointsTransform<SphericalSimplexSigmaPoints> t;
  auto out1 = t.transform_with_cross_covariance(input, std::tuple {IdentityTransformation(), noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(out1, full_output2));
  auto out2 = t.transform_with_cross_covariance(input_chol, std::tuple {IdentityTransformation(), noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(out2, full_output2));
}

TEST(transform_nonlinear, Basic_linear_identity_cubature_aug)
{
  SamplePointsTransform<CubaturePoints> t;
  auto out1 = t.transform_with_cross_covariance(input, std::tuple {IdentityTransformation(), noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(out1, full_output2));
  auto out2 = t.transform_with_cross_covariance(input_chol, std::tuple {IdentityTransformation(), noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(out2, full_output2));
}
