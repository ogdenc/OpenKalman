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
using namespace OpenKalman::coordinates;
using namespace OpenKalman::test;

using stdcompat::numbers::pi;

inline namespace
{
  using C2 = Dimensions<2>;
  using M2 = Mean<C2>;
  using Mat2 = Matrix<C2,C2>;
  Mat2 a {1, 2,
          3, 4};
  LinearTransformation g {a};
  GaussianDistribution input {M2(1, 2), Mat2 {1, 0.1, 0.1, 1}};
  GaussianDistribution input_chol {M2(1, 2), make_covariance<triangle_type::lower>(Mat2 {1, 0.1, 0.1, 1})};
  Covariance<C2> P_output = {5.4, 12,
                             12, 27.4};
  Mat2 cross_output {1.2, 3.4,
                     2.1, 4.3};
  GaussianDistribution output {M2(5, 11), P_output};
  auto full_output = std::tuple {output, cross_output};

  /// Unscented with alpha==1 and kappa = 0 (to avoid negative first weight).
  struct Params
  {
    static constexpr double alpha = 1;
    static constexpr double beta = 2.0;
    template<int dim> static constexpr double kappa = 3 - dim;
  };

  using UnscentedSigmaPoints2 = Unscented<Params>;
}


TEST(transform_nonlinear, Basic_linear_unscented)
{
  SamplePointsTransform<UnscentedSigmaPoints> unscented;
  EXPECT_TRUE(is_near(unscented.transform_with_cross_covariance(input, g), full_output));
  EXPECT_TRUE(is_near(unscented.transform_with_cross_covariance(input_chol, g), full_output));
  static_assert(not cholesky_form<decltype(unscented(input, g))>);
  static_assert(cholesky_form<decltype(unscented(input_chol, g))>);
}

TEST(transform_nonlinear, Basic_linear_unscented2)
{
  SamplePointsTransform<UnscentedSigmaPoints2> unscented2;
  EXPECT_TRUE(is_near(unscented2.transform_with_cross_covariance(input, g), full_output));
  EXPECT_TRUE(is_near(unscented2.transform_with_cross_covariance(input_chol, g), full_output));
  static_assert(not cholesky_form<decltype(unscented2(input, g))>);
  static_assert(cholesky_form<decltype(unscented2(input_chol, g))>);
}

TEST(transform_nonlinear, Basic_linear_spherical_simplex)
{
  SamplePointsTransform<SphericalSimplexSigmaPoints> spherical_simplex;
  EXPECT_TRUE(is_near(spherical_simplex.transform_with_cross_covariance(input, g), full_output));
  EXPECT_TRUE(is_near(spherical_simplex.transform_with_cross_covariance(input_chol, g), full_output));
  static_assert(not cholesky_form<decltype(spherical_simplex(input, g))>);
  static_assert(cholesky_form<decltype(spherical_simplex(input_chol, g))>);
}

TEST(transform_nonlinear, Basic_linear_cubature)
{
  SamplePointsTransform<CubaturePoints> cubature;
  EXPECT_TRUE(is_near(cubature.transform_with_cross_covariance(input, g), full_output));
  EXPECT_TRUE(is_near(cubature.transform_with_cross_covariance(input_chol, g), full_output));
  static_assert(not cholesky_form<decltype(cubature(input, g))>);
  static_assert(cholesky_form<decltype(cubature(input_chol, g))>);
}

TEST(transform_nonlinear, Basic_linear_identity_unscented)
{
  SamplePointsTransform<UnscentedSigmaPoints> t;
  auto out1 = t.transform_with_cross_covariance(input, std::tuple {IdentityTransformation()}, std::tuple {g});
  EXPECT_TRUE(is_near(out1, full_output));
  auto out2 = t.transform_with_cross_covariance(input_chol, std::tuple {IdentityTransformation()}, std::tuple {g});
  EXPECT_TRUE(is_near(out2, full_output));
}

TEST(transform_nonlinear, Basic_linear_identity_unscented2)
{
  SamplePointsTransform<UnscentedSigmaPoints2> t;
  auto out1 = t.transform_with_cross_covariance(input, std::tuple {IdentityTransformation()}, std::tuple {g});
  EXPECT_TRUE(is_near(out1, full_output));
  auto out2 = t.transform_with_cross_covariance(input_chol, std::tuple {IdentityTransformation()}, std::tuple {g});
  EXPECT_TRUE(is_near(out2, full_output));
}

TEST(transform_nonlinear, Basic_linear_identity_spherical_simplex)
{
  SamplePointsTransform<SphericalSimplexSigmaPoints> t;
  auto out1 = t.transform_with_cross_covariance(input, std::tuple {IdentityTransformation()}, std::tuple {g});
  EXPECT_TRUE(is_near(out1, full_output));
  auto out2 = t.transform_with_cross_covariance(input_chol, std::tuple {IdentityTransformation()}, std::tuple {g});
  EXPECT_TRUE(is_near(out2, full_output));
}

TEST(transform_nonlinear, Basic_linear_identity_cubature)
{
  SamplePointsTransform<CubaturePoints> t;
  auto out1 = t.transform_with_cross_covariance(input, std::tuple {IdentityTransformation()}, std::tuple {g});
  EXPECT_TRUE(is_near(out1, full_output));
  auto out2 = t.transform_with_cross_covariance(input_chol, std::tuple {IdentityTransformation()}, std::tuple {g});
  EXPECT_TRUE(is_near(out2, full_output));
}
