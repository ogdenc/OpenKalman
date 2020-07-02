/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transform-nonlinear-tests.h"

using namespace OpenKalman;

namespace test_sample_aug
{
  using C2 = Coefficients<Axis, Axis>;
  using M2 = Mean<C2>;
  using Mat2 = TypedMatrix<C2,C2>;
  Mat2 a {1, 2,
          3, 4};
  Mat2 n {0.1, 0.2,
          0.3, 0.4};
  LinearTransformation g {a, n};
  GaussianDistribution input {M2(1, 2), Mat2 {1, 0.1, 0.1, 1}};
  GaussianDistribution input_chol {M2(1, 2), make_Covariance<TriangleType::lower>(Mat2 {1, 0.1, 0.1, 1})};
  GaussianDistribution noise {M2::zero(), Mat2 {1, 0.1, 0.1, 1}};
  Covariance<C2> P_output = {5.454, 12.12,
                             12.12, 27.674};
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

  using UnscentedSigmaPoints2 = SigmaPoints<Unscented<Params>>;
}

using namespace test_sample_aug;

TEST_F(transform_tests, Basic_linear_unscented_aug)
{
  auto unscented = make_SamplePointsTransform<UnscentedSigmaPoints>(g);
  EXPECT_TRUE(is_near(unscented(input, noise), full_output));
  EXPECT_TRUE(is_near(unscented(input_chol, noise), full_output));
  static_assert(not is_Cholesky_v<decltype(std::get<0>(unscented(input)))>);
  static_assert(is_Cholesky_v<decltype(std::get<0>(unscented(input_chol)))>);
}

TEST_F(transform_tests, Basic_linear_unscented2_aug)
{
  auto unscented2 = make_SamplePointsTransform<UnscentedSigmaPoints2>(g);
  EXPECT_TRUE(is_near(unscented2(input, noise), full_output));
  EXPECT_TRUE(is_near(unscented2(input_chol, noise), full_output));
  static_assert(not is_Cholesky_v<decltype(std::get<0>(unscented2(input)))>);
  static_assert(is_Cholesky_v<decltype(std::get<0>(unscented2(input_chol)))>);
}

TEST_F(transform_tests, Basic_linear_spherical_simplex_aug)
{
  auto spherical_simplex = make_SamplePointsTransform<SphericalSimplexSigmaPoints>(g);
  EXPECT_TRUE(is_near(spherical_simplex(input, noise), full_output));
  EXPECT_TRUE(is_near(spherical_simplex(input_chol, noise), full_output));
  static_assert(not is_Cholesky_v<decltype(std::get<0>(spherical_simplex(input)))>);
  static_assert(is_Cholesky_v<decltype(std::get<0>(spherical_simplex(input_chol)))>);
}

TEST_F(transform_tests, Basic_linear_cubature_aug)
{
  auto cubature = make_SamplePointsTransform<CubaturePoints>(g);
  EXPECT_TRUE(is_near(cubature(input, noise), full_output));
  EXPECT_TRUE(is_near(cubature(input_chol, noise), full_output));
  static_assert(not is_Cholesky_v<decltype(std::get<0>(cubature(input)))>);
  static_assert(is_Cholesky_v<decltype(std::get<0>(cubature(input_chol)))>);
}
