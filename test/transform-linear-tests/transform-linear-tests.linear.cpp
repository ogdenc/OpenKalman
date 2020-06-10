/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transform-linear-tests.h"

using namespace OpenKalman;

using Axis2 = Coefficients<Axis, Axis>;
using M2 = Mean<Axis2>;
using Mat2 = TypedMatrix<Axis2, Axis2>;

TEST_F(transform_tests, linear_additive)
{
  Mat2 a {1, 2,
          3, 4};
  LinearTransform t {a};
  GaussianDistribution input {M2(1, 2), Mat2::identity()};
  GaussianDistribution noise {M2::zero(), Mat2::identity()};
  Mat2 P_output {5, 11,
                 11, 25};
  Mat2 cross_output {1, 3,
                     2, 4};
  auto [out, cross] = t(input);
  EXPECT_TRUE(is_near(mean(out), M2(5, 11)));
  EXPECT_TRUE(is_near(covariance(out), P_output));
  EXPECT_TRUE(is_near(out + noise, GaussianDistribution {M2(5, 11), P_output + Mat2::identity()}));
  EXPECT_TRUE(is_near(cross, cross_output));
}

TEST_F(transform_tests, linear_additive_Cholesky)
{
  Mat2 a {1, 2,
          3, 4};
  LinearTransform t {a};
  GaussianDistribution input {M2(1, 2), make_Covariance<TriangleType::lower>(Mat2::identity())};
  GaussianDistribution noise {M2::zero(), make_Covariance<TriangleType::lower>(Mat2::identity())};
  Mat2 P_output {5, 11,
                 11, 25};
  Mat2 cross_output {1, 3,
                     2, 4};
  auto [out, cross] = t(input);
  EXPECT_TRUE(is_near(mean(out), M2(5, 11)));
  EXPECT_TRUE(is_near(covariance(out), P_output));
  EXPECT_TRUE(is_near(out + noise, GaussianDistribution {M2(5, 11), P_output + Mat2::identity()}));
  EXPECT_TRUE(is_near(cross, cross_output));
}

TEST_F(transform_tests, linear_augmented)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  LinearTransform t {a, an};
  GaussianDistribution input {M2(1, 2), Mat2::identity()};
  GaussianDistribution noise {M2::zero(), Mat2::identity()};
  Mat2 P_output {30, 20,
                 20, 30};
  Mat2 cross_output {1, 4,
                     2, 3};
  EXPECT_TRUE(is_near(t(input, noise), std::tuple {GaussianDistribution {M2(5, 10), P_output}, cross_output}));
}

TEST_F(transform_tests, linear_augmented_Cholesky)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  LinearTransform t {a, an};
  GaussianDistribution input {M2(1, 2), make_Covariance<TriangleType::lower>(Mat2::identity())};
  GaussianDistribution noise {M2::zero(), make_Covariance<TriangleType::lower>(Mat2::identity())};
  Mat2 P_output {30, 20,
                 20, 30};
  Mat2 cross_output {1, 4,
                     2, 3};
  EXPECT_TRUE(is_near(t(input, noise), std::tuple {GaussianDistribution {M2(5, 10), P_output}, cross_output}));
}

