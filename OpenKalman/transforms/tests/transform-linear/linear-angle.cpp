/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transform-linear.gtest.hpp"
#include <cmath>

using namespace OpenKalman;
using namespace OpenKalman::test;

using std::numbers::pi;

using C2 = Coefficients<Polar<>>;
using M2 = Mean<C2>;
using Mat2 = Matrix<C2, C2>;
using Cov2 = Covariance<C2>;

TEST(transform_linear, linear_additive_angle)
{
  auto a = Mat2::identity() * 1.1;
  const GaussianDistribution input {M2(1, pi * 19 / 20), Cov2(pi * pi / 9, 0, 0, 0.01)};
  Cov2 P_output {1.21 * pi * pi / 9, 0,
                 0, 0.0121};
  const GaussianDistribution output {M2(1.1, 1.1 * 19 / 20 * pi - 2 * pi), P_output};
  Mat2 cross_output {1.1 * pi * pi / 9, 0,
                     0, 0.011};
  LinearTransformation g {a};
  LinearTransform t;
  auto [out, cross] = t.transform_with_cross_covariance(input, g);
  EXPECT_TRUE(is_near(out, output));
  EXPECT_TRUE(is_near(cross, cross_output));
}

TEST(transform_linear, linear_additive_angle_Cholesky)
{
  auto a = Mat2::identity() * 1.1;
  const GaussianDistribution input {M2(1, pi * 19 / 20), make_covariance<C2, TriangleType::lower>(pi * pi / 9, 0, 0, 0.01)};
  Cov2 P_output {1.21 * pi * pi / 9, 0,
                 0, 0.0121};
  const GaussianDistribution output {M2(1.1, 1.1 * 19 / 20 * pi - 2 * pi), P_output};
  Mat2 cross_output {1.1 * pi * pi / 9, 0,
                     0, 0.011};
  LinearTransformation g {a};
  LinearTransform t;
  auto [out, cross] = t.transform_with_cross_covariance(input, g);
  EXPECT_TRUE(is_near(out, output));
  EXPECT_TRUE(is_near(cross, cross_output));
}

TEST(transform_linear, linear_augmented_angle)
{
  auto a = Mat2::identity() * 1.1;
  auto an = Mat2::identity();
  const GaussianDistribution input {M2(1, pi * 19 / 20), Cov2(pi * pi / 9, 0, 0, 0.01)};
  const GaussianDistribution noise {M2::zero(), Cov2(Mat2::identity() * 0.01)};
  Cov2 P_output {1.21 * pi * pi / 9 + 0.01, 0,
            0, 0.0121 + 0.01};
  Mat2 cross_output {1.1 * pi * pi / 9, 0,
                     0, 0.011};
  const GaussianDistribution output {M2(1.1, 1.1 * 19 / 20 * pi - 2 * pi), P_output};
  LinearTransformation g {a, an};
  LinearTransform t;
  auto [out, cross] = t.transform_with_cross_covariance(input, g, noise);
  EXPECT_TRUE(is_near(out, output));
  EXPECT_TRUE(is_near(cross, cross_output));
}

TEST(transform_linear, linear_augmented_angle_Cholesky)
{
  auto a = Mat2::identity() * 1.1;
  auto an = Mat2::identity();
  const GaussianDistribution input {M2(1, pi * 19 / 20), make_covariance<C2, TriangleType::lower>(pi * pi / 9, 0, 0, 0.01)};
  const GaussianDistribution noise {M2::zero(), make_covariance<TriangleType::lower>(Mat2::identity() * 0.01)};
  Cov2 P_output {1.21 * pi * pi / 9 + 0.01, 0,
                 0, 0.0121 + 0.01};
  Mat2 cross_output{1.1 * pi * pi / 9, 0,
                    0, 0.011};
  const GaussianDistribution output {M2(1.1, 1.1 * 19 / 20 * pi - 2 * pi), P_output};
  LinearTransformation g {a, an};
  LinearTransform t;
  auto [out, cross] = t.transform_with_cross_covariance(input, g, noise);
  EXPECT_TRUE(is_near(out, output));
  EXPECT_TRUE(is_near(cross, cross_output));
}

