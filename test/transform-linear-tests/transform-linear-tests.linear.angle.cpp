/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <cmath>
#include "transform-linear-tests.h"

using namespace OpenKalman;

using C2 = Coefficients<Polar<>>;
using M2 = Mean<C2>;
using Mat2 = TypedMatrix<C2, C2>;

TEST_F(transform_tests, linear_additive_angle)
{
  auto a = Mat2::identity() * 1.1;
  const GaussianDistribution input {M2(1, M_PI * 19 / 20), Mat2(M_PI *M_PI / 9, 0, 0, 0.01)};
  Mat2 P_output1 {1.21 * M_PI *M_PI / 9, 0,
                  0, 0.0121};
  const GaussianDistribution output1 {M2(1.1, 1.1 * 19 / 20 * M_PI - 2 * M_PI), P_output1};
  Mat2 cross_output {1.1 * M_PI *M_PI / 9, 0,
                     0, 0.011};
  const auto full_output1 = std::tuple {output1, cross_output};
  const LinearTransform t {a};
  EXPECT_TRUE(is_near(t(input), full_output1));
}

TEST_F(transform_tests, linear_augmented_angle)
{
  auto a = Mat2::identity() * 1.1;
  auto an = Mat2::identity();
  const LinearTransformation g {a, an};
  const GaussianDistribution input {M2(1, M_PI * 19 / 20), Mat2(M_PI * M_PI / 9, 0, 0, 0.01)};
  const GaussianDistribution noise {M2::zero(), Mat2(Mat2::identity() * 0.01)};
  Mat2 P_output {1.21 * M_PI * M_PI / 9 + 0.01, 0,
            0, 0.0121 + 0.01};
  Mat2 cross_output {1.1 * M_PI * M_PI / 9, 0,
                     0, 0.011};
  const GaussianDistribution output {M2(1.1, 1.1 * 19 / 20 * M_PI - 2 * M_PI), P_output};
  const auto full_output = std::tuple {output, cross_output};
  const LinearTransform t {g};
  EXPECT_TRUE(is_near(t(input, noise), full_output));
}

TEST_F(transform_tests, linear_additive_angle_Cholesky)
{
  auto a = Mat2::identity() * 1.1;
  const GaussianDistribution input {M2(1, M_PI * 19 / 20), make_Covariance<C2, TriangleType::lower>(M_PI * M_PI / 9, 0, 0, 0.01)};
  Mat2 P_output {1.21 * M_PI * M_PI / 9, 0,
                 0, 0.0121};
  Mat2 cross_output {1.1 * M_PI * M_PI / 9, 0,
                     0, 0.011};
  const GaussianDistribution output {M2(1.1, 1.1 * 19 / 20 * M_PI - 2 * M_PI), P_output};
  const LinearTransform t {a};
  auto [out, cross] = t(input);
  EXPECT_TRUE(is_near(out, output));
  EXPECT_TRUE(is_near(cross, cross_output));
}

TEST_F(transform_tests, linear_augmented_angle_Cholesky)
{
  auto a = Mat2::identity() * 1.1;
  auto an = Mat2::identity();
  const GaussianDistribution input {M2(1, M_PI * 19 / 20), make_Covariance<C2, TriangleType::lower>(M_PI * M_PI / 9, 0, 0, 0.01)};
  const GaussianDistribution noise {M2::zero(), make_Covariance<TriangleType::lower>(Mat2::identity() * 0.01)};
  Mat2 P_output {1.21 * M_PI * M_PI / 9 + 0.01, 0,
                 0, 0.0121 + 0.01};
  Mat2 cross_output{1.1 * M_PI * M_PI / 9, 0,
                    0, 0.011};
  const GaussianDistribution output {GaussianDistribution {M2(1.1, 1.1 * 19 / 20 * M_PI - 2 * M_PI), P_output}};
  const LinearTransform t {a, an};
  auto [out, cross] = t(input, noise);
  EXPECT_TRUE(is_near(out, output));
  EXPECT_TRUE(is_near(cross, cross_output));

}

