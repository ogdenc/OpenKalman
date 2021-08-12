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

using namespace OpenKalman;
using namespace OpenKalman::test;

using Axis2 = Coefficients<Axis, Axis>;
using M2 = Mean<Axis2>;
using Mat2 = Matrix<Axis2, Axis2>;


TEST(transform_linear, linearized_additive1)
{
  Mat2 a {1, 2,
          4, 3};
  LinearTransformation g {a};
  LinearizedTransform t;
  //
  const GaussianDistribution input0 {M2(1, 2), Mat2::identity()};
  const GaussianDistribution noise0 {M2::zero(), Mat2::identity()};
  const GaussianDistribution output0 {M2(5, 10),  Mat2 {6, 10, 10, 26}};
  auto [out0, cross0] = t.transform_with_cross_covariance(input0, g);
  EXPECT_TRUE(is_near(out0 + noise0, output0));
  EXPECT_TRUE(is_near(cross0, Mat2 {1, 4, 2, 3}));
  //
  const GaussianDistribution input1 {M2(1, 2), make_covariance<TriangleType::lower>(Mat2::identity())};
  const GaussianDistribution noise1 {M2::zero(), make_covariance<TriangleType::lower>(Mat2::identity())};
  const GaussianDistribution output1 {M2(5, 10), make_covariance<TriangleType::lower>( Mat2 {6, 10, 10, 26})};
  auto [out1, cross1] = t.transform_with_cross_covariance(input1, g);
  EXPECT_TRUE(is_near(out1 + noise1, output1));
  EXPECT_TRUE(is_near(cross1, Mat2 {1, 4, 2, 3}));
  //
  const GaussianDistribution input2 {M2(1, 2), make_covariance<TriangleType::lower>(Mat2 {2, 1, 1, 2})};
  const GaussianDistribution noise2 {M2::zero(), make_covariance<TriangleType::lower>(Mat2 {1, 0, 0, 1})};
  auto [out2, cross2] = t.transform_with_cross_covariance(input2, g);
  EXPECT_TRUE(is_near(out2 + noise2, GaussianDistribution{M2(5, 10), Mat2 {15, 31, 31, 75}}));
  EXPECT_TRUE(is_near(cross2, Mat2 {4, 11, 5, 10}));

  const GaussianDistribution input3 {M2(1, 2), make_covariance<TriangleType::upper>(Mat2 {2, 1, 1, 2})};
  const GaussianDistribution noise3 {M2::zero(), make_covariance<TriangleType::upper>(Mat2 {1, 0, 0, 1})};
  auto [out3, cross3] = t.transform_with_cross_covariance(input3, g);
  EXPECT_TRUE(is_near(out3 + noise3, GaussianDistribution{M2(5, 10), Mat2 {15, 31, 31, 75}}));
  EXPECT_TRUE(is_near(cross3, Mat2 {4, 11, 5, 10}));

  const GaussianDistribution input4 {M2(1, 2), DiagonalMatrix(1., 1)};
  const GaussianDistribution noise4 {M2::zero(), DiagonalMatrix(1., 1)};
  auto [out4, cross4] = t.transform_with_cross_covariance(input4, g);
  EXPECT_TRUE(is_near(out4 + noise4, GaussianDistribution{M2(5, 10), Mat2 {6, 10, 10, 26}}));
  EXPECT_TRUE(is_near(cross4, Mat2 {1, 4, 2, 3}));
}


TEST(transform_linear, linearized_dual)
{
  Mat2 a {1, 2,
          3, 4};
  LinearTransformation g {a};
  LinearizedTransform t;
  GaussianDistribution input {M2(1, 2), Covariance(Mat2::identity())};
  GaussianDistribution noise {M2::zero(), Covariance(Mat2::identity())};
  auto [out1, cross] = t.transform_with_cross_covariance(input, std::tuple {g}, std::tuple {g});
  EXPECT_TRUE(is_near(mean_of(out1), M2(27, 59)));
  EXPECT_TRUE(is_near(covariance_of(out1), Mat2 {149, 325, 325, 709}));
  EXPECT_TRUE(is_near(cross, Mat2 {27, 59, 61, 133}));
}
