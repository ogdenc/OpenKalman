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
using namespace OpenKalman::coordinates;
using namespace OpenKalman::test;

using Axis2 = Dimensions<2>;
using M2 = Mean<Axis2>;
using Mat2 = Matrix<Axis2, Axis2>;

TEST(transform_linear, linearized_augmented_order1)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  Mat2 P_output {30, 20,
                 20, 30};
  Mat2 cross_output {1, 4,
                     2, 3};
  LinearTransformation g {a, an};
  LinearizedTransform t;
  //
  const GaussianDistribution input {M2(1, 2), make_identity_matrix_like<Mat2>()};
  const GaussianDistribution noise {make_zero<M2>(), make_identity_matrix_like<Mat2>()};
  const GaussianDistribution output {M2(5, 10), P_output};
  const auto full_output = std::tuple {output, cross_output};
  EXPECT_TRUE(is_near(t.transform_with_cross_covariance(input, g, noise), full_output));
  //
  const GaussianDistribution input2 {M2(1, 2), make_covariance<TriangleType::lower>(make_identity_matrix_like<Mat2>())};
  const GaussianDistribution noise2 {make_zero<M2>(), make_covariance<TriangleType::lower>(make_identity_matrix_like<Mat2>())};
  const GaussianDistribution output2 {M2(5, 10), make_covariance<TriangleType::lower>(P_output)};
  const auto full_output2 = std::tuple {output2, cross_output};
  EXPECT_TRUE(is_near(t.transform_with_cross_covariance(input2, g, noise2), full_output2));
}

TEST(transform_linear, linearized_dual_augmented)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  LinearTransformation g {a, an};
  LinearizedTransform t;
  GaussianDistribution input {M2(1, 2), Covariance(make_identity_matrix_like<Mat2>())};
  GaussianDistribution noise {make_zero<M2>(), Covariance(make_identity_matrix_like<Mat2>())};
  auto [out1, cross] = t.transform_with_cross_covariance(input, std::tuple {g, noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(mean_of(out1), M2(25, 50)));
  EXPECT_TRUE(is_near(covariance_of(out1), Mat2 {255, 530, 530, 1235}));
  EXPECT_TRUE(is_near(cross, Mat2 {70, 180, 80, 170}));
}

TEST(transform_linear, linearized_augmented_order2)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  LinearTransformation g {a, an};
  LinearizedTransform<2> t;
  //
  const GaussianDistribution input {M2(1, 2), make_identity_matrix_like<Mat2>()};
  const GaussianDistribution noise {make_zero<M2>(), make_identity_matrix_like<Mat2>()};
  auto [out, cross] = t.transform_with_cross_covariance(input, g, noise);
  EXPECT_TRUE(is_near(out, GaussianDistribution {M2(5, 10), Mat2 {30, 20, 20, 30}}));
  EXPECT_TRUE(is_near(cross, Mat2 {1, 4, 2, 3}));
  //
  const GaussianDistribution input2 {M2(1, 2), make_covariance<TriangleType::lower>(Mat2 {2, 1, 1, 2})};
  const GaussianDistribution noise2 {make_zero<M2>(), make_covariance<TriangleType::lower>(Mat2 {1, 0, 0, 1})};
  auto [out2, cross2] = t.transform_with_cross_covariance(input2, g, noise2);
  EXPECT_TRUE(is_near(out2, GaussianDistribution {M2(5, 10), Mat2 {39, 41, 41, 79}}));
  EXPECT_TRUE(is_near(cross2, Mat2 {4, 11, 5, 10}));

  const GaussianDistribution input3 {M2(1, 2), make_covariance<TriangleType::upper>(Mat2 {2, 1, 1, 2})};
  const GaussianDistribution noise3 {make_zero<M2>(), make_covariance<TriangleType::upper>(Mat2 {1, 0, 0, 1})};
  auto [out3, cross3] = t.transform_with_cross_covariance(input3, g, noise3);
  EXPECT_TRUE(is_near(out3, GaussianDistribution {M2(5, 10), Mat2 {39, 41, 41, 79}}));
  EXPECT_TRUE(is_near(cross3, Mat2 {4, 11, 5, 10}));

  const GaussianDistribution input4 {M2(1, 2), DiagonalAdapter(1., 1)};
  const GaussianDistribution noise4 {make_zero<M2>(), DiagonalAdapter(1., 1)};
  auto [out4, cross4] = t.transform_with_cross_covariance(input4, g, noise4);
  EXPECT_TRUE(is_near(out4, GaussianDistribution {M2(5, 10), Mat2 {30, 20, 20, 30}}));
  EXPECT_TRUE(is_near(cross4, Mat2 {1, 4, 2, 3}));
}
