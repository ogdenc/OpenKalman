/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transform-linear.hpp"

using namespace OpenKalman;

using Axis2 = Coefficients<Axis, Axis>;
using M2 = Mean<Axis2>;
using Mat2 = Matrix<Axis2, Axis2>;

TEST_F(transform_linear, linear_additive)
{
  Mat2 a {1, 2,
          3, 4};
  LinearTransformation g {a};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), Covariance(Mat2::identity())};
  GaussianDistribution noise {M2::zero(), Covariance(Mat2::identity())};
  Mat2 P_output {5, 11,
                 11, 25};
  Mat2 cross_output {1, 3,
                     2, 4};
  auto [out, cross] = t.transform_with_cross_covariance(input, g);
  EXPECT_TRUE(is_near(mean_of(out), M2(5, 11)));
  EXPECT_TRUE(is_near(covariance_of(out), P_output));
  EXPECT_TRUE(is_near(out + noise, GaussianDistribution {M2(5, 11), Covariance(P_output + Mat2::identity())}));
  EXPECT_TRUE(is_near(cross, cross_output));
}

TEST_F(transform_linear, linear_dual)
{
  Mat2 a {1, 2,
          3, 4};
  LinearTransformation g {a};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), Covariance(Mat2::identity())};
  GaussianDistribution noise {M2::zero(), Covariance(Mat2::identity())};
  auto [out1, cross, y] = t.transform_with_cross_covariance(input, std::tuple {g}, std::tuple {g});
  EXPECT_TRUE(is_near(mean_of(out1), M2(27, 59)));
  EXPECT_TRUE(is_near(covariance_of(out1), Mat2 {149, 325, 325, 709}));
  EXPECT_TRUE(is_near(cross, Mat2 {27, 59, 61, 133}));
  EXPECT_TRUE(is_near(covariance_of(y), Mat2 {5, 11, 11, 25}));
}

TEST_F(transform_linear, linear_additive_Cholesky)
{
  Mat2 a {1, 2,
          3, 4};
  LinearTransformation g {a};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), make_Covariance<TriangleType::lower>(Mat2::identity())};
  GaussianDistribution noise {M2::zero(), make_Covariance<TriangleType::lower>(Mat2::identity())};
  Mat2 P_output {5, 11,
                 11, 25};
  Mat2 cross_output {1, 3,
                     2, 4};
  auto [out, cross] = t.transform_with_cross_covariance(input, g);
  EXPECT_TRUE(is_near(mean_of(out), M2(5, 11)));
  EXPECT_TRUE(is_near(covariance_of(out), P_output));
  EXPECT_TRUE(is_near(out + noise, GaussianDistribution {M2(5, 11), Covariance(P_output + Mat2::identity())}));
  EXPECT_TRUE(is_near(cross, cross_output));
}

TEST_F(transform_linear, linear_dual_Cholesky)
{
  Mat2 a {1, 2,
          3, 4};
  LinearTransformation g {a};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), make_Covariance<TriangleType::lower>(Mat2::identity())};
  GaussianDistribution noise {M2::zero(), make_Covariance<TriangleType::lower>(Mat2::identity())};
  auto [out1, cross, y] = t.transform_with_cross_covariance(input, std::tuple {g}, std::tuple {g});
  EXPECT_TRUE(is_near(mean_of(out1), M2(27, 59)));
  EXPECT_TRUE(is_near(covariance_of(out1), Mat2 {149, 325, 325, 709}));
  EXPECT_TRUE(is_near(cross, Mat2 {27, 59, 61, 133}));
  EXPECT_TRUE(is_near(covariance_of(y), Mat2 {5, 11, 11, 25}));
}

TEST_F(transform_linear, linear_augmented)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  LinearTransformation g {a, an};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), Covariance(Mat2::identity())};
  GaussianDistribution noise {M2::zero(), Covariance(Mat2::identity())};
  Mat2 P_output {30, 20,
                 20, 30};
  Mat2 cross_output {1, 4,
                     2, 3};
  EXPECT_TRUE(is_near(t.transform_with_cross_covariance(input, g, noise),
    std::tuple {GaussianDistribution {M2(5, 10), Covariance(P_output)}, cross_output}));
}

TEST_F(transform_linear, linear_dual_augmented)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  LinearTransformation g {a, an};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), Covariance(Mat2::identity())};
  GaussianDistribution noise {M2::zero(), Covariance(Mat2::identity())};
  auto [out1, cross, y] = t.transform_with_cross_covariance(input, std::tuple {g, noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(mean_of(out1), M2(25, 50)));
  EXPECT_TRUE(is_near(covariance_of(out1), Mat2 {255, 530, 530, 1235}));
  EXPECT_TRUE(is_near(cross, Mat2 {70, 180, 80, 170}));
  EXPECT_TRUE(is_near(covariance_of(y), Mat2 {30, 20, 20, 30}));
}

TEST_F(transform_linear, linear_augmented_Cholesky)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  LinearTransformation g {a, an};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), make_Covariance<TriangleType::lower>(Mat2::identity())};
  GaussianDistribution noise {M2::zero(), make_Covariance<TriangleType::lower>(Mat2::identity())};
  Mat2 P_output {30, 20,
                 20, 30};
  Mat2 cross_output {1, 4,
                     2, 3};
  EXPECT_TRUE(is_near(t.transform_with_cross_covariance(input, g, noise),
    std::tuple {GaussianDistribution {M2(5, 10), Covariance(P_output)}, cross_output}));
}

TEST_F(transform_linear, linear_dual_augmented_Cholesky)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  LinearTransformation g {a, an};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), make_Covariance<TriangleType::lower>(Mat2::identity())};
  GaussianDistribution noise {M2::zero(), make_Covariance<TriangleType::lower>(Mat2::identity())};
  auto [out1, cross, y] = t.transform_with_cross_covariance(input, std::tuple {g, noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(mean_of(out1), M2(25, 50)));
  EXPECT_TRUE(is_near(covariance_of(out1), Mat2 {255, 530, 530, 1235}));
  EXPECT_TRUE(is_near(cross, Mat2 {70, 180, 80, 170}));
  EXPECT_TRUE(is_near(covariance_of(y), Mat2 {30, 20, 20, 30}));
}

