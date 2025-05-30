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
#include "collections/concepts/tuple_like.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;
using namespace OpenKalman::test;

using Axis2 = Dimensions<2>;
using M2 = Mean<Axis2>;
using Mat2 = Matrix<Axis2, Axis2>;

TEST(transform_linear, linear_additive)
{
  Mat2 a {1, 2,
          3, 4};
  LinearTransformation g {a};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), Covariance(make_identity_matrix_like<Mat2>())};
  GaussianDistribution noise {make_zero<M2>(), Covariance(make_identity_matrix_like<Mat2>())};
  Mat2 P_output {5, 11,
                 11, 25};
  Mat2 cross_output {1, 3,
                     2, 4};
  auto [out, cross] = t.transform_with_cross_covariance(input, g);
  EXPECT_TRUE(is_near(mean_of(out), M2(5, 11)));
  EXPECT_TRUE(is_near(covariance_of(out), P_output));
  EXPECT_TRUE(is_near(out + noise, GaussianDistribution {M2(5, 11), Covariance(P_output + make_identity_matrix_like<Mat2>())}));
  EXPECT_TRUE(is_near(cross, cross_output));
}

TEST(transform_linear, linear_dual)
{
  Mat2 a {1, 2,
          3, 4};
  LinearTransformation g {a};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), Covariance(make_identity_matrix_like<Mat2>())};
  GaussianDistribution noise {make_zero<M2>(), Covariance(make_identity_matrix_like<Mat2>())};
  static_assert(collections::tuple_like<decltype(std::tuple {g})>);
  auto [out1, cross] = t.transform_with_cross_covariance(input, std::tuple {g}, std::tuple {g});
  EXPECT_TRUE(is_near(mean_of(out1), M2(27, 59)));
  EXPECT_TRUE(is_near(covariance_of(out1), Mat2 {149, 325, 325, 709}));
  EXPECT_TRUE(is_near(cross, Mat2 {27, 59, 61, 133}));
}

TEST(transform_linear, linear_additive_Cholesky)
{
  Mat2 a {1, 2,
          3, 4};
  LinearTransformation g {a};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), make_covariance<TriangleType::lower>(make_identity_matrix_like<Mat2>())};
  GaussianDistribution noise {make_zero<M2>(), make_covariance<TriangleType::lower>(make_identity_matrix_like<Mat2>())};
  Mat2 P_output {5, 11,
                 11, 25};
  Mat2 cross_output {1, 3,
                     2, 4};
  auto [out, cross] = t.transform_with_cross_covariance(input, g);
  EXPECT_TRUE(is_near(mean_of(out), M2(5, 11)));
  EXPECT_TRUE(is_near(covariance_of(out), P_output));
  EXPECT_TRUE(is_near(out + noise, GaussianDistribution {M2(5, 11), Covariance(P_output + make_identity_matrix_like<Mat2>())}));
  EXPECT_TRUE(is_near(cross, cross_output));
}

TEST(transform_linear, linear_dual_Cholesky)
{
  Mat2 a {1, 2,
          3, 4};
  LinearTransformation g {a};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), make_covariance<TriangleType::lower>(make_identity_matrix_like<Mat2>())};
  GaussianDistribution noise {make_zero<M2>(), make_covariance<TriangleType::lower>(make_identity_matrix_like<Mat2>())};
  auto [out1, cross] = t.transform_with_cross_covariance(input, std::tuple {g}, std::tuple {g});
  EXPECT_TRUE(is_near(mean_of(out1), M2(27, 59)));
  EXPECT_TRUE(is_near(covariance_of(out1), Mat2 {149, 325, 325, 709}));
  EXPECT_TRUE(is_near(cross, Mat2 {27, 59, 61, 133}));
}

TEST(transform_linear, linear_augmented)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  LinearTransformation g {a, an};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), Covariance(make_identity_matrix_like<Mat2>())};
  GaussianDistribution noise {make_zero<M2>(), Covariance(make_identity_matrix_like<Mat2>())};
  Mat2 P_output {30, 20,
                 20, 30};
  Mat2 cross_output {1, 4,
                     2, 3};
  EXPECT_TRUE(is_near(t.transform_with_cross_covariance(input, g, noise),
    std::tuple {GaussianDistribution {M2(5, 10), Covariance(P_output)}, cross_output}));
}

TEST(transform_linear, linear_dual_augmented)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  LinearTransformation g {a, an};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), Covariance(make_identity_matrix_like<Mat2>())};
  GaussianDistribution noise {make_zero<M2>(), Covariance(make_identity_matrix_like<Mat2>())};
  auto [out1, cross] = t.transform_with_cross_covariance(input, std::tuple {g, noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(mean_of(out1), M2(25, 50)));
  EXPECT_TRUE(is_near(covariance_of(out1), Mat2 {255, 530, 530, 1235}));
  EXPECT_TRUE(is_near(cross, Mat2 {70, 180, 80, 170}));
}

TEST(transform_linear, linear_augmented_Cholesky)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  LinearTransformation g {a, an};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), make_covariance<TriangleType::lower>(make_identity_matrix_like<Mat2>())};
  GaussianDistribution noise {make_zero<M2>(), make_covariance<TriangleType::lower>(make_identity_matrix_like<Mat2>())};
  Mat2 P_output {30, 20,
                 20, 30};
  Mat2 cross_output {1, 4,
                     2, 3};
  EXPECT_TRUE(is_near(t.transform_with_cross_covariance(input, g, noise),
    std::tuple {GaussianDistribution {M2(5, 10), Covariance(P_output)}, cross_output}));
}

TEST(transform_linear, linear_dual_augmented_Cholesky)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  LinearTransformation g {a, an};
  LinearTransform t;
  GaussianDistribution input {M2(1, 2), make_covariance<TriangleType::lower>(make_identity_matrix_like<Mat2>())};
  GaussianDistribution noise {make_zero<M2>(), make_covariance<TriangleType::lower>(make_identity_matrix_like<Mat2>())};
  auto [out1, cross] = t.transform_with_cross_covariance(input, std::tuple {g, noise}, std::tuple {g, noise});
  EXPECT_TRUE(is_near(mean_of(out1), M2(25, 50)));
  EXPECT_TRUE(is_near(covariance_of(out1), Mat2 {255, 530, 530, 1235}));
  EXPECT_TRUE(is_near(cross, Mat2 {70, 180, 80, 170}));
}

