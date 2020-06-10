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
#include "transforms/classes/LinearizedTransform.h"

using namespace OpenKalman;

using Axis2 = Coefficients<Axis, Axis>;
using M2 = Mean<Axis2>;
using Mat2 = TypedMatrix<Axis2, Axis2>;

TEST_F(transform_tests, linearized_augmented_order1)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  Mat2 P_output {30, 20,
                 20, 30};
  Mat2 cross_output {1, 4,
                     2, 3};
  const LinearizedTransform t {LinearTransformation(a, an)};
  //
  const GaussianDistribution input {M2(1, 2), Mat2::identity()};
  const GaussianDistribution noise {M2::zero(), Mat2::identity()};
  const GaussianDistribution output {M2(5, 10), P_output};
  const auto full_output = std::tuple {output, cross_output};
  EXPECT_TRUE(is_near(t(input, noise), full_output));
  //
  const GaussianDistribution input2 {M2(1, 2), make_Covariance<TriangleType::lower>(Mat2::identity())};
  const GaussianDistribution noise2 {M2::zero(), make_Covariance<TriangleType::lower>(Mat2::identity())};
  const GaussianDistribution output2 {M2(5, 10), make_Covariance<TriangleType::lower>(P_output)};
  const auto full_output2 = std::tuple {output2, cross_output};
  EXPECT_TRUE(is_near(t(input2, noise2), full_output2));
}

TEST_F(transform_tests, linearized_augmented_order2)
{
  Mat2 a {1, 2,
          4, 3};
  Mat2 an {3, 4,
           2, 1};
  Mat2 P_output {30, 20,
                 20, 30};
  Mat2 cross_output {1, 4,
                     2, 3};
  auto trans = LinearTransformation(a, an);
  const LinearizedTransform<decltype(trans), 2> t {trans};
  //
  const GaussianDistribution input {M2(1, 2), Mat2::identity()};
  const GaussianDistribution noise {M2::zero(), Mat2::identity()};
  const GaussianDistribution output {M2(5, 10), P_output};
  const auto full_output = std::tuple {output, cross_output};
  EXPECT_TRUE(is_near(t(input, noise), full_output));
  //
  const GaussianDistribution input2 {M2(1, 2), make_Covariance<TriangleType::lower>(Mat2::identity())};
  const GaussianDistribution noise2 {M2::zero(), make_Covariance<TriangleType::lower>(Mat2::identity())};
  const GaussianDistribution output2 {M2(5, 10), make_Covariance<TriangleType::lower>(P_output)};
  const auto full_output2 = std::tuple {output2, cross_output};
  EXPECT_TRUE(is_near(t(input2, noise2), full_output2));
}
