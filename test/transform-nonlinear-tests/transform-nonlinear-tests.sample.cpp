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

using C2 = Coefficients<Axis, Axis>;
using M2 = Mean<C2>;
using Mat2 = TypedMatrix<C2,C2>;

TEST_F(transform_tests, Transform_none)
{
  Mat2 a {1, 2,
          3, 4};
  const LinearTransformation g {a};
  const GaussianDistribution input {M2(1, 2), Mat2::Identity()};
  const Covariance<C2> P_output = {5, 11,
                                   11, 25};
  const TypedMatrix<C2, C2> cross_output {1, 3,
                                          2, 4};
  const GaussianDistribution output {M2(5, 11), P_output};
  const auto full_output = std::tuple {output, cross_output};

  const SamplePointsTransform<UnscentedSigmaPoints, decltype(g)> t {g};
  EXPECT_TRUE(is_near(t(input), full_output));

  const SamplePointsTransform<CubaturePoints, decltype(g)> t2 {g};
  EXPECT_TRUE(is_near(t2(input), full_output));

  const SamplePointsTransform<SphericalSimplexSigmaPoints, decltype(g)> t3 {g};
  EXPECT_TRUE(is_near(t3(input), full_output));
}

TEST_F(transform_tests, Transform_Cholesky_none)
{
  const Mat2 a {1, 2,
                3, 4};
  const LinearTransformation g {a};
  const GaussianDistribution input {M2(1, 2), make_Covariance<TriangleType::lower>(Mat2::identity())};
  const Covariance<C2> P_output = EigenTriangularMatrix {5, 11,
                                                         11, 25};
  const Mat2 cross_output {1, 3,
                           2, 4};
  const GaussianDistribution output {GaussianDistribution {M2(5, 11), P_output}};
  const auto full_output = std::tuple {output, cross_output};

  const auto t1 = make_SamplePointsTransform<UnscentedSigmaPoints>(g);
  EXPECT_TRUE(is_near(t1(input), full_output));

  const SamplePointsTransform<CubaturePoints, decltype(g)> t2 {g};
  EXPECT_TRUE(is_near(t2(input), full_output));

  const SamplePointsTransform<SphericalSimplexSigmaPoints, decltype(g)> t3 {g};
  EXPECT_TRUE(is_near(t3(input), full_output));
}
