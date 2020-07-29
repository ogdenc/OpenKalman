/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear-tests.h"

using C2 = Coefficients<Axis, Axis>;
using M21 = Eigen::Matrix<double, 2, 1>;
using M22 = Eigen::Matrix<double, 2, 2>;
using Mean2 = Mean<C2, M21>;
using Mat2 = TypedMatrix<C2, C2, M22>;
using CovSA2 = Covariance<C2, EigenSelfAdjointMatrix<M22>>;
using CovT2 = Covariance<C2, EigenTriangularMatrix<M22>>;
using Scalar = double;

TEST_F(linear_tests, Linear2x2UnscentedSA)
{
  for (int i=1; i<=20; i++)
  {
    auto a = randomize<Mat2, std::uniform_real_distribution>(-i*10., i*10.);
    auto n = randomize<Mat2, std::uniform_real_distribution>(-double(i), double(i));
    auto g = LinearTransformation(a, n);
    auto t = make_SamplePointsTransform<UnscentedSigmaPoints>(g);
    static_assert(is_strict_v<decltype(CovSA2 {1.2, 0.2, 0.2, 2.1})>);
    static_assert(not is_strict_v<decltype(i * CovSA2 {1.2, 0.2, 0.2, 2.1})>);
    auto in = GaussianDistribution {Mean2::zero(), strict(i * CovSA2 {1.2, 0.2, 0.2, 2.1})};
    auto b = randomize<Mean2, std::normal_distribution>(0., i*2.);
    auto noise = GaussianDistribution {b, i / 5. * CovSA2::identity()};
    EXPECT_TRUE(run_linear_test(g, t, in, noise));
  }
}

TEST_F(linear_tests, Linear2x2UnscentedT)
{
  for (int i=1; i<=20; i++)
  {
    auto a = randomize<Mat2, std::uniform_real_distribution>(-i*10., i*10.);
    auto n = randomize<Mat2, std::uniform_real_distribution>(-double(i), double(i));
    auto g = LinearTransformation(a, n);
    auto t = make_SamplePointsTransform<UnscentedSigmaPoints>(g);
    auto in = GaussianDistribution {Mean2::zero(), strict(i * CovT2 {1.2, 0.2, 0.2, 2.1})};
    auto b = randomize<Mean2, std::normal_distribution>(0., i*2.);
    auto noise = GaussianDistribution {b, i / 5. * CovT2::identity()};
    EXPECT_TRUE(run_linear_test(g, t, in, noise));
  }
}

TEST_F(linear_tests, Linear2x2UnscentedParamSA)
{
  for (int i=1; i<=20; i++)
  {
    auto a = randomize<Mat2, std::uniform_real_distribution>(-i*10., i*10.);
    auto n = randomize<Mat2, std::uniform_real_distribution>(-double(i), double(i));
    auto g = LinearTransformation(a, n);
    auto t = make_SamplePointsTransform<UnscentedSigmaPointsParameterEstimation>(g);
    auto in = GaussianDistribution {Mean2::zero(), strict(i * CovSA2 {1.2, 0.2, 0.2, 2.1})};
    auto b = randomize<Mean2, std::normal_distribution>(0., i*2.);
    auto noise = GaussianDistribution {b, i / 5. * CovSA2::identity()};
    EXPECT_TRUE(run_linear_test(g, t, in, noise));
  }
}

TEST_F(linear_tests, Linear2x2UnscentedParamT)
{
  for (int i=1; i<=20; i++)
  {
    auto a = randomize<Mat2, std::uniform_real_distribution>(-i*10., i*10.);
    auto n = randomize<Mat2, std::uniform_real_distribution>(-double(i), double(i));
    auto g = LinearTransformation(a, n);
    auto t = make_SamplePointsTransform<UnscentedSigmaPointsParameterEstimation>(g);
    auto in = GaussianDistribution {Mean2::zero(), strict(i * CovT2 {1.2, 0.2, 0.2, 2.1})};
    auto b = randomize<Mean2, std::normal_distribution>(0., i*2.);
    auto noise = GaussianDistribution {b, i / 5. * CovT2::identity()};
    EXPECT_TRUE(run_linear_test(g, t, in, noise));
  }
}

TEST_F(linear_tests, Linear2x2UnscentedSphericalSA)
{
  for (int i=1; i<=20; i++)
  {
    auto a = randomize<Mat2, std::uniform_real_distribution>(-i*10., i*10.);
    auto n = randomize<Mat2, std::uniform_real_distribution>(-double(i), double(i));
    auto g = LinearTransformation(a, n);
    auto t = make_SamplePointsTransform<SphericalSimplexSigmaPoints>(g);
    auto in = GaussianDistribution {Mean2::zero(), strict(i * CovSA2 {1.2, 0.2, 0.2, 2.1})};
    auto b = randomize<Mean2, std::normal_distribution>(0., i*2.);
    auto noise = GaussianDistribution {b, i / 5. * CovSA2::identity()};
    EXPECT_TRUE(run_linear_test(g, t, in, noise));
  }
}

TEST_F(linear_tests, Linear2x2UnscentedSphericalT)
{
  for (int i=1; i<=20; i++)
  {
    auto a = randomize<Mat2, std::uniform_real_distribution>(-i*10., i*10.);
    auto n = randomize<Mat2, std::uniform_real_distribution>(-double(i), double(i));
    auto g = LinearTransformation(a, n);
    auto t = make_SamplePointsTransform<SphericalSimplexSigmaPoints>(g);
    auto in = GaussianDistribution {Mean2::zero(), strict(i * CovT2 {1.2, 0.2, 0.2, 2.1})};
    auto b = randomize<Mean2, std::normal_distribution>(0., i*2.);
    auto noise = GaussianDistribution {b, i / 5. * CovT2::identity()};
    EXPECT_TRUE(run_linear_test(g, t, in, noise));
  }
}

