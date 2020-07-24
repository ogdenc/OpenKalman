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

TEST_F(linear_tests, Linear2x2LinearSA)
{
  for (int i=1; i<=20; i++)
  {
    auto a = randomize<Mat2, std::uniform_real_distribution>(-i*10., i*10.);
    auto n = randomize<Mat2, std::uniform_real_distribution>(-i, i);
    auto g = LinearTransformation(a, n);
    auto t = LinearTransform {g};
    auto in = GaussianDistribution {Mean2::zero(), i * CovSA2 {1.2, 0.2, 0.2, 2.1}};
    auto b = randomize<Mean2, std::normal_distribution>(0, i*2.);
    auto noise = GaussianDistribution {b, i / 5. * CovSA2::identity()};
    EXPECT_TRUE(run_linear_test(g, t, in, noise));
  }
}

TEST_F(linear_tests, Linear2x2LinearT)
{
  for (int i=1; i<=20; i++)
  {
    auto a = randomize<Mat2, std::uniform_real_distribution>(-i*10., i*10.);
    auto n = randomize<Mat2, std::uniform_real_distribution>(-i, i);
    auto g = LinearTransformation(a, n);
    auto t = LinearTransform {g};
    auto in = GaussianDistribution {Mean2::zero(), i * CovT2 {1.2, 0.2, 0.2, 2.1}};
    auto b = randomize<Mean2, std::normal_distribution>(0, i*2.);
    auto noise = GaussianDistribution {b, i / 5. * CovT2::identity()};
    EXPECT_TRUE(run_linear_test(g, t, in, noise));
  }
}

TEST_F(linear_tests, Linear2x2TT1SA)
{
  for (int i=1; i<=20; i++)
  {
    auto a = randomize<Mat2, std::uniform_real_distribution>(-i*10., i*10.);
    auto n = randomize<Mat2, std::uniform_real_distribution>(-i, i);
    auto g = LinearTransformation(a, n);
    auto t = LinearizedTransform {g};
    auto in = GaussianDistribution {Mean2::zero(), i * CovSA2 {1.2, 0.2, 0.2, 2.1}};
    auto b = randomize<Mean2, std::normal_distribution>(0, i*2.);
    auto noise = GaussianDistribution {b, i / 5. * CovSA2::identity()};
    EXPECT_TRUE(run_linear_test(g, t, in, noise));
  }
}

TEST_F(linear_tests, Linear2x2TT1T)
{
  for (int i=1; i<=20; i++)
  {
    auto a = randomize<Mat2, std::uniform_real_distribution>(-i*10., i*10.);
    auto n = randomize<Mat2, std::uniform_real_distribution>(-i, i);
    auto g = LinearTransformation(a, n);
    auto t = LinearizedTransform {g};
    auto in = GaussianDistribution {Mean2::zero(), i * CovT2 {1.2, 0.2, 0.2, 2.1}};
    auto b = randomize<Mean2, std::normal_distribution>(0, i*2.);
    auto noise = GaussianDistribution {b, i / 5. * CovT2::identity()};
    EXPECT_TRUE(run_linear_test(g, t, in, noise));
  }
}

TEST_F(linear_tests, Linear2x2TT2SA)
{
  for (int i=1; i<=20; i++)
  {
    auto a = randomize<Mat2, std::uniform_real_distribution>(-i*10., i*10.);
    auto n = randomize<Mat2, std::uniform_real_distribution>(-i, i);
    auto g = LinearTransformation(a, n);
    auto t = LinearizedTransform<decltype(g), 2> {g};
    auto in = GaussianDistribution {Mean2::zero(), i * CovSA2 {1.2, 0.2, 0.2, 2.1}};
    auto b = randomize<Mean2, std::normal_distribution>(0, i*2.);
    auto noise = GaussianDistribution {b, i / 5. * CovSA2::identity()};
    EXPECT_TRUE(run_linear_test(g, t, in, noise));
  }
}

TEST_F(linear_tests, Linear2x2TT2T)
{
  for (int i=1; i<=20; i++)
  {
    auto a = randomize<Mat2, std::uniform_real_distribution>(-i*10., i*10.);
    auto n = randomize<Mat2, std::uniform_real_distribution>(-i, i);
    auto g = LinearTransformation(a, n);
    auto t = LinearizedTransform<decltype(g), 2> {g};
    auto in = GaussianDistribution {Mean2::zero(), i * CovT2 {1.2, 0.2, 0.2, 2.1}};
    auto b = randomize<Mean2, std::normal_distribution>(0, i*2.);
    auto noise = GaussianDistribution {b, i / 5. * CovT2::identity()};
    EXPECT_TRUE(run_linear_test(g, t, in, noise));
  }
}
