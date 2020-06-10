/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "nonlinear-sqrt-tests.h"
#include <transforms/classes/LinearizedTransform.h>

using namespace OpenKalman;

template<
    typename Scalar,
    typename InCoeffs,
    typename OutCoeffs,
    NoiseType noise_type,
    int order>
::testing::AssertionResult sqrtComparison(
    const LinearizedTransformation<Scalar, InCoeffs, OutCoeffs, noise_type, order>& transformation,
    const GaussianDistribution<Scalar, InCoeffs>& d)
{
  const LinearizedTransform t {transformation};
  if constexpr (noise_type == NoiseType::none)
  {
    const LinearizedTransform<SquareRootGaussianDistribution, Scalar, InCoeffs, OutCoeffs, noise_type> t_sqrt {
        transformation};
    const auto res = std::get<0>(t(d));
    const auto res_sqrt = std::get<0>(t_sqrt(SquareRootGaussianDistribution<Scalar, InCoeffs> {d}));
    return is_near(res, res_sqrt);
  }
  else
  {
    const auto id = Covariance<Scalar, OutCoeffs>::Matrix::Identity();
    const LinearizedTransform<SquareRootGaussianDistribution, Scalar, InCoeffs, OutCoeffs,
        noise_type, SquareRootGaussianDistribution<Scalar, OutCoeffs>> t_sqrt {transformation};
    const auto res = std::get<0>(t(d, GaussianDistribution<Scalar, OutCoeffs> {id}));
    const auto res_sqrt = std::get<0>(t_sqrt(SquareRootGaussianDistribution<Scalar, InCoeffs> {d},
        SquareRootGaussianDistribution<Scalar, OutCoeffs> {id}));
    return is_near(res, res_sqrt);
  }
}

using C2 = Coefficients<Axis, Axis>;
IndependentNoise<GaussianDistribution, double, C2> noise;

TEST_F(nonlinear_sqrt_tests, TT1Sqrt2x2None)
{
  for (int i = 0; i < 20; i++)
  {
    auto res = sqrtComparison<double, C2, C2, NoiseType::none, 1>(
        LinearizedTransformation<double, C2, C2, NoiseType::none, 1> {radar<double>}, noise());
    EXPECT_TRUE(res);
  }
}

TEST_F(nonlinear_sqrt_tests, TT2Sqrt2x2None)
{
  for (int i = 0; i < 20; i++)
  {
    auto res = sqrtComparison<double, C2, C2, NoiseType::none, 2>(radar<double>, noise());
    EXPECT_TRUE(res);
  }
}

TEST_F(nonlinear_sqrt_tests, TT1Sqrt2x2Add)
{
  for (int i = 0; i < 20; i++)
  {
    auto res = sqrtComparison<double, C2, C2, NoiseType::additive, 1>(radar<double>, noise());
    EXPECT_TRUE(res);
  }
}

TEST_F(nonlinear_sqrt_tests, TT2Sqrt2x2Add)
{
  for (int i = 0; i < 20; i++)
  {
    auto res = sqrtComparison<double, C2, C2, NoiseType::additive, 2>(radar<double>, noise());
    EXPECT_TRUE(res);
  }
}

TEST_F(nonlinear_sqrt_tests, TT1Sqrt2x2Aug)
{
  for (int i = 0; i < 20; i++)
  {
    auto res = sqrtComparison<double, C2, C2, NoiseType::augmented, 1>(radar_aug<double>, noise());
    EXPECT_TRUE(res);
  }
}

TEST_F(nonlinear_sqrt_tests, TT2Sqrt2x2Aug)
{
  for (int i = 0; i < 20; i++)
  {
    auto res = sqrtComparison<double, C2, C2, NoiseType::augmented, 2>(radar_aug<double>, noise());
    EXPECT_TRUE(res);
  }
}
