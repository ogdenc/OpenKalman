/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef LINEAR_TESTS_H
#define LINEAR_TESTS_H

#include "../tests.h"
#include "../independent-noise.h"

using namespace OpenKalman;

struct linear_tests : public ::testing::Test
{
  linear_tests() {}

  void SetUp() override {}

  void TearDown() override {}

  ~linear_tests() override {}

  template<typename TransformationMatrix, typename Transform, typename InputDist, typename ... Noise>
  void run_linear_tests(
    const TransformationMatrix& a,
    const Transform& t,
    const InputDist& in,
    const Noise& ... noise)
  {
    const auto x = mean(in);
    const auto p = covariance(in);
    const auto y = g(x, mean(noise)...);
    const auto cross_cov = p*adjoint(a);
    const auto cov = (Covariance {a*cross_cov} + ... + covariance(noise));
    const std::tuple out_true {GaussianDistribution {y, cov}, cross_cov};
    EXPECT_TRUE(is_near(t(in, noise...), out_true));
  }

  template<
    typename Scalar,
    typename InCoeff,
    typename OutCoeff,
    NoiseType noise_type>
  std::tuple<const VectorTransformation<Scalar, InCoeff, OutCoeff, noise_type>, const Mean<Scalar, OutCoeff, InCoeff::size>>
  linear_function()
  {
    const Mean<Scalar, OutCoeff, InCoeff::size> A = Eigen::Matrix<Scalar, OutCoeff::size, InCoeff::size>::Random();
    const Mean<Scalar, OutCoeff> b = Eigen::Matrix<Scalar, OutCoeff::size, 1>::Random();
    const auto linear_function =
        [=] (const Mean<Scalar, InCoeff>& x) noexcept -> Mean<Scalar, OutCoeff> {
            return A * x + b;
        };
    return {VectorTransformation<Scalar, InCoeff, OutCoeff, noise_type> {linear_function}, A};
  }

  template<
    template<typename, typename> typename Dist,
    typename Scalar,
    typename InCoeff,
    typename OutCoeff,
    NoiseType noise_type,
    int count,
    template<typename, typename, typename, NoiseType> typename Transformation,
    template<template<typename, typename> typename, typename, typename, int> typename SamplePoints>
  void construct_linear_test(
    const SamplePoints<Dist, Scalar, InCoeff, count>& s,
    const Transformation<Scalar, InCoeff, OutCoeff, noise_type>& g,
    const Mean<Scalar, OutCoeff, InCoeff::size>& A,
    const int iterations)
  {
    SamplePointsTransform t {s, g};
    IndependentNoise<Dist, Scalar, InCoeff> dist {10};
    IndependentNoise<Dist, Scalar, OutCoeff> noise;
    for (int i=1; i<iterations; i++)
    {
        run_linear_tests(t, g, dist(), A, noise());
    }
  }

};


#endif //LINEAR_TESTS_H
