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

#include <iostream>
#include <random>

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "../tests.h"
#include "../independent-noise.h"
#include "distributions/GaussianDistribution.h"
#include "transforms/classes/SamplePointsTransform.h"

using namespace OpenKalman;

struct linear_tests : public ::testing::Test
{
  linear_tests() {}

  void SetUp() override {}

  void TearDown() override {}

  ~linear_tests() override {}

  template<
    template<typename, typename, typename, NoiseType, typename...> typename Transformation,
    template<template<typename, typename> typename, typename, typename, typename, NoiseType, typename...> typename Transform,
    template<typename, typename> typename Dist,
    typename Scalar,
    typename InCoeff,
    typename OutCoeff,
    NoiseType noise_type,
    typename ... order,
    typename ... Noise>
  void run_linear_tests(
    const Transform<Dist, Scalar, InCoeff, OutCoeff, noise_type, Noise...>& t,
    const Transformation<Scalar, InCoeff, OutCoeff, noise_type, order...>& g,
    const Dist<Scalar, InCoeff>& in,
    const Mean<Scalar, OutCoeff, InCoeff::size>& A,
    const Noise& ... noise)
  {
    const auto X = mean(in);
    const auto P = covariance(in);
    const auto Y = g(X, mean(noise)...);
    const Eigen::Matrix<Scalar, InCoeff::size, OutCoeff::size> XY = P*A.adjoint();
    const Covariance YY_ {A*XY};
    const auto YY = (YY_ + ... + covariance(noise));
    const auto out_d_true = Dist<Scalar, OutCoeff> {GaussianDistribution<Scalar, OutCoeff> {Y, YY}};
    const std::tuple out_true {out_d_true, XY};

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
