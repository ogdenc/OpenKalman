/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef KALMAN_TESTS_H
#define KALMAN_TESTS_H

#include <array>
#include <initializer_list>
#include <iostream>
#include <gtest/gtest.h>

#include "../tests.h"
#include "../independent-noise.h"
#include "../transformations.h"
#include "OpenKalman.h"
#include "transforms/OpenKalman-transforms"
#include "filters/KalmanFilter.h"

using namespace OpenKalman;

class kalman_tests : public ::testing::Test
{
public:
  std::random_device rd;
  std::mt19937 gen;

  kalman_tests()
  {
    gen = std::mt19937(rd());
  }

  void SetUp() override {}

  void TearDown() override {}

  ~kalman_tests() override {}

  template<
      template<typename, typename> typename Dist,
      typename Scalar,
      typename StateCoeffs,
      typename MeasurementCoeffs>
  void parameterTest(
      const KalmanFilter<Dist, Scalar, StateCoeffs, MeasurementCoeffs, std::tuple<>, std::tuple<Dist<Scalar, MeasurementCoeffs>>>& filter,
      const Dist<Scalar, MeasurementCoeffs>& measurement_noise,
      const Mean<Scalar, StateCoeffs>& true_state,
      const Dist<Scalar, StateCoeffs>& state_0,
      const Scalar resolution,
      const int iterations = 100)
  {
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "true_state" << std::endl << true_state << std::endl << std::flush;
    std::cout << "state_0" << std::endl << state_0 << std::endl << std::flush;
    int count = 0;
    Scalar norm = std::numeric_limits<Scalar>::infinity();

    Dist<Scalar, StateCoeffs> x {state_0};
    while (norm > resolution && count++ < iterations)
    {
      const auto z {measurement_noise()};
      const Dist<Scalar, MeasurementCoeffs> n =
          GaussianDistribution<Scalar, MeasurementCoeffs> {Mean<Scalar, MeasurementCoeffs>::Zero(), covariance(measurement_noise)};

      x = filter.predict_update(x, z, n);

      std::cout << "state_" << count - 1 << " update" << std::endl << x << std::endl << std::flush;

      norm = (mean(x) - true_state).norm() / std::sqrt(StateCoeffs::size);
      std::cout << "sqnorm: " << norm << std::endl << "----" << std::endl << std::flush;
    }
    EXPECT_LT(count, iterations);
    EXPECT_LE(norm, resolution);
  }


  template<
      template<template<typename, typename> typename, typename, typename, typename, NoiseType, typename ...> typename MeasurementTransform,
      template<typename, typename> typename Dist,
      typename Scalar,
      typename StateCoeffs,
      typename MeasurementCoeffs,
      NoiseType measurement_noise_t = NoiseType::additive,
      typename ... Noise>
  void parameterTestSet(
      const MeasurementTransform<Dist, Scalar, StateCoeffs, MeasurementCoeffs, measurement_noise_t, Noise ...>& measurement_transform,
      const Mean<Scalar, StateCoeffs>& min_state,
      const Mean<Scalar, StateCoeffs>& max_state,
      const Scalar noise,
      const Scalar resolution,
      const int tests = 20,
      const int iterations = 100)
  {
    using State = Mean<Scalar, StateCoeffs>;
    RecursiveLeastSquaresTransform<Dist, Scalar, StateCoeffs, StateCoeffs, NoiseType::none> state_transform {0.995};

    const auto max_distance = (max_state - min_state).cwiseAbs();
    const Dist<Scalar, StateCoeffs> state_0 = GaussianDistribution<Scalar, StateCoeffs> {(max_state + min_state) / 2,
                                                                  max_distance.cwiseProduct(max_distance).asDiagonal()};

    IndependentNoise<Dist, Scalar, StateCoeffs> rand_dist {};
    for (int i = 0; i < tests; i++)
    {
      const State true_state {mean(rand_dist()).cwiseProduct(max_distance) + min_state};

      const KalmanFilter filter(state_transform, measurement_transform);

      const auto[measurement_noise, _] = measurement_transform(
          GaussianDistribution<Scalar, StateCoeffs> {true_state, (noise * noise * Mean<Scalar, StateCoeffs, StateCoeffs::size>::Identity())},
          GaussianDistribution<Scalar, MeasurementCoeffs>
              {
                  Mean<Scalar, MeasurementCoeffs>::Zero(),
                  resolution * resolution * Mean<Scalar, MeasurementCoeffs, MeasurementCoeffs::size>::Identity()
              });
      std::cout << "measurement_noise" << std::endl << measurement_noise << std::endl << "----" << std::endl
          << std::flush;
      parameterTest(filter, measurement_noise, true_state, state_0, resolution, iterations);
    }

  }

};


#endif //KALMAN_TESTS_H
