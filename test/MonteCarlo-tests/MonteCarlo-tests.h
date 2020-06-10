/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef MONTECARLO_TESTS_H
#define MONTECARLO_TESTS_H

#include <iostream>
#include <random>

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "../tests.h"
#include "../independent-noise.h"
#include "../transformations.h"
#include <transforms/classes/MonteCarloTransform.h>
#include "distributions/GaussianDistribution.h"

using namespace OpenKalman;


class MonteCarlo_tests : public ::testing::Test
{
public:
  MonteCarlo_tests()
  {
  }

  void SetUp() override
  {
    // code here will execute just before the test ensues
  }

  void TearDown() override
  {
    // code here will be called just after the test completes
    // ok to throw exceptions from here if need be
  }

  ~MonteCarlo_tests() override
  {
    // cleanup any pending stuff, but no exceptions allowed
  }

  // put in any custom members that you need

  template<
      typename Scalar,
      typename InputCoeffs,
      typename OutputCoeffs,
      NoiseType noise_type>
  ::testing::AssertionResult sqrtComparison(
      const VectorTransformation<Scalar, InputCoeffs, OutputCoeffs, noise_type>& transformation,
      const GaussianDistribution<Scalar, InputCoeffs>& d,
      const Scalar err = 0.01)
  {
    const MonteCarloTransform t {transformation, 1000000};
    if constexpr (noise_type == NoiseType::none)
    {
      const MonteCarloTransform<SquareRootGaussianDistribution, Scalar, InputCoeffs, OutputCoeffs, noise_type> t_sqrt {
          transformation, 1000000};
      const auto res = std::get<0>(t(d));
      const auto res_sqrt = std::get<0>(t_sqrt(SquareRootGaussianDistribution<Scalar, InputCoeffs> {d}));
      return is_near(res, res_sqrt, err);
    }
    else
    {
      const auto id = Covariance<Scalar, OutputCoeffs>::Matrix::Identity();
      const MonteCarloTransform<SquareRootGaussianDistribution, Scalar, InputCoeffs, OutputCoeffs,
          noise_type, SquareRootGaussianDistribution<Scalar, OutputCoeffs>> t_sqrt {transformation, 1000000};
      const auto res = std::get<0>(t(d, GaussianDistribution<Scalar, OutputCoeffs> {id}));
      const auto res_sqrt = std::get<0>(t_sqrt(SquareRootGaussianDistribution<Scalar, InputCoeffs> {d},
          SquareRootGaussianDistribution<Scalar, OutputCoeffs> {id}));
      return is_near(res, res_sqrt, err);
    }
  }


};


#endif //MONTECARLO_TESTS_H
