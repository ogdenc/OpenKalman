/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "kalman-tests.h"

using namespace OpenKalman;

using C2 = Coefficients<Axis, Axis>;
using Vec2 = Mean<double, C2>;
using Mat2 = Mean<double, C2, 2>;

TEST_F(kalman_tests, parameterLTLinear)
{
  const Mat2 A = Mat2::Random();
  const LinearTransformation<double, C2, C2, NoiseType::additive> g {A};
  const LinearTransform measurement_transform {g};

  const Vec2 min_state {Vec2::Constant(-1.0)};
  const Vec2 max_state {Vec2::Constant(+1.0)};
  parameterTestSet(measurement_transform, min_state, max_state, 1e-6, 1e-6);
}

TEST_F(kalman_tests, parameterLTLinearSqrt)
{
  const Mat2 A = Mat2::Random();
  const LinearTransformation<double, C2, C2, NoiseType::additive> g {A};
  const LinearTransform<SquareRootGaussianDistribution, double, C2, C2,
      NoiseType::additive, SquareRootGaussianDistribution<double, C2>> measurement_transform {g};

  const Vec2 min_state {Vec2::Constant(-1.0)};
  const Vec2 max_state {Vec2::Constant(+1.0)};
  parameterTestSet(measurement_transform, min_state, max_state, 1e-6, 1e-6);
}

TEST_F(kalman_tests, parameterTT1Radar)
{
  LinearizedTransform measurement_transform
      {
          LinearizedTransformation<double, C2, C2, NoiseType::additive, 1> {radar_add<double>}
      };

  const Vec2 min_state {0.5, -M_PI_4};
  const Vec2 max_state {1.0, +M_PI_4};
  parameterTestSet(measurement_transform, min_state, max_state, 1e-6, 1e-2);
}

TEST_F(kalman_tests, parameterTT1RadarSqrt)
{
  LinearizedTransform<SquareRootGaussianDistribution, double, C2, C2,
      NoiseType::additive, SquareRootGaussianDistribution<double, C2>> measurement_transform
      {
          LinearizedTransformation<double, C2, C2, NoiseType::additive, 1> {radar_add<double>}
      };

  const Vec2 min_state {0.5, -M_PI_4};
  const Vec2 max_state {1.0, +M_PI_4};
  parameterTestSet(measurement_transform, min_state, max_state, 1e-6, 1e-2);
}

TEST_F(kalman_tests, parameterTT2radar)
{
  LinearizedTransform measurement_transform {radar_add<double>};

  const Vec2 min_state {0.5, -M_PI_4};
  const Vec2 max_state {1.0, +M_PI_4};
  parameterTestSet(measurement_transform, min_state, max_state, 1e-6, 1e-6);
}

TEST_F(kalman_tests, parameterTT2radar_Sqrt)
{
  LinearizedTransform<SquareRootGaussianDistribution, double, C2, C2,
      NoiseType::additive, SquareRootGaussianDistribution<double, C2>> measurement_transform {radar_add<double>};

  const Vec2 min_state {0.5, -M_PI_4};
  const Vec2 max_state {1.0, +M_PI_4};
  parameterTestSet(measurement_transform, min_state, max_state, 1e-6, 1e-6);
}
