/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <cmath>
#include <tuple>
#include "kalman-tests.h"
#include "transforms/classes/LinearTransform.h"
#include "transforms/classes/LinearizedTransform.h"

using namespace OpenKalman;

using C2 = Coefficients<Axis, Axis>;
using Vec2 = Mean<double, C2>;
using Mat2 = Mean<double, C2, 2>;

TEST_F(kalman_tests, parameterLTUpdateGaussian)
{
  for (int i=0; i<5; i++)
  {
    const Mat2 A = Mat2::Random();
    const LinearTransformation<double, C2, C2, NoiseType::none> g_state {Mat2::Identity()};
    const LinearTransformation<double, C2, C2, NoiseType::additive> g_meas {A};
    const LinearTransform state_transform {g_state};
    const LinearTransform measurement_transform {g_meas};

    KalmanFilter filter(state_transform, measurement_transform);

    auto state0 = GaussianDistribution {Vec2::Random(), Mat2 {0.25, 0, 0, 0.25}};
    auto measurement_noise = GaussianDistribution {Vec2::Random()/100, Mat2 {0.01, 0, 0, 0.01}};
    const auto[y, P_xy] = measurement_transform(state0, measurement_noise);
    const auto P_yy = covariance(y);
    const Mat2 K = P_xy * P_yy.base_matrix().inverse();

    const Mean<double, C2> z = {Vec2::Random()};
    auto state1 = filter.update(state0, z, measurement_noise);

    EXPECT_TRUE(is_near(mean(state1) - mean(state0), K * (z - mean(y))));
    EXPECT_TRUE(is_near(covariance(state0) - covariance(state1), K * P_yy * K.adjoint()));
  }
}

TEST_F(kalman_tests, parameterLTUpdateSQGaussian)
{
  for (int i=0; i<5; i++)
  {
    const Mat2 A = Mat2::Random();
    const LinearTransformation<double, C2, C2, NoiseType::none> g_state {Mat2::Identity()};
    const LinearTransformation<double, C2, C2, NoiseType::additive> g_meas {A};
    const LinearTransform<SquareRootGaussianDistribution, double, C2, C2, NoiseType::none> state_transform {g_state};
    const LinearTransform<SquareRootGaussianDistribution, double, C2, C2, NoiseType::additive, SquareRootGaussianDistribution<double, C2>> measurement_transform {
        g_meas};

    KalmanFilter filter(state_transform, measurement_transform);

    auto state0 = SquareRootGaussianDistribution {Vec2::Random(), Mat2 {0.5, 0, 0, 0.5}};
    auto measurement_noise = SquareRootGaussianDistribution {Vec2::Random()/100, Mat2 {0.1, 0, 0, 0.1}};
    const auto[y, P_xy] = measurement_transform(state0, measurement_noise);
    const auto P_yy = covariance(y);
    const Mat2 K = P_xy * P_yy.base_matrix().inverse();

    const Mean<double, C2> z = {Vec2::Random()};
    auto state1 = filter.update(state0, z, measurement_noise);

    EXPECT_TRUE(is_near(mean(state1) - mean(state0), K * (z - mean(y))));
    EXPECT_TRUE(is_near(covariance(state0) - covariance(state1), K * P_yy * K.adjoint()));
  }
}
