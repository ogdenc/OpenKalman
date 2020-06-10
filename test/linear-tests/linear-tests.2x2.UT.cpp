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
#include "transforms/sample-points/SigmaPointsTypes/Unscented.h"
#include "transforms/sample-points/SigmaPointsTypes/SphericalSimplex.h"
#include "transforms/classes/SamplePointsTransform.h"

using C2 = Coefficients<Axis, Axis>;
using Scalar = double;

TEST_F(linear_tests, Linear2x2Symmetrickappa0)
{
    const auto [g, A] = linear_function<double, C2, C2, NoiseType::additive>();
    const Unscented<GaussianDistribution, double, C2> s {0.001, 2, 0};
    construct_linear_test(s, g, A, 50);
    const Unscented<SquareRootGaussianDistribution, double, C2> s2 {0.001, 2, 0};
    construct_linear_test(s2, g, A, 50);
}

TEST_F(linear_tests, Linear2x2Symmetrickappa3_n)
{
    const auto [g, A] = linear_function<double, C2, C2, NoiseType::additive>();
    const Unscented<GaussianDistribution, double, C2> s {0.001, 2, 3 - 2};
    construct_linear_test(s, g, A, 50);
    //const SymmetricSigmaPoints<SquareRootGaussianDistribution, double, in_dim> s2 {0.001, 2, 3 - 2};
    //construct_linear_test(s2, g, A, 50);
}

TEST_F(linear_tests, Linear2x2SphericalW03)
{
    const auto [g, A] = linear_function<double, C2, C2, NoiseType::additive>();
    const SphericalSimplex<GaussianDistribution, double, C2> s {0.001, 2, 0.3};
    construct_linear_test(s, g, A, 50);
    //const SphericalSimplex<SquareRootGaussianDistribution, double, in_dim> s2 {0.001, 2, 0.3};
    //construct_linear_test(s2, g, A, 50);
}

TEST_F(linear_tests, Linear2x2SphericalW07)
{
    const auto [g, A] = linear_function<double, C2, C2, NoiseType::additive>();
    const SphericalSimplex<GaussianDistribution, double, C2> s {0.001, 2, 0.7};
    construct_linear_test(s, g, A, 50);
    //const SphericalSimplex<SquareRootGaussianDistribution, double, in_dim> s2 {0.001, 2, 0.7};
    //construct_linear_test(s2, g, A, 50);
}
