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
#include "transforms/classes/SamplePointsTransform.h"

using C4 = Axes<4>;
using C2 = Coefficients<Axis, Axis>;

TEST_F(linear_tests, Linear4x2Symmetrickappa0) {
    const auto [g, A] = linear_function<double, C4, C2, NoiseType::additive>();
    const Unscented<GaussianDistribution, double, C4> s {0.001, 2, 0};
    construct_linear_test(s, g, A, 50);
}

TEST_F(linear_tests, Linear4x2Symmetrickappa3_n) {
    const auto [g, A] = linear_function<double, C4, C2, NoiseType::additive>();
    const Unscented<GaussianDistribution, double, C4> s {0.001, 2, 3 - 4};
    construct_linear_test(s, g, A, 50);
}
