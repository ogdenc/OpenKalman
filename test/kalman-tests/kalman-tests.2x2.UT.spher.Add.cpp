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
#include "transforms/classes/SamplePointsTransform.h"
#include "transforms/sample-points/SigmaPointsTypes/SphericalSimplex.h"

using namespace OpenKalman;

using C2 = Coefficients<Axis, Axis>;
using Vec2 = Mean<double, C2>;

TEST_F(kalman_tests, parameterUTSpherRadarAdd) {
    SphericalSimplex<GaussianDistribution, double, C2> s {0.001, 2, 0.5};
    SamplePointsTransform measurement_transform {s, radar_add<double>};

    const Vec2 min_state {0.5, -M_PI_4};
    const Vec2 max_state {1.0, +M_PI_4};
    parameterTestSet(measurement_transform, min_state, max_state, 1e-6, 1e-2);
}

TEST_F(kalman_tests, parameterUTSpherRadarAddSqrt) {
    SphericalSimplex<SquareRootGaussianDistribution, double, C2> s {0.001, 2, 0.5};
    SamplePointsTransform measurement_transform {s, radar_add<double>};

    const Vec2 min_state {0.5, -M_PI_4};
    const Vec2 max_state {1.0, +M_PI_4};
    parameterTestSet(measurement_transform, min_state, max_state, 1e-6, 1e-2);
}
