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
#include "transforms/sample-points/CubaturePoints.h"
#include "../transformations.h"

using namespace std;
using namespace OpenKalman;
using namespace Eigen;

using Vec2 = Mean<double, Coefficients<Polar>>;

TEST_F(kalman_tests, parameterCTRadarAugPolar) {
    CubaturePoints<GaussianDistribution, double, Concatenate<Coefficients<Polar>,C2>> s {};
    SamplePointsTransform measurement_transform {s, radarP_aug<double>};

    const Vec2 min_state {0.5, -M_PI/6};
    const Vec2 max_state {1.0, +M_PI/6};
    parameterTestSet(measurement_transform, min_state, max_state, 1e-6, 1e-2);
}
