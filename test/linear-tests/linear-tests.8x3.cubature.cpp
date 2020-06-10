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
#include <transforms/sample-points/CubaturePoints.h>
#include "transforms/classes/SamplePointsTransform.h"

using C8 = Axes<8>;
using C3 = Axes<3>;

TEST_F(linear_tests, Linear8x3Cubature)
{
    const auto [g, A] = linear_function<double, C8, C3, NoiseType::additive>();
    const CubaturePoints<GaussianDistribution, double, C8> s;
    construct_linear_test(s, g, A, 10);
}
