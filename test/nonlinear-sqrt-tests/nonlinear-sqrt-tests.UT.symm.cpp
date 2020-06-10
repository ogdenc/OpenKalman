/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "nonlinear-sqrt-tests.ST.h"
#include "transforms/sample-points/SigmaPointsTypes/Unscented.h"

using namespace OpenKalman;

using C2 = Coefficients<Axis, Axis>;

TEST_F(nonlinear_sqrt_tests, UT2SqrtSymmetric2x2) {
    IndependentNoise<GaussianDistribution, double, C2> noise;
    for (int i=0; i<20; i++)
    {
        auto res = sqrtComparison<Unscented, double, C2, C2, NoiseType::none, double, double, double>(radar<double>, noise(), 0.001, 2, 0);
        EXPECT_TRUE(res);
    }
}
