/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "MonteCarlo-tests.h"

using C2 = Coefficients<Axis, Axis>;

TEST_F(MonteCarlo_tests, Radar2x2) {
    IndependentNoise<GaussianDistribution, double, C2> noise;
    for (int i=0; i<5; i++)
    {
        auto res = sqrtComparison(radar<double>, noise());
        EXPECT_TRUE(res);
    }
}

TEST_F(MonteCarlo_tests, Radar2x2Aug) {
    IndependentNoise<GaussianDistribution, double, C2> noise;
    for (int i=0; i<5; i++)
    {
        auto res = sqrtComparison(radar_aug<double>, noise());
        EXPECT_TRUE(res);
    }
}

TEST_F(MonteCarlo_tests, Radar2x2Add) {
    IndependentNoise<GaussianDistribution, double, C2> noise;
    for (int i=0; i<5; i++)
    {
        auto res = sqrtComparison(radar_add<double>, noise());
        EXPECT_TRUE(res);
    }
}

