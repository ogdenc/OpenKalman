/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "nonlinear-tests.h"
#include <transforms/classes/SamplePointsTransform.h>
#include <transforms/sample-points/CubaturePoints.h>

using namespace OpenKalman;

/*
 * Test data from Gustafsson & Hendeby. Some Relations Between Extended and Unscented Kalman Filters.
 * IEEE Transactions on Signal Processing, (60), 2, 545-555. 2012.
 */

using C2 = Coefficients<Axis, Axis>;

TEST_F(nonlinear_tests, CTRadar1) {
    CubaturePoints<GaussianDistribution, double, C2> s {};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(3.0, 0.0),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
            {1.73, 1e-2,
             0.0, 1e-1},
            {2.6, 1e-1, 0.0,  1e-1,
             0.0, 1e-1, 4.39, 1e-2});
}

TEST_F(nonlinear_tests, CTRadar2) {
    CubaturePoints<GaussianDistribution, double, C2> s {};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(3.0, M_PI/6),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
            {1.52, 2e-2,
             0.83, 4e-2},
            { 3.01, 4e-2, -0.75, 3e-2,
             -0.75, 3e-2,  3.98, 4e-2});
}

TEST_F(nonlinear_tests, CTRadar3) {
    CubaturePoints<GaussianDistribution, double, C2> s {};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(3.0, M_PI_4),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
            {1.21, 2e-2,
             1.24, 2e-2},
            { 3.52,  3e-2, -0.893, 1e-3,
             -0.893, 1e-3,  3.47,  3e-2});
}

