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
#include <transforms/sample-points/SigmaPointsTypes/Unscented.h>

using namespace OpenKalman;

/*
 * Test data from Gustafsson & Hendeby. Some Relations Between Extended and Unscented Kalman Filters.
 * IEEE Transactions on Signal Processing, (60), 2, 545-555. 2012.
 */

using C2 = Coefficients<Axis, Axis>;

TEST_F(nonlinear_tests, UT1Radar1A) {
    Unscented<GaussianDistribution, double, C2> s {1, 0, 3-2};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(3.0, 0.0),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
            {1.8, 1e-1,
             0.0, 1e-1},
            {3.7, 1e-1, 0.0, 1e-1,
             0.0, 1e-1, 2.9, 1e-1}
    );
}

TEST_F(nonlinear_tests, UT1Radar2A) {
    Unscented<GaussianDistribution, double, Axes<3>> s3 {1, 0, 3-2};
    Unscented<GaussianDistribution, double, C2> s {s3.template resize_dimensions<C2>()};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(3.0, M_PI/6),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
            {1.6, 1e-1,
             0.9, 1e-1},
            {3.5, 1e-1, 0.3, 1e-1,
             0.3, 1e-1, 3.1, 1e-1}
    );
}

TEST_F(nonlinear_tests, UT1Radar3A) {
    Unscented<GaussianDistribution, double, C2> s {1, 0, 3-2};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(3.0, M_PI_4),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
            {1.3, 1e-1,
             1.3, 1e-1},
            {3.3, 1e-1, 0.4, 1e-1,
             0.4, 1e-1, 3.3, 1e-1}
    );
}

TEST_F(nonlinear_tests, UT2Radar1A) {
    Unscented<GaussianDistribution, double, C2> s {1e-3, 2, 0};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(3.0, 0.0),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
            {1.5, 1e-1,
             0.0, 1e-1},
            {5.5, 1e-1, 0.0, 1e-1,
             0.0, 1e-1, 9.0, 1e-1}
    );
}

TEST_F(nonlinear_tests, UT2Radar2A) {
    Unscented<GaussianDistribution, double, Axes<3>> s3 {1e-3, 2, 0};
    Unscented<GaussianDistribution, double, C2> s {s3.template resize_dimensions<C2>()};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(3.0, M_PI/6),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
            {1.3, 1e-1,
             0.8, 1e-1},
            { 6.4, 1e-1, -1.5, 1e-1,
             -1.5, 1e-1,  8.1, 1e-1}
    );
}

TEST_F(nonlinear_tests, UT2Radar3A) {
    Unscented<GaussianDistribution, double, C2> s {1e-3, 2, 0};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(3.0, M_PI_4),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
            {1.1, 1e-1,
             1.1, 1e-1},
            { 7.2, 1e-1, -1.7, 1e-1,
             -1.7, 1e-1,  7.2, 1e-1}
    );
}


/*
 * Test data from Hendeby & Gustafsson, On Nonlinear Transformations of Stochastic Variables and its Application
 * to Nonlinear Filtering. International Conference on Acoustics, Speech, and Signal Processing (ICASSP). 2007:
 */

TEST_F(nonlinear_tests, UT1Radar2B) {
    Unscented<GaussianDistribution, double, C2> s {1, 0, 3-2};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(20.0, M_PI/6),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
            {16.5, 1e-1,
              9.5, 1e-1},
            { 11.2, 1e-1, -14.4, 1e-1,
             -14.4, 1e-1,  27.8, 1e-1}
    );
}

TEST_F(nonlinear_tests, UT1Radar3B) {
    Unscented<GaussianDistribution, double, C2> s {1, 0, 3-2};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(20.0, M_PI_4),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
            {13.5, 1e-1,
             13.5, 1e-1},
            { 19.5, 1e-1, -16.6, 1e-1,
             -16.6, 1e-1,  19.5, 1e-1}
    );
}

TEST_F(nonlinear_tests, UT2Radar1B) {
    Unscented<GaussianDistribution, double, C2> s {1e-3, 2, 0};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(20.0, 0.0),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
            {19.0, 1e-1,
              0.0, 1e-1},
            {3.0, 1e-1,  0.0, 1e-1,
             0.0, 1e-1, 40.1, 2e-1}
    );
}

TEST_F(nonlinear_tests, UT2Radar2B) {
    Unscented<GaussianDistribution, double, C2> s {1e-3, 2, 0};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(20.0, M_PI/6),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
            {16.5, 1e-1,
              9.5, 1e-1},
            { 12.3, 1e-1, -16.0, 1e-1,
             -16.0, 1e-1,  30.7, 1e-1}
    );
}

TEST_F(nonlinear_tests, UT2Radar3B) {
    Unscented<GaussianDistribution, double, C2> s {1e-3, 2, 0};
    SamplePointsTransform t {s, radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(20.0, M_PI_4),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
            {13.4, 1e-1,
             13.4, 1e-1},
            { 21.5, 1e-1, -18.5, 1e-1,
             -18.5, 1e-1,  21.5, 1e-1}
    );
}

