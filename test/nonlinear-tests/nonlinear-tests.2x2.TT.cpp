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
#include "transforms/classes/LinearizedTransform.h"

using namespace OpenKalman;

/*
 * Test data from Gustafsson & Hendeby. Some Relations Between Extended and Unscented Kalman Filters.
 * IEEE Transactions on Signal Processing, (60), 2, 545-555. 2012 (TT2 data appears to be in error, and is omitted).
 */

using C2 = Coefficients<Axis, Axis>;

TEST_F(nonlinear_tests, TT1RadarA1) {
    LinearizedTransform t {LinearizedTransformation<double, C2, C2, NoiseType::none, 1> {radar<double>}};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(3.0, 0.0),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
            {3.0, 1e-1,
             0.0, 1e-1},
            {1.0, 1e-1, 0.0, 1e-1,
             0.0, 1e-1, 9.0, 1e-1}
    );
}

TEST_F(nonlinear_tests, TT1RadarA2) {
    LinearizedTransform t {LinearizedTransformation<double, C2, C2, NoiseType::none, 1> {radar<double>}};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(3.0, M_PI/6),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
            {2.6, 1e-1,
             1.5, 1e-1},
            { 3.0, 1e-1, -3.5, 1e-1,
             -3.5, 1e-1,  7.0, 1e-1}
    );
}

TEST_F(nonlinear_tests, TT1RadarA3) {
    LinearizedTransform t {LinearizedTransformation<double, C2, C2, NoiseType::none, 1> {radar<double>}};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(3.0, M_PI_4),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
            {2.1, 1e-1,
             2.1, 1e-1},
            { 5.0, 1e-1, -4.0, 1e-1,
             -4.0, 1e-1,  5.0, 1e-1}
    );
}


/*
 * Test data from Hendeby & Gustafsson, On Nonlinear Transformations of Stochastic Variables and its Application
 * to Nonlinear Filtering. International Conference on Acoustics, Speech, and Signal Processing (ICASSP). 2007:
 */

TEST_F(nonlinear_tests, TT1RadarB1) {
    LinearizedTransform t {LinearizedTransformation<double, C2, C2, NoiseType::none, 1> {radar<double>}};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(20.0, 0.0),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
            {20.0, 1e-1,
             0.0, 1e-1},
            {1.0, 1e-1,  0.0, 1e-1,
             0.0, 1e-1, 40.0, 1e-1}
    );
}

TEST_F(nonlinear_tests, TT1RadarB2) {
    LinearizedTransform t {LinearizedTransformation<double, C2, C2, NoiseType::none, 1> {radar<double>}};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(20.0, M_PI/6),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
            {17.3, 1e-1,
             10.0, 1e-1},
            { 10.7, 1e-1, -16.9, 1e-1,
             -16.9, 1e-1,  30.3, 1e-1}
    );
}

TEST_F(nonlinear_tests, TT1RadarB3) {
    LinearizedTransform t {LinearizedTransformation<double, C2, C2, NoiseType::none, 1> {radar<double>}};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(20.0, M_PI_4),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
            {14.1, 1e-1,
             14.1, 1e-1},
            { 20.5, 1e-1, -19.5, 1e-1,
             -19.5, 1e-1,  20.5, 1e-1}
    );
}

TEST_F(nonlinear_tests, TT2RadarB1) {
    LinearizedTransform t {radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(20.0, 0.0),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
            {19.0, 1e-1,
              0.0, 1e-1},
            {3.0, 1e-1,  0.0, 1e-1,
             0.0, 1e-1, 40.1, 1e-1}
    );
}

TEST_F(nonlinear_tests, TT2RadarB2) {
    LinearizedTransform t {radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(20.0, M_PI/6),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
            {16.5, 1e-1,
              9.5, 1e-1},
            { 12.3, 1e-1, -16.1, 1e-1,
             -16.1, 1e-1,  30.8, 1e-1}
    );
}

TEST_F(nonlinear_tests, TT2RadarB3) {
    LinearizedTransform t {radar<double>};
    do2x2Transform(
            t,
            Eigen::Matrix<double,2,1>(20.0, M_PI_4),
            Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
            {13.4, 1e-1,
             13.4, 1e-1},
            { 21.5, 1e-1, -18.5, 1e-1,
             -18.5, 1e-1,  21.6, 1e-1}
    );
}
