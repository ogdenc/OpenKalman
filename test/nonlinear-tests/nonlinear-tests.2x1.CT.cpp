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

TEST_F(nonlinear_tests, CTSumOfSquares2) {
    constexpr int n = 2;
    CubaturePoints<GaussianDistribution, double, Axes<n>> s {};
    SamplePointsTransform t {s, sum_of_squares<double, n>};
    const Eigen::Matrix<double,n,1> mu = Eigen::Matrix<double,n,1>::Zero();
    const Eigen::Matrix<double,n,n> P = Eigen::Matrix<double,n,n>::Identity();
    doReduction(t, mu, P, (double) n, 0., 1e-6, 1e-6);
}

TEST_F(nonlinear_tests, CTSumOfSquares5) {
    constexpr int n = 5;
    CubaturePoints<GaussianDistribution, double, Axes<n>> s {};
    SamplePointsTransform t {s, sum_of_squares<double, n>};
    const Eigen::Matrix<double,n,1> mu = Eigen::Matrix<double,n,1>::Zero();
    const Eigen::Matrix<double,n,n> P = Eigen::Matrix<double,n,n>::Identity();
    doReduction(t, mu, P, (double) n, 0., 1e-6, 1e-6);
}

TEST_F(nonlinear_tests, CTTOA2) {
    constexpr int n = 2;
    CubaturePoints<GaussianDistribution, double, Axes<n>> s {};
    SamplePointsTransform t {s, time_of_arrival<double, n>};
    Eigen::Matrix<double,n,1> mu = Eigen::Matrix<double,n,1>::Zero();
    mu(0) = 3;
    Eigen::Matrix<double,n,n> P = Eigen::Matrix<double,n,n>::Identity();
    P *= 10;
    P(0,0) = 1;
    doReduction(t, mu, P, 4.19, 2.42, 1e-2, 1e-2);
}

TEST_F(nonlinear_tests, CTTOA3) {
    constexpr int n = 3;
    CubaturePoints<GaussianDistribution, double, Axes<n+1>> s4 {}; // testing dimension change
    CubaturePoints<GaussianDistribution, double, Axes<n>> s {s4.template resize_dimensions<Axes<n>>()};
    SamplePointsTransform t {s, time_of_arrival<double, n>};
    Eigen::Matrix<double,n,1> mu = Eigen::Matrix<double,n,1>::Zero();
    mu(0) = 3;
    Eigen::Matrix<double,n,n> P = Eigen::Matrix<double,n,n>::Identity();
    P *= 10;
    P(0,0) = 1;
    doReduction(t, mu, P, 5.16, 3.34, 1e-2, 1e-2);
}

