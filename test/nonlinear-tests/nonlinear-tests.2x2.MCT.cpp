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
#include <transforms/classes/MonteCarloTransform.h>
#include <transforms/classes/LinearizedTransform.h>

using namespace std;
using namespace OpenKalman;
using namespace Eigen;

/*
 * Test data from Gustafsson & Hendeby. Some Relations Between Extended and Unscented Kalman Filters.
 * IEEE Transactions on Signal Processing, (60), 2, 545-555. 2012.
 */

using C2 = Coefficients<Axis, Axis>;

TEST_F(nonlinear_tests, MCTRadarA1)
{
    MonteCarloTransform<GaussianDistribution, double, C2, C2, NoiseType::none> t {radar<double>, 100000};
    do2x2Transform(
        t,
        Matrix<double, 2, 1>(3.0, 0.0),
        Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
        {1.8, 1e-1,
         0.0, 1e-1},
        {2.5, 2e-1, 0.0, 1e-1, // need a little more latitude here
         0.0, 1e-1, 4.4, 1e-1}
    );
}

TEST_F(nonlinear_tests, MCTRadarA2)
{
    MonteCarloTransform<GaussianDistribution, double, C2, C2, NoiseType::none> t {radar<double>, 100000};
    do2x2Transform(
        t,
        Matrix<double, 2, 1>(3.0, M_PI / 6),
        Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
        {1.6, 1e-1,
         0.9, 1e-1},
        {2.9, 1e-1, -0.8, 1e-1,
         -0.8, 1e-1, 3.9, 1e-1}
    );
}

TEST_F(nonlinear_tests, MCTRadarA3)
{
    MonteCarloTransform<GaussianDistribution, double, C2, C2, NoiseType::none> t {radar<double>, 100000};
    do2x2Transform(
        t,
        Matrix<double, 2, 1>(3.0, M_PI_4),
        Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,2>::Identity()},
        {1.3, 1e-1,
         1.3, 1e-1},
        {3.4, 1e-1, -1.0, 1e-1,
         -1.0, 1e-1, 3.4, 1e-1}
    );
}


/*
 * Test data from Hendeby & Gustafsson, On Nonlinear Transformations of Stochastic Variables and its Application
 * to Nonlinear Filtering. International Conference on Acoustics, Speech, and Signal Processing (ICASSP). 2007:
 */

TEST_F(nonlinear_tests, MCTRadarB1)
{
    MonteCarloTransform<GaussianDistribution, double, C2, C2, NoiseType::none> t {radar<double>, 100000};
    do2x2Transform(
        t,
        Matrix<double, 2, 1>(20.0, 0.0),
        Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
        {19.0, 1e-1,
         -0.1, 2e-1},
        {2.9, 2e-1, 0.3, 1e0, // need a little more latitude here
         0.3, 1e0, 36.6, 1e0}
    );
}

TEST_F(nonlinear_tests, MCTRadarB2)
{
    MonteCarloTransform<GaussianDistribution, double, C2, C2, NoiseType::none> t {radar<double>, 100000};
    do2x2Transform(
        t,
        Matrix<double, 2, 1>(20.0, M_PI / 6),
        Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
        {16.3, 1e0,
         9.8, 1e0},
        {12.2, 2e0, -15.4, 1e0,
         -15.4, 1e0, 27.9, 1e0}
    );
}

TEST_F(nonlinear_tests, MCTRadarB3)
{
    MonteCarloTransform<GaussianDistribution, double, C2, C2, NoiseType::none> t {radar<double>, 100000};
    do2x2Transform(
        t,
        Matrix<double, 2, 1>(20.0, M_PI_4),
        Eigen::Matrix<double,2,2> {Eigen::Matrix<double,2,1>(1.0, 0.1).asDiagonal()},
        {13.3, 1e0,
         13.6, 1e0},
        {20.3, 1e0, -17.1, 1e0,
         -17.1, 1e0, 20.0, 1e0}
    );
}
