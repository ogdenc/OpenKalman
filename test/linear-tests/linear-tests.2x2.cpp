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

using C2 = Coefficients<Axis, Axis>;
using Scalar = double;

TEST_F(linear_tests, Linear2x2Linear) {
    IndependentNoise<GaussianDistribution, Scalar, C2> dist {10};
    IndependentNoise<GaussianDistribution, Scalar, C2> noise;
    for (int i=1; i<50; i++) {
        const Mean<Scalar, C2, 2> A = Eigen::Matrix<Scalar, 2, 2>::Random() / i;
        const LinearTransformation<Scalar, C2, C2, NoiseType::additive> g {A};
        const LinearTransform t {g};
        run_linear_tests(t, g, dist(), A, noise());
    }
}

TEST_F(linear_tests, Linear2x2TT1) {
    IndependentNoise<GaussianDistribution, Scalar, C2> dist {10};
    IndependentNoise<GaussianDistribution, Scalar, C2> noise;
    for (int i=1; i<50; i++) {
        const Mean<Scalar, C2, 2> A = Eigen::Matrix<Scalar, 2, 2>::Random() / i;
        const LinearizedTransformation<Scalar, C2, C2, NoiseType::additive> g
        {
            [=](const Mean<Scalar, C2>& x) {
                return A * x;
            },
            [=](const Mean<Scalar, C2>& x) {
                return A;
            }
        };
        const LinearizedTransform t {g};
        run_linear_tests(t, g, dist(), A, noise());
    }
}

TEST_F(linear_tests, Linear2x2TT2) {
    IndependentNoise<GaussianDistribution, Scalar, C2> dist {10};
    IndependentNoise<GaussianDistribution, Scalar, C2> noise;
    for (int i=1; i<50; i++) {
        const Mean<Scalar, C2, 2> A = Eigen::Matrix<Scalar, 2, 2>::Random() / i;
        const LinearizedTransformation<Scalar, C2, C2, NoiseType::additive, 2> g
            {
                [=](const Mean<Scalar, C2>& x) {
                    return A * x;
                },
                [=](const Mean<Scalar, C2>& x) {
                    return A;
                },
                [=](const Mean<Scalar, C2>& x) {
                    return std::array<Eigen::Matrix<Scalar, 2, 2>, 2>
                    {Eigen::Matrix<Scalar, 2, 2>::Zero(), Eigen::Matrix<Scalar, 2, 2>::Zero()};
                }
            };
        const LinearizedTransform t {g};
        run_linear_tests(t, g, dist(), A, noise());
    }
}
