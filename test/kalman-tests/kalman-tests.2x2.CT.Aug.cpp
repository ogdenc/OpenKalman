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


using namespace OpenKalman;

using C2 = Coefficients<Axis, Axis>;
using Vec2 = Mean<double, C2>;

static const VectorTransformation<double, C2, C2, NoiseType::augmented> polynomial
{
    [](Eigen::Matrix<double, 4, 1> x) -> Vec2
    {
        return Vec2
        {
            std::pow(x(0), 3) + 5 * x(1) + x(2),
            std::pow(x(1), 3) + 3 * x(0) + x(3)
        };
    }
};

TEST_F(kalman_tests, parameterCTPolyAug) {
    CubaturePoints<GaussianDistribution, double, Concatenate<C2, C2>> s {};
    SamplePointsTransform measurement_transform {s, polynomial};

    const Vec2 min_state {1., 1.};
    const Vec2 max_state {5., 5.};
    parameterTestSet(measurement_transform, min_state, max_state, 1e-6, 1e-2);
}

TEST_F(kalman_tests, parameterCTPolyAugSqrt) {
    CubaturePoints<SquareRootGaussianDistribution, double, Concatenate<C2, C2>> s {};
    SamplePointsTransform measurement_transform {s, polynomial};

    const Vec2 min_state {1., 1.};
    const Vec2 max_state {5., 5.};
    parameterTestSet(measurement_transform, min_state, max_state, 1e-6, 1e-2);
}
