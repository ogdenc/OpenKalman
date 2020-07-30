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

using C3 = Axes<3>;
using M33 = Eigen::Matrix<double, 3, 3>;
using CovSA3 = Covariance<C3, EigenSelfAdjointMatrix<M33>>;
using CovT3 = Covariance<C3, EigenTriangularMatrix<M33>>;

TEST_F(linear_tests, Linear3x2UnscentedSA)
{
  run_multiple_linear_tests<3, 2>(CovSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1},
    [] (const auto& g) { return make_SamplePointsTransform<UnscentedSigmaPoints>(g); });
}

TEST_F(linear_tests, Linear3x2UnscentedT)
{
  run_multiple_linear_tests<3, 2>(CovT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1},
    [] (const auto& g) { return make_SamplePointsTransform<UnscentedSigmaPoints>(g); });
}

TEST_F(linear_tests, Linear3x2UnscentedParamSA)
{
  run_multiple_linear_tests<3, 2>(CovSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1},
    [] (const auto& g) { return make_SamplePointsTransform<UnscentedSigmaPointsParameterEstimation>(g); });
}

TEST_F(linear_tests, Linear3x2UnscentedParamT)
{
  run_multiple_linear_tests<3, 2>(CovT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1},
    [] (const auto& g) { return make_SamplePointsTransform<UnscentedSigmaPointsParameterEstimation>(g); });
}

TEST_F(linear_tests, Linear3x2UnscentedSphericalSA)
{
  run_multiple_linear_tests<3, 2>(CovSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1},
    [] (const auto& g) { return make_SamplePointsTransform<SphericalSimplexSigmaPoints>(g); });
}

TEST_F(linear_tests, Linear3x2UnscentedSphericalT)
{
  run_multiple_linear_tests<3, 2>(CovT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1},
    [] (const auto& g) { return make_SamplePointsTransform<SphericalSimplexSigmaPoints>(g); });
}

TEST_F(linear_tests, Linear3x2CubatureSA)
{
  run_multiple_linear_tests<3, 2>(CovSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1},
    [] (const auto& g) { return make_SamplePointsTransform<CubaturePoints>(g); });
}

TEST_F(linear_tests, Linear3x2CubatureT)
{
  run_multiple_linear_tests<3, 2>(CovT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1},
    [] (const auto& g) { return make_SamplePointsTransform<CubaturePoints>(g); });
}

