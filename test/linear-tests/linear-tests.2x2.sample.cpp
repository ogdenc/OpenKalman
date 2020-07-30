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
using M22 = Eigen::Matrix<double, 2, 2>;
using CovSA2 = Covariance<C2, EigenSelfAdjointMatrix<M22>>;
using CovT2 = Covariance<C2, EigenTriangularMatrix<M22>>;

TEST_F(linear_tests, Linear2x2UnscentedSA)
{
  run_multiple_linear_tests<2, 2>(CovSA2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return make_SamplePointsTransform<UnscentedSigmaPoints>(g); });
}

TEST_F(linear_tests, Linear2x2UnscentedT)
{
  run_multiple_linear_tests<2, 2>(CovT2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return make_SamplePointsTransform<UnscentedSigmaPoints>(g); });
}

TEST_F(linear_tests, Linear2x2UnscentedParamSA)
{
  run_multiple_linear_tests<2, 2>(CovSA2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return make_SamplePointsTransform<UnscentedSigmaPointsParameterEstimation>(g); });
}

TEST_F(linear_tests, Linear2x2UnscentedParamT)
{
  run_multiple_linear_tests<2, 2>(CovT2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return make_SamplePointsTransform<UnscentedSigmaPointsParameterEstimation>(g); });
}

TEST_F(linear_tests, Linear2x2UnscentedSphericalSA)
{
  run_multiple_linear_tests<2, 2>(CovSA2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return make_SamplePointsTransform<SphericalSimplexSigmaPoints>(g); });
}

TEST_F(linear_tests, Linear2x2UnscentedSphericalT)
{
  run_multiple_linear_tests<2, 2>(CovT2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return make_SamplePointsTransform<SphericalSimplexSigmaPoints>(g); });
}

TEST_F(linear_tests, Linear2x2CubatureSA)
{
  run_multiple_linear_tests<2, 2>(CovSA2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return make_SamplePointsTransform<CubaturePoints>(g); });
}

TEST_F(linear_tests, Linear2x2CubatureT)
{
  run_multiple_linear_tests<2, 2>(CovT2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return make_SamplePointsTransform<CubaturePoints>(g); });
}

