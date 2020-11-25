/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear.hpp"

using C3 = Axes<3>;
using M33 = native_matrix_t<double, 3, 3>;
using CovSA3 = Covariance<C3, SelfAdjointMatrix<M33>>;
using CovT3 = Covariance<C3, TriangularMatrix<M33>>;
inline SamplePointsTransform<UnscentedSigmaPoints> UT1;
inline SamplePointsTransform<UnscentedSigmaPointsParameterEstimation> UT2;
inline SamplePointsTransform<SphericalSimplexSigmaPoints> UTS;
inline CubatureTransform CT;

TEST_F(linear_tests, Linear3x2UnscentedSA)
{
  run_multiple_linear_tests<3, 2>(CovSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, UT1);
}

TEST_F(linear_tests, Linear3x2UnscentedT)
{
  run_multiple_linear_tests<3, 2>(CovT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, UT1);
}

TEST_F(linear_tests, Linear3x2UnscentedParamSA)
{
  run_multiple_linear_tests<3, 2>(CovSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, UT2);
}

TEST_F(linear_tests, Linear3x2UnscentedParamT)
{
  run_multiple_linear_tests<3, 2>(CovT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, UT2);
}

TEST_F(linear_tests, Linear3x2UnscentedSphericalSA)
{
  run_multiple_linear_tests<3, 2>(CovSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, UTS);
}

TEST_F(linear_tests, Linear3x2UnscentedSphericalT)
{
  run_multiple_linear_tests<3, 2>(CovT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, UTS);
}

TEST_F(linear_tests, Linear3x2CubatureSA)
{
  run_multiple_linear_tests<3, 2>(CovSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, CT);
}

TEST_F(linear_tests, Linear3x2CubatureT)
{
  run_multiple_linear_tests<3, 2>(CovT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, CT);
}

