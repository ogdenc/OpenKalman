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

using C2 = Coefficients<Axis, Axis>;
using M22 = native_matrix_t<double, 2, 2>;
using CovSA2 = Covariance<C2, SelfAdjointMatrix<M22>>;
using CovT2 = Covariance<C2, TriangularMatrix<M22>>;
inline SamplePointsTransform<UnscentedSigmaPoints> UT1;
inline SamplePointsTransform<UnscentedSigmaPointsParameterEstimation> UT2;
inline SamplePointsTransform<SphericalSimplexSigmaPoints> UTS;
inline CubatureTransform CT;

TEST_F(linear_tests, Linear2x3UnscentedSA)
{
  run_multiple_linear_tests<2, 3>(CovSA2 {1.2, 0.2, 0.2, 2.1}, UT1);
}

TEST_F(linear_tests, Linear2x3UnscentedT)
{
  run_multiple_linear_tests<2, 3>(CovT2 {1.2, 0.2, 0.2, 2.1}, UT1);
}

TEST_F(linear_tests, Linear2x3UnscentedParamSA)
{
  run_multiple_linear_tests<2, 3>(CovSA2 {1.2, 0.2, 0.2, 2.1}, UT2);
}

TEST_F(linear_tests, Linear2x3UnscentedParamT)
{
  run_multiple_linear_tests<2, 3>(CovT2 {1.2, 0.2, 0.2, 2.1}, UT2);
}

TEST_F(linear_tests, Linear2x3UnscentedSphericalSA)
{
  run_multiple_linear_tests<2, 3>(CovSA2 {1.2, 0.2, 0.2, 2.1}, UTS);
}

TEST_F(linear_tests, Linear2x3UnscentedSphericalT)
{
  run_multiple_linear_tests<2, 3>(CovT2 {1.2, 0.2, 0.2, 2.1}, UTS);
}

TEST_F(linear_tests, Linear2x3CubatureSA)
{
  run_multiple_linear_tests<2, 3>(CovSA2 {1.2, 0.2, 0.2, 2.1}, CT);
}

TEST_F(linear_tests, Linear2x3CubatureT)
{
  run_multiple_linear_tests<2, 3>(CovT2 {1.2, 0.2, 0.2, 2.1}, CT);
}

