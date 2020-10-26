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

using C4 = Axes<4>;
using M44 = Eigen::Matrix<double, 4, 4>;
using CovSA4 = Covariance<C4, SelfAdjointMatrix<M44>>;
using CovT4 = Covariance<C4, TriangularMatrix<M44>>;
inline CovSA4 covSA4 = {1.4, 0.1, 0.2, 0.3, 0.1, 2.3, 0.4, 0.5, 0.2, 0.4, 3.2, 0.6, 0.3, 0.5, 0.6, 4.1};
inline CovT4 covT4 = {1.4, 0.1, 0.2, 0.3, 0.1, 2.3, 0.4, 0.5, 0.2, 0.4, 3.2, 0.6, 0.3, 0.5, 0.6, 4.1};
inline SamplePointsTransform<UnscentedSigmaPoints> UT1;
inline SamplePointsTransform<UnscentedSigmaPointsParameterEstimation> UT2;
inline SamplePointsTransform<SphericalSimplexSigmaPoints> UTS;
inline CubatureTransform CT;

TEST_F(linear_tests, Linear4x3UnscentedSA)
{
  run_multiple_linear_tests<4, 3>(covSA4, UT1);
}

TEST_F(linear_tests, Linear4x3UnscentedT)
{
  run_multiple_linear_tests<4, 3>(covT4, UT1);
}

TEST_F(linear_tests, Linear4x3UnscentedParamSA)
{
  run_multiple_linear_tests<4, 3>(covSA4, UT2);
}

TEST_F(linear_tests, Linear4x3UnscentedParamT)
{
  run_multiple_linear_tests<4, 3>(covT4, UT2);
}

TEST_F(linear_tests, Linear4x3UnscentedSphericalSA)
{
  run_multiple_linear_tests<4, 3>(covSA4, UTS);
}

TEST_F(linear_tests, Linear4x3UnscentedSphericalT)
{
  run_multiple_linear_tests<4, 3>(covT4, UTS);
}

TEST_F(linear_tests, Linear4x3CubatureSA)
{
  run_multiple_linear_tests<4, 3>(covSA4, CT);
}

TEST_F(linear_tests, Linear4x3CubatureT)
{
  run_multiple_linear_tests<4, 3>(covT4, CT);
}

