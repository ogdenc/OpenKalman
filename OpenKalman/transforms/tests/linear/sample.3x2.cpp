/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;

inline namespace
{
  using C3 = Dimensions<3>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using CovSA3 = Covariance <C3, HermitianAdapter<M33>>;
  using CovT3 = Covariance <C3, TriangularAdapter<M33>>;
  inline SamplePointsTransform <UnscentedSigmaPoints> UT1;
  inline SamplePointsTransform <UnscentedSigmaPointsParameterEstimation> UT2;
  inline SamplePointsTransform <SphericalSimplexSigmaPoints> UTS;
  inline CubatureTransform CT;
}

TEST(linear_tests, Linear3x2UnscentedSA)
{
  run_multiple_linear_tests<3, 2>(CovSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, UT1);
}

TEST(linear_tests, Linear3x2UnscentedT)
{
  run_multiple_linear_tests<3, 2>(CovT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, UT1);
}

TEST(linear_tests, Linear3x2UnscentedParamSA)
{
  run_multiple_linear_tests<3, 2>(CovSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, UT2);
}

TEST(linear_tests, Linear3x2UnscentedParamT)
{
  run_multiple_linear_tests<3, 2>(CovT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, UT2);
}

TEST(linear_tests, Linear3x2UnscentedSphericalSA)
{
  run_multiple_linear_tests<3, 2>(CovSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, UTS);
}

TEST(linear_tests, Linear3x2UnscentedSphericalT)
{
  run_multiple_linear_tests<3, 2>(CovT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, UTS);
}

TEST(linear_tests, Linear3x2CubatureSA)
{
  run_multiple_linear_tests<3, 2>(CovSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, CT);
}

TEST(linear_tests, Linear3x2CubatureT)
{
  run_multiple_linear_tests<3, 2>(CovT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1}, CT);
}

