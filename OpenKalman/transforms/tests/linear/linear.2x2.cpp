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
  using C2 = Coefficients<Axis, Axis>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using CovSA2 = Covariance <C2, SelfAdjointMatrix<M22>>;
  using CovT2 = Covariance <C2, TriangularMatrix<M22>>;
  inline LinearTransform L;
  inline LinearizedTransform L1;
  inline LinearizedTransform<2> L2;
}

TEST(linear_tests, Linear2x2LinearSA)
{
  run_multiple_linear_tests<2, 2>(CovSA2 {1.2, 0.2, 0.2, 2.1}, L);
}

TEST(linear_tests, Linear2x2LinearT)
{
  run_multiple_linear_tests<2, 2>(CovT2 {1.2, 0.2, 0.2, 2.1}, L);
}

TEST(linear_tests, Linear2x2TT1SA)
{
  run_multiple_linear_tests<2, 2>(CovSA2 {1.2, 0.2, 0.2, 2.1}, L1);
}

TEST(linear_tests, Linear2x2TT1T)
{
  run_multiple_linear_tests<2, 2>(CovT2 {1.2, 0.2, 0.2, 2.1}, L1);
}

TEST(linear_tests, Linear2x2TT2SA)
{
  run_multiple_linear_tests<2, 2>(CovSA2 {1.2, 0.2, 0.2, 2.1}, L2);
}

TEST(linear_tests, Linear2x2TT2T)
{
  run_multiple_linear_tests<2, 2>(CovT2 {1.2, 0.2, 0.2, 2.1}, L2);
}
