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
  using C2 = FixedDescriptor<Axis, Axis>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using CovSA2 = Covariance <C2, SelfAdjointMatrix<M22>>;
  using CovT2 = Covariance <C2, TriangularMatrix<M22>>;
  inline LinearTransform L;
  inline LinearizedTransform L1;
  inline LinearizedTransform<2> L2;
}

TEST(linear_tests, Linear2x3LinearSA)
{
  run_multiple_linear_tests<2, 3>(CovSA2 {1.2, 0.2, 0.2, 2.1}, L);
}

TEST(linear_tests, Linear2x3LinearT)
{
  run_multiple_linear_tests<2, 3>(CovT2 {1.2, 0.2, 0.2, 2.1}, L);
}

TEST(linear_tests, Linear2x3TT1SA)
{
  run_multiple_linear_tests<2, 3>(CovSA2 {1.2, 0.2, 0.2, 2.1}, L1);
}

TEST(linear_tests, Linear2x3TT1T)
{
  run_multiple_linear_tests<2, 3>(CovT2 {1.2, 0.2, 0.2, 2.1}, L1);
}

TEST(linear_tests, Linear2x3TT2SA)
{
  run_multiple_linear_tests<2, 3>(CovSA2 {1.2, 0.2, 0.2, 2.1}, L2);
}

TEST(linear_tests, Linear2x3TT2T)
{
  run_multiple_linear_tests<2, 3>(CovT2 {1.2, 0.2, 0.2, 2.1}, L2);
}
