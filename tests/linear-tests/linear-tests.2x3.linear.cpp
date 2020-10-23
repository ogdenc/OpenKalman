/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear-tests.hpp"

using C2 = Coefficients<Axis, Axis>;
using M22 = Eigen::Matrix<double, 2, 2>;
using CovSA2 = Covariance<C2, EigenSelfAdjointMatrix<M22>>;
using CovT2 = Covariance<C2, EigenTriangularMatrix<M22>>;
inline LinearTransform L;
inline LinearizedTransform L1;
inline LinearizedTransform<2> L2;

TEST_F(linear_tests, Linear2x3LinearSA)
{
  run_multiple_linear_tests<2, 3>(CovSA2 {1.2, 0.2, 0.2, 2.1}, L);
}

TEST_F(linear_tests, Linear2x3LinearT)
{
  run_multiple_linear_tests<2, 3>(CovT2 {1.2, 0.2, 0.2, 2.1}, L);
}

TEST_F(linear_tests, Linear2x3TT1SA)
{
  run_multiple_linear_tests<2, 3>(CovSA2 {1.2, 0.2, 0.2, 2.1}, L1);
}

TEST_F(linear_tests, Linear2x3TT1T)
{
  run_multiple_linear_tests<2, 3>(CovT2 {1.2, 0.2, 0.2, 2.1}, L1);
}

TEST_F(linear_tests, Linear2x3TT2SA)
{
  run_multiple_linear_tests<2, 3>(CovSA2 {1.2, 0.2, 0.2, 2.1}, L2);
}

TEST_F(linear_tests, Linear2x3TT2T)
{
  run_multiple_linear_tests<2, 3>(CovT2 {1.2, 0.2, 0.2, 2.1}, L2);
}
