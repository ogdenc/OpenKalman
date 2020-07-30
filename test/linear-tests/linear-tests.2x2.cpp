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

TEST_F(linear_tests, Linear2x2LinearSA)
{
  run_multiple_linear_tests<2, 2>(CovSA2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return LinearTransform {g}; });
}

TEST_F(linear_tests, Linear2x2LinearT)
{
  run_multiple_linear_tests<2, 2>(CovT2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return LinearTransform {g}; });
}

TEST_F(linear_tests, Linear2x2TT1SA)
{
  run_multiple_linear_tests<2, 2>(CovSA2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return LinearizedTransform {g}; });
}

TEST_F(linear_tests, Linear2x2TT1T)
{
  run_multiple_linear_tests<2, 2>(CovT2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return LinearizedTransform {g}; });
}

TEST_F(linear_tests, Linear2x2TT2SA)
{
  run_multiple_linear_tests<2, 2>(CovSA2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return LinearizedTransform<std::decay_t<decltype(g)>, 2> {g}; });
}

TEST_F(linear_tests, Linear2x2TT2T)
{
  run_multiple_linear_tests<2, 2>(CovT2 {1.2, 0.2, 0.2, 2.1},
    [] (const auto& g) { return LinearizedTransform<std::decay_t<decltype(g)>, 2> {g}; });
}
