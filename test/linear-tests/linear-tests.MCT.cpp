/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
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
inline CovSA2 covSA2 {1.2, 0.2, 0.2, 2.1};
inline CovT2 covT2 {1.2, 0.2, 0.2, 2.1};

using C3 = Axes<3>;
using M33 = Eigen::Matrix<double, 3, 3>;
using CovSA3 = Covariance<C3, EigenSelfAdjointMatrix<M33>>;
using CovT3 = Covariance<C3, EigenTriangularMatrix<M33>>;
inline CovSA3 covSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1};
inline CovT3 covT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1};

using C4 = Axes<4>;
using M44 = Eigen::Matrix<double, 4, 4>;
using CovSA4 = Covariance<C4, EigenSelfAdjointMatrix<M44>>;
using CovT4 = Covariance<C4, EigenTriangularMatrix<M44>>;
inline CovSA4 covSA4 = {1.4, 0.1, 0.2, 0.3, 0.1, 2.3, 0.4, 0.5, 0.2, 0.4, 3.2, 0.6, 0.3, 0.5, 0.6, 4.1};
inline CovT4 covT4 = {1.4, 0.1, 0.2, 0.3, 0.1, 2.3, 0.4, 0.5, 0.2, 0.4, 3.2, 0.6, 0.3, 0.5, 0.6, 4.1};

inline int N = 1e6;
inline int M = 3;
inline double err = 5e-2;

TEST_F(linear_tests, Linear2x2MonteCarloSA)
{
  run_multiple_linear_tests<2, 2>(covSA2, [] (const auto& g) { return make_MonteCarloTransform(g, N); }, err * 4, M);
}

TEST_F(linear_tests, Linear2x2MonteCarloT)
{
  run_multiple_linear_tests<2, 2>(covT2, [] (const auto& g) { return make_MonteCarloTransform(g, N); }, err * 4, M);
}

TEST_F(linear_tests, Linear3x2MonteCarloSA)
{
  run_multiple_linear_tests<3, 2>(covSA3, [] (const auto& g) { return make_MonteCarloTransform(g, N); }, err * 6, M);
}

TEST_F(linear_tests, Linear3x2MonteCarloT)
{
  run_multiple_linear_tests<3, 2>(covT3, [] (const auto& g) { return make_MonteCarloTransform(g, N); }, err * 6, M);
}

TEST_F(linear_tests, Linear4x3MonteCarloSA)
{
  run_multiple_linear_tests<4, 3>(covSA4, [] (const auto& g) { return make_MonteCarloTransform(g, N); }, err * 12, M);
}

TEST_F(linear_tests, Linear4x3MonteCarloT)
{
  run_multiple_linear_tests<4, 3>(covT4, [] (const auto& g) { return make_MonteCarloTransform(g, N); }, err * 12, M);
}
