/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear.hpp"

using C2 = Coefficients<Axis, Axis>;
using M22 = Eigen::Matrix<double, 2, 2>;
using CovSA2 = Covariance<C2, SelfAdjointMatrix<M22>>;
using CovT2 = Covariance<C2, TriangularMatrix<M22>>;
inline CovSA2 covSA2 {1.2, 0.2, 0.2, 2.1};
inline CovT2 covT2 {1.2, 0.2, 0.2, 2.1};

using C3 = Axes<3>;
using M33 = Eigen::Matrix<double, 3, 3>;
using CovSA3 = Covariance<C3, SelfAdjointMatrix<M33>>;
using CovT3 = Covariance<C3, TriangularMatrix<M33>>;
inline CovSA3 covSA3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1};
inline CovT3 covT3 {1.2, 0.2, 0.1, 0.2, 2.1, 0.3, 0.1, 0.3, 3.1};

using C4 = Axes<4>;
using M44 = Eigen::Matrix<double, 4, 4>;
using CovSA4 = Covariance<C4, SelfAdjointMatrix<M44>>;
using CovT4 = Covariance<C4, TriangularMatrix<M44>>;
inline CovSA4 covSA4 = {1.4, 0.1, 0.2, 0.3, 0.1, 2.3, 0.4, 0.5, 0.2, 0.4, 3.2, 0.6, 0.3, 0.5, 0.6, 4.1};
inline CovT4 covT4 = {1.4, 0.1, 0.2, 0.3, 0.1, 2.3, 0.4, 0.5, 0.2, 0.4, 3.2, 0.6, 0.3, 0.5, 0.6, 4.1};

inline int M = 3;
inline double err = 5e-2;
inline MonteCarloTransform MCT(1e6);

TEST_F(linear_tests, Linear2x2MonteCarloSA)
{
  run_multiple_linear_tests<2, 2>(covSA2, MCT, err * 4, M);
}

TEST_F(linear_tests, Linear2x2MonteCarloT)
{
  run_multiple_linear_tests<2, 2>(covT2, MCT, err * 4, M);
}

TEST_F(linear_tests, Linear2x3MonteCarloSA)
{
  run_multiple_linear_tests<2, 3>(covSA2, MCT, err * 4, M);
}

TEST_F(linear_tests, Linear2x3MonteCarloT)
{
  run_multiple_linear_tests<2, 3>(covT2, MCT, err * 4, M);
}

TEST_F(linear_tests, Linear3x2MonteCarloSA)
{
  run_multiple_linear_tests<3, 2>(covSA3, MCT, err * 6, M);
}

TEST_F(linear_tests, Linear3x2MonteCarloT)
{
  run_multiple_linear_tests<3, 2>(covT3, MCT, err * 6, M);
}

TEST_F(linear_tests, Linear4x3MonteCarloSA)
{
  run_multiple_linear_tests<4, 3>(covSA4, MCT, err * 12, M);
}

TEST_F(linear_tests, Linear4x3MonteCarloT)
{
  run_multiple_linear_tests<4, 3>(covT4, MCT, err * 12, M);
}

