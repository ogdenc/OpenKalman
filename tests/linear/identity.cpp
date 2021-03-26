/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
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
using C3 = Coefficients<Axis, Axis, Axis>;
using M33 = native_matrix_t<double, 3, 3>;
using CovSA3 = Covariance<C3, SelfAdjointMatrix<M33>>;
using CovT3 = Covariance<C3, TriangularMatrix<M33>>;


TEST_F(linear_tests, Linear2x2IdentitySA)
{
  run_multiple_identity_tests<2>(CovSA2 {1.2, 0.2, 0.2, 2.1});
}

TEST_F(linear_tests, Linear2x2IdentityT)
{
  run_multiple_identity_tests<2>(CovT2 {1.2, 0.2, 0.2, 2.1});
}

TEST_F(linear_tests, Linear3x3IdentitySA)
{
  run_multiple_identity_tests<3>(CovSA3 {
  1.2, 0.2, 0.1,
  0.2, 2.1, 0.3,
  0.1, 0.3, 1.8});
}

TEST_F(linear_tests, Linear3x3IdentityT)
{
  run_multiple_identity_tests<3>(CovT3 {
  1.2, 0.2, 0.1,
  0.2, 2.1, 0.3,
  0.1, 0.3, 1.8});
}

