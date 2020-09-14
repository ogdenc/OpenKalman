/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "kalman-tests.h"

using M22 = Eigen::Matrix<double, 2, 2>;

TEST_F(kalman_tests, rotation_2D_linear_SA)
{
  rotation_2D<EigenSelfAdjointMatrix<M22>>(LinearTransform());
}

TEST_F(kalman_tests, rotation_2D_linear_T)
{
  rotation_2D<EigenTriangularMatrix<M22>>(LinearTransform());
}

TEST_F(kalman_tests, rotation_2D_linearized_SA)
{
  rotation_2D<EigenSelfAdjointMatrix<M22>>(LinearizedTransform<2>());
}

TEST_F(kalman_tests, rotation_2D_linearized_T)
{
  rotation_2D<EigenTriangularMatrix<M22>>(LinearizedTransform<2>());
}

TEST_F(kalman_tests, rotation_2D_cubature_SA)
{
  rotation_2D<EigenSelfAdjointMatrix<M22>>(CubatureTransform());
}

TEST_F(kalman_tests, rotation_2D_cubature_T)
{
  rotation_2D<EigenTriangularMatrix<M22>>(CubatureTransform());
}

TEST_F(kalman_tests, rotation_2D_MCT_SA)
{
  rotation_2D<EigenSelfAdjointMatrix<M22>>(MonteCarloTransform(1e6));
}
