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

using M33 = Eigen::Matrix<double, 3, 3>;

TEST_F(kalman_tests, rotation_3D_linear_SA)
{
  rotation_3D<EigenSelfAdjointMatrix<M33>>(LinearTransform());
}

TEST_F(kalman_tests, rotation_3D_linear_T)
{
  rotation_3D<EigenTriangularMatrix<M33>>(LinearTransform());
}

TEST_F(kalman_tests, rotation_3D_linearized_SA)
{
  rotation_3D<EigenSelfAdjointMatrix<M33>>(LinearizedTransform<2>());
}

TEST_F(kalman_tests, rotation_3D_linearized_T)
{
  rotation_3D<EigenTriangularMatrix<M33>>(LinearizedTransform<2>());
}

TEST_F(kalman_tests, rotation_3D_cubature_SA)
{
  rotation_3D<EigenSelfAdjointMatrix<M33>>(CubatureTransform());
}

TEST_F(kalman_tests, rotation_3D_cubature_T)
{
  rotation_3D<EigenTriangularMatrix<M33>>(CubatureTransform());
}

TEST_F(kalman_tests, rotation_3D_unscented_SA)
{
  rotation_3D<EigenSelfAdjointMatrix<M33>>(SamplePointsTransform<UnscentedSigmaPointsParameterEstimation>());
}

TEST_F(kalman_tests, rotation_3D_unscented_T)
{
  rotation_3D<EigenTriangularMatrix<M33>>(SamplePointsTransform<UnscentedSigmaPointsParameterEstimation>());
}

TEST_F(kalman_tests, rotation_3D_simplex_SA)
{
  rotation_3D<EigenSelfAdjointMatrix<M33>>(SamplePointsTransform<SphericalSimplexSigmaPoints>());
}

TEST_F(kalman_tests, rotation_3D_simplex_T)
{
  rotation_3D<EigenTriangularMatrix<M33>>(SamplePointsTransform<SphericalSimplexSigmaPoints>());
}
