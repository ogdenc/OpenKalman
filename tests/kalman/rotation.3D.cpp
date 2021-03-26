/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "kalman.hpp"

inline auto get_t3()
{
  using M3 = native_matrix_t<double, 3, 1>;
  using Mean3 = Mean<Axes<3>, M3>;
  using M33 = native_matrix_t<double, 3, 3>;
  auto angles = randomize<Mean3, std::uniform_real_distribution>(-pi, pi);
  auto ax = Matrix<Axes<3>, Axes<3>, M33> {
    1, 0, 0,
    0, std::cos(angles[0]), -std::sin(angles[0]),
    0, std::sin(angles[0]), std::cos(angles[0])};
  auto ay = Matrix<Axes<3>, Axes<3>, M33> {
    std::cos(angles[0]), 0, std::sin(angles[0]),
    0, 1, 0,
    -std::sin(angles[0]), 0, std::cos(angles[0])};
  auto az = Matrix<Axes<3>, Axes<3>, M33> {
    std::cos(angles[0]), -std::sin(angles[0]), 0,
    std::sin(angles[0]), std::cos(angles[0]), 0,
    0, 0, 1};
  return LinearTransformation(ax * ay * az);
}

template<typename Cov, typename Trans>
void kalman::rotation_3D(const Trans& transform)
{
  using M3 = native_matrix_t<double, 3, 1>;
  using Mean3 = Mean<Axes<3>, M3>;
  for (int i = 0; i < 5; i++)
  {
    auto true_state = randomize<Mean3, std::uniform_real_distribution>(5.0, 10.0);
    auto x = GaussianDistribution<Axes<3>, M3, Cov> {Mean3 {7.5, 7.5, 7.5}, Cov::identity()};
    auto meas_cov = Cov {0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.1};
    auto r = GaussianDistribution<Axes<3>, M3, Cov> {Mean3::zero(), meas_cov};
    parameter_test(transform, get_t3(), x, true_state, r, 0.1, 100);
  }
}

using M33 = native_matrix_t<double, 3, 3>;

TEST_F(kalman, rotation_3D_linear_SA)
{
  rotation_3D<SelfAdjointMatrix<M33>>(LinearTransform());
}

TEST_F(kalman, rotation_3D_linear_T)
{
  rotation_3D<TriangularMatrix<M33>>(LinearTransform());
}

TEST_F(kalman, rotation_3D_linearized_SA)
{
  rotation_3D<SelfAdjointMatrix<M33>>(LinearizedTransform<2>());
}

TEST_F(kalman, rotation_3D_linearized_T)
{
  rotation_3D<TriangularMatrix<M33>>(LinearizedTransform<2>());
}

TEST_F(kalman, rotation_3D_cubature_SA)
{
  rotation_3D<SelfAdjointMatrix<M33>>(CubatureTransform());
}

TEST_F(kalman, rotation_3D_cubature_T)
{
  rotation_3D<TriangularMatrix<M33>>(CubatureTransform());
}

TEST_F(kalman, rotation_3D_unscented_SA)
{
  rotation_3D<SelfAdjointMatrix<M33>>(SamplePointsTransform<UnscentedSigmaPointsParameterEstimation>());
}

TEST_F(kalman, rotation_3D_unscented_T)
{
  rotation_3D<TriangularMatrix<M33>>(SamplePointsTransform<UnscentedSigmaPointsParameterEstimation>());
}

TEST_F(kalman, rotation_3D_simplex_SA)
{
  rotation_3D<SelfAdjointMatrix<M33>>(SamplePointsTransform<SphericalSimplexSigmaPoints>());
}

TEST_F(kalman, rotation_3D_simplex_T)
{
  rotation_3D<TriangularMatrix<M33>>(SamplePointsTransform<SphericalSimplexSigmaPoints>());
}
