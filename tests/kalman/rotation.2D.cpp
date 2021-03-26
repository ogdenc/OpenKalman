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

inline auto get_t2()
{
  auto theta = trace(randomize<Mean<Axis, native_matrix_t<double, 1, 1>>, std::uniform_real_distribution>(-pi, pi));
  using M22 = native_matrix_t<double, 2, 2>;
  auto a = Matrix<Axes<2>, Axes<2>, M22> {std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta)};
  return LinearTransformation(a);
}

template<typename Cov, typename Trans>
void kalman::rotation_2D(const Trans& transform)
{
  using M2 = native_matrix_t<double, 2, 1>;
  using Mean2 = Mean<Axes<2>, M2>;
  for (int i = 0; i < 5; i++)
  {
    auto true_state = randomize<Mean2, std::uniform_real_distribution>(5.0, 10.0);
    auto x = GaussianDistribution<Axes<2>, M2, Cov> {Mean2 {7.5, 7.5}, Cov::identity()};
    auto meas_cov = Cov {0.01, 0, 0, 0.01};
    auto r = GaussianDistribution<Axes<2>, M2, Cov> {Mean2::zero(), meas_cov};
    parameter_test(transform, get_t2(), x, true_state, r, 0.2, 100);
  }
}

using M22 = native_matrix_t<double, 2, 2>;

TEST_F(kalman, rotation_2D_linear_SA)
{
  rotation_2D<SelfAdjointMatrix<M22>>(LinearTransform());
}

TEST_F(kalman, rotation_2D_linear_T)
{
  rotation_2D<TriangularMatrix<M22>>(LinearTransform());
}

TEST_F(kalman, rotation_2D_linearized_SA)
{
  rotation_2D<SelfAdjointMatrix<M22>>(LinearizedTransform<2>());
}

TEST_F(kalman, rotation_2D_linearized_T)
{
  rotation_2D<TriangularMatrix<M22>>(LinearizedTransform<2>());
}

TEST_F(kalman, rotation_2D_cubature_SA)
{
  rotation_2D<SelfAdjointMatrix<M22>>(CubatureTransform());
}

TEST_F(kalman, rotation_2D_cubature_T)
{
  rotation_2D<TriangularMatrix<M22>>(CubatureTransform());
}

TEST_F(kalman, rotation_2D_MCT_SA)
{
  rotation_2D<SelfAdjointMatrix<M22>>(MonteCarloTransform(1e6));
}
