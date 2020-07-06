/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transformation-tests.h"

using namespace OpenKalman;

using M = Mean<Axes<2>>;
using A = TypedMatrix<Axes<2>, Axes<2>>;

TEST_F(transformation_tests, stochastic_additive)
{
  auto f = [](const auto& x) -> M { return A {1, 2, 3, 4} * x; };
  auto dist = GaussianDistribution {M::zero(), base_matrix(A::identity())};
  auto t = Transformation<Axes<2>, Axes<2>, decltype(f)> {f};
  M x {2, 3};
  M true_y {f(x)};
  M mean_y {M::zero()};
  for (int i = 0; i < 100; i++)
  {
    const M y {t(x) + dist()};
    mean_y = (mean_y * i + y) / (i + 1);
  }
  EXPECT_NE(mean_y, true_y);
  EXPECT_TRUE(is_near(mean_y, true_y, MatrixTraits<M>::BaseMatrix::Constant(0.5)));
}


TEST_F(transformation_tests, stochastic_augmented)
{
  auto f = [](const auto& x, const auto&...n) { return strict(((A {1, 2, 4, 3} * x) + ... + (A {3, 4, 2, 1} * n))); };
  auto dist = GaussianDistribution {M::zero(), base_matrix(A::identity())};
  auto t = Transformation<Axes<2>, Axes<2>, decltype(f)> {f};
  M x {2, 3}, n {0, 0};
  M true_y {f(x, n)};
  M mean_y {M::zero()};
  for (int i = 0; i < 100; i++)
  {
    const M y {t(x, dist)};
    mean_y = (mean_y * i + y) / (i + 1);
  }
  EXPECT_NE(mean_y, true_y);
  EXPECT_TRUE(is_near(mean_y, true_y, MatrixTraits<M>::BaseMatrix::Constant(0.5)));
}