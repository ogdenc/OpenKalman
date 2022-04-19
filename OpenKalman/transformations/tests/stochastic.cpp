/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transformations.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;

using M = Mean<Dimensions<2>>;
using A = Matrix<Dimensions<2>, Dimensions<2>>;

TEST(transformations, stochastic_additive)
{
  auto f = [](const auto& x) -> M { return A {1, 2, 3, 4} * x; };
  auto dist = GaussianDistribution {make_zero_matrix_like<M>(), make_identity_matrix_like<A>()};
  auto t = Transformation<decltype(f)> {f};
  M x {2, 3};
  M true_y {f(x)};
  M mean_y {make_zero_matrix_like<M>()};
  for (int i = 0; i < 100; i++)
  {
    const M y {t(x) + dist()};
    mean_y = (mean_y * i + y) / (i + 1);
  }
  EXPECT_NE(mean_y, true_y);
  EXPECT_TRUE(is_near(mean_y, true_y, nested_matrix_of_t<M>::Constant(1.0)));
}


TEST(transformations, stochastic_augmented)
{
  auto f = [](const auto& x, const auto&...n) { return make_self_contained(((A {1, 2, 4, 3} * x) + ... + (A {3, 4, 2, 1} * n))); };
  auto dist = GaussianDistribution {make_zero_matrix_like<M>(), make_identity_matrix_like<A>()};
  auto t = Transformation<decltype(f)> {f};
  M x {2, 3}, n {0, 0};
  M true_y {f(x, n)};
  M mean_y {make_zero_matrix_like<M>()};
  for (int i = 0; i < 100; i++)
  {
    const M y {t(x, dist)};
    mean_y = (mean_y * i + y) / (i + 1);
  }
  EXPECT_NE(mean_y, true_y);
  EXPECT_TRUE(is_near(mean_y, true_y, nested_matrix_of_t<M>::Constant(1.0)));
}
