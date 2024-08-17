/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, Eigen_Solve)
{
  static_assert(index_dimension_of_v<Eigen::Solve<Eigen::PartialPivLU<M14>, M13>, 0> == 4);
  static_assert(index_dimension_of_v<Eigen::Solve<Eigen::PartialPivLU<M14>, M13>, 1> == 3);
}

