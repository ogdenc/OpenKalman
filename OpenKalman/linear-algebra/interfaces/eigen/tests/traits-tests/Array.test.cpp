/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "linear-algebra/interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, Eigen_Array)
{
  static_assert(eigen_array_general<Eigen::Array<double, 3, 2, true>>);
  static_assert(not eigen_matrix_general<Eigen::Array<double, 3, 2, true>>);
  static_assert(index_dimension_of_v<Eigen::Array<double, 3, 2>, 0> == 3);
  static_assert(index_dimension_of_v<Eigen::Array<double, 3, 2>, 1> == 2);
  static_assert(not square_shaped<Eigen::Array<double, 2, 1>>);
}

