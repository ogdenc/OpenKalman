/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, Eigen_VectorWiseOp)
{
  static_assert(index_count_v<decltype(std::declval<M34>().colwise())> == 2);
  static_assert(std::is_same_v<scalar_type_of_t<decltype(std::declval<M34>().colwise())>, double>);

  static_assert(index_dimension_of_v<decltype(std::declval<M34>().rowwise()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(std::declval<M3x>().rowwise()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(std::declval<M34>().colwise()), 1> == 4);
  static_assert(index_dimension_of_v<decltype(std::declval<Mx4>().colwise()), 1> == 4);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().rowwise())> == 2);

  static_assert(zero<decltype(std::declval<Z22>().colwise())>);
  static_assert(zero<decltype(std::declval<Z22>().rowwise())>);
}
