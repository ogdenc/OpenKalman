/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, broadcast)
{
  auto m23 = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  M2x m2x_3 {m23};
  Mx3 mx3_2 {m23};
  Mxx mxx_23 {m23};
  std::integral_constant<std::size_t, 1> n1{};
  std::integral_constant<std::size_t, 2> n2{};
  std::integral_constant<std::size_t, 3> n3{};

  EXPECT_TRUE(is_near(broadcast(m23), m23));
  EXPECT_TRUE(is_near(broadcast(m23, n1), m23));
  EXPECT_TRUE(is_near(broadcast(m23, n1, n1), m23));
  EXPECT_TRUE(is_near(broadcast(m23, n1, n1, n1, n1), m23));
  EXPECT_TRUE(is_near(broadcast(m23, n2), m23.template replicate<2,1>()));
  EXPECT_TRUE(is_near(broadcast(m23, n2, n1), m23.template replicate<2,1>()));
  static_assert(index_dimension_of_v<decltype(broadcast(m23, n2, n1)), 0> == 4);
  static_assert(index_dimension_of_v<decltype(broadcast(m23, n2, n1)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(broadcast(m23, n2, n1, n1)), 1> == 3);
  EXPECT_TRUE(is_near(broadcast(m23, n1, n2), m23.template replicate<1,2>()));
  EXPECT_TRUE(is_near(broadcast(m23, n2, n3), m23.template replicate<2,3>()));
  static_assert(index_dimension_of_v<decltype(broadcast(m23, n2, n3)), 0> == 4);
  static_assert(index_dimension_of_v<decltype(broadcast(m23, n2, n3)), 1> == 9);

  EXPECT_TRUE(is_near(broadcast(m23, 1), m23));
  EXPECT_TRUE(is_near(broadcast(m23, 1, n1), m23));
  EXPECT_TRUE(is_near(broadcast(m23, 2), m23.template replicate<2,1>()));
  EXPECT_TRUE(is_near(broadcast(m23, 2, n1), m23.template replicate<2,1>()));
  static_assert(index_dimension_of_v<decltype(broadcast(m23, 2, n1)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(broadcast(m23, 2, n1, n1)), 1> == 3);
  EXPECT_TRUE(is_near(broadcast(m23, 1, n2), m23.template replicate<1,2>()));
  EXPECT_TRUE(is_near(broadcast(m23, 2, 3), m23.template replicate<2,3>()));

  EXPECT_TRUE(is_near(broadcast(m2x_3, n1), m23));
  EXPECT_TRUE(is_near(broadcast(m2x_3, 1, n1), m23));
  EXPECT_TRUE(is_near(broadcast(m2x_3, n2), m23.template replicate<2,1>()));
  EXPECT_TRUE(is_near(broadcast(m2x_3, 2, n1), m23.template replicate<2,1>()));
  EXPECT_TRUE(is_near(broadcast(m2x_3, 1, n2), m23.template replicate<1,2>()));
  EXPECT_TRUE(is_near(broadcast(m2x_3, 2, 3), m23.template replicate<2,3>()));

  EXPECT_TRUE(is_near(broadcast(mx3_2, n1), m23));
  EXPECT_TRUE(is_near(broadcast(mx3_2, 1, n1), m23));
  EXPECT_TRUE(is_near(broadcast(mx3_2, n2), m23.template replicate<2,1>()));
  EXPECT_TRUE(is_near(broadcast(mx3_2, 2, n1), m23.template replicate<2,1>()));
  static_assert(index_dimension_of_v<decltype(broadcast(mx3_2, 2, n1)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(broadcast(mx3_2, 2, n1, n1)), 1> == 3);
  EXPECT_TRUE(is_near(broadcast(mx3_2, 1, n2), m23.template replicate<1,2>()));
  EXPECT_TRUE(is_near(broadcast(mx3_2, 2, 3), m23.template replicate<2,3>()));

  EXPECT_TRUE(is_near(broadcast(mxx_23, n1), m23));
  EXPECT_TRUE(is_near(broadcast(mxx_23, 1, n1), m23));
  EXPECT_TRUE(is_near(broadcast(mxx_23, n2), m23.template replicate<2,1>()));
  EXPECT_TRUE(is_near(broadcast(mxx_23, 2, n1), m23.template replicate<2,1>()));
  EXPECT_TRUE(is_near(broadcast(mxx_23, 1, n2), m23.template replicate<1,2>()));
  EXPECT_TRUE(is_near(broadcast(mxx_23, 2, 3), m23.template replicate<2,3>()));

  EXPECT_ANY_THROW(broadcast(m23, 3, 3));
  EXPECT_ANY_THROW(broadcast(mxx_23, 2, 4));
  EXPECT_ANY_THROW(broadcast(mxx_23, -1, 3));
}
