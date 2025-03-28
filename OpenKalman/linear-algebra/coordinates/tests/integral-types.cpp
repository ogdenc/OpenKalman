/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for integral coefficient types
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/dynamic_pattern.hpp"
#include "linear-algebra/coordinates/functions/get_size.hpp"
#include "linear-algebra/coordinates/traits/size_of.hpp"
#include "linear-algebra/coordinates/functions/get_euclidean_size.hpp"
#include "linear-algebra/coordinates/traits/euclidean_size_of.hpp"
#include "linear-algebra/coordinates/functions/get_component_count.hpp"
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp"
#include "linear-algebra/coordinates/traits/component_count_of.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinate;

TEST(coordinates, integral_constant)
{
  static_assert(fixed_pattern<std::integral_constant<std::size_t, 3>>);
  static_assert(fixed_pattern<std::integral_constant<int, 3>>);
  static_assert(pattern<std::integral_constant<int, 3>>);
  static_assert(euclidean_pattern<std::integral_constant<std::size_t, 0>>);
  static_assert(euclidean_pattern<std::integral_constant<std::size_t, 1>>);
  static_assert(euclidean_pattern<std::integral_constant<std::size_t, 2>>);
  static_assert(descriptor<std::integral_constant<std::size_t, 1>>);
  static_assert(descriptor<std::integral_constant<std::size_t, 2>>);

  static_assert(get_size(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_euclidean_size(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_component_count(std::integral_constant<std::size_t, 3>{}) == 1);
  static_assert(get_is_euclidean(std::integral_constant<std::size_t, 5>{}));

  static_assert(size_of_v<std::integral_constant<std::size_t, 0>> == 0);
  static_assert(size_of_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(size_of_v<std::integral_constant<int, 3>> == 3);
  static_assert(euclidean_size_of_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(component_count_of_v<std::integral_constant<std::size_t, 3>> == 1);
}


TEST(coordinates, integral)
{
  static_assert(pattern<unsigned>);
  static_assert(not fixed_pattern<unsigned>);
  static_assert(dynamic_pattern<unsigned>);
  static_assert(euclidean_pattern<unsigned>);
  static_assert(descriptor<unsigned>);
  static_assert(euclidean_pattern<unsigned>);
  static_assert(size_of_v<unsigned> == dynamic_size);
  static_assert(euclidean_size_of_v<unsigned> == dynamic_size);
  static_assert(get_size(3u) == 3);
  EXPECT_EQ(get_size(0u), 0);
  EXPECT_EQ(get_size(3u), 3);
  static_assert(get_euclidean_size(3u) == 3);
  EXPECT_EQ(get_euclidean_size(3u), 3);
}
