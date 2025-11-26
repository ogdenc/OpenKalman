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

#include "collections/tests/tests.hpp"
#include "coordinates/concepts/descriptor.hpp"
#include "coordinates/concepts/pattern.hpp"
#include "coordinates/concepts/euclidean_pattern.hpp"
#include "coordinates/concepts/fixed_pattern.hpp"
#include "coordinates/concepts/dynamic_pattern.hpp"
#include "coordinates/functions/get_dimension.hpp"
#include "coordinates/traits/dimension_of.hpp"
#include "coordinates/functions/get_stat_dimension.hpp"
#include "coordinates/traits/stat_dimension_of.hpp"
#include "coordinates/functions/get_is_euclidean.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;

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

  static_assert(get_dimension(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_stat_dimension(std::integral_constant<std::size_t, 3>{}) == 3);
  static_assert(get_is_euclidean(std::integral_constant<std::size_t, 5>{}));

  static_assert(dimension_of_v<std::integral_constant<std::size_t, 0>> == 0);
  static_assert(dimension_of_v<std::integral_constant<std::size_t, 3>> == 3);
  static_assert(dimension_of_v<std::integral_constant<int, 3>> == 3);
  static_assert(stat_dimension_of_v<std::integral_constant<std::size_t, 3>> == 3);
}


TEST(coordinates, integral)
{
  static_assert(pattern<unsigned>);
  static_assert(not fixed_pattern<unsigned>);
  static_assert(dynamic_pattern<unsigned>);
  static_assert(euclidean_pattern<unsigned>);
  static_assert(descriptor<unsigned>);
  static_assert(euclidean_pattern<unsigned>);
  static_assert(dimension_of_v<unsigned> == stdex::dynamic_extent);
  static_assert(stat_dimension_of_v<unsigned> == stdex::dynamic_extent);
  static_assert(get_dimension(3u) == 3);
  EXPECT_EQ(get_dimension(0u), 0);
  EXPECT_EQ(get_dimension(3u), 3);
  static_assert(get_stat_dimension(3u) == 3);
  EXPECT_EQ(get_stat_dimension(3u), 3);
}
