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
 * \brief Tests for coordinates::Distance
 */

#include "collections/tests/tests.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/functions/get_stat_dimension.hpp"
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"
#include "linear-algebra/coordinates/traits/stat_dimension_of.hpp"
#include "linear-algebra/coordinates/descriptors/Distance.hpp"

using namespace OpenKalman::coordinates;

TEST(coordinates, Distance)
{
  static_assert(descriptor<Distance>);
  static_assert(fixed_pattern<Distance>);
  static_assert(pattern<Distance>);
  static_assert(not euclidean_pattern<Distance>);

  static_assert(get_dimension(Distance{}) == 1);
  static_assert(get_stat_dimension(Distance{}) == 1);
  static_assert(not get_is_euclidean(Distance{}));
  static_assert(dimension_of_v<Distance> == 1);
  static_assert(stat_dimension_of_v<Distance> == 1);

  static_assert(std::is_same_v<std::common_type_t<Distance, Distance>, Distance>);
}

#include "linear-algebra/coordinates/functions/to_stat_space.hpp"
#include "linear-algebra/coordinates/functions/from_stat_space.hpp"
#include "linear-algebra/coordinates/functions/wrap.hpp"

TEST(coordinates, Distance_transformations)
{
  EXPECT_NEAR(to_stat_space(Distance{}, std::vector{3.})[0U], 3., 1e-6);
  EXPECT_NEAR(to_stat_space(Distance{}, std::array{-3.})[0U], 3., 1e-6);
  EXPECT_NEAR((to_stat_space(Distance{}, std::array{-3.})[std::integral_constant<std::size_t, 0>{}]), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(Distance{}, std::tuple{-3.})[std::integral_constant<std::size_t, 0>{}]), 3., 1e-6);
  EXPECT_NEAR(to_stat_space(Distance{}, std::tuple{-3.})[0U], 3., 1e-6);

  EXPECT_NEAR(from_stat_space(Distance{}, std::vector{3.})[0U], 3., 1e-6);
  EXPECT_NEAR((from_stat_space(Distance{}, std::array{3.})[std::integral_constant<std::size_t, 0>{}]), 3., 1e-6);
  EXPECT_NEAR(from_stat_space(Distance{}, std::tuple{3.})[0U], 3., 1e-6);

  EXPECT_NEAR(wrap(Distance{}, std::vector{3.})[0U], 3., 1e-6);
  EXPECT_NEAR(wrap(Distance{}, std::array{-3.})[0U], 3., 1e-6);
  EXPECT_NEAR((wrap(Distance{}, std::array{-3.})[std::integral_constant<std::size_t, 0>{}]), 3., 1e-6);
  EXPECT_NEAR((wrap(Distance{}, std::tuple{-3.})[std::integral_constant<std::size_t, 0>{}]), 3., 1e-6);
  EXPECT_NEAR(wrap(Distance{}, std::tuple{-3.})[0U], 3., 1e-6);
}
