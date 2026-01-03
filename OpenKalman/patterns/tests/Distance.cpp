/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for patterns::Distance
 */

#include "collections/tests/tests.hpp"
#include "patterns/concepts/fixed_pattern.hpp"
#include "patterns/concepts/pattern.hpp"
#include "patterns/concepts/euclidean_pattern.hpp"
#include "patterns/concepts/descriptor.hpp"
#include "patterns/functions/get_stat_dimension.hpp"
#include "patterns/functions/get_is_euclidean.hpp"
#include "patterns/traits/dimension_of.hpp"
#include "patterns/traits/stat_dimension_of.hpp"
#include "patterns/descriptors/Distance.hpp"

using namespace OpenKalman::patterns;

TEST(patterns, Distance)
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

#include "patterns/functions/to_stat_space.hpp"
#include "patterns/functions/from_stat_space.hpp"
#include "patterns/functions/wrap.hpp"

TEST(patterns, Distance_transformations)
{
  EXPECT_NEAR(to_stat_space(Distance{}, std::vector{3.})[0U], 3., 1e-6);
  EXPECT_NEAR(to_stat_space(Distance{}, std::array{-3.})[0U], 3., 1e-6);
  EXPECT_NEAR((to_stat_space(Distance{}, std::array{-3.})[std::integral_constant<std::size_t, 0>{}]), 3., 1e-6);
  EXPECT_NEAR((to_stat_space(Distance{}, std::tuple{-3.})[std::integral_constant<std::size_t, 0>{}]), 3., 1e-6);
  EXPECT_NEAR(to_stat_space(Distance{}, std::tuple{-3.})[0U], 3., 1e-6);

  EXPECT_NEAR(from_stat_space(Distance{}, std::vector{3.})[0U], 3., 1e-6);
  EXPECT_NEAR((from_stat_space(Distance{}, std::array{3.})[std::integral_constant<std::size_t, 0>{}]), 3., 1e-6);
  EXPECT_NEAR(std::get<0>(from_stat_space(Distance{}, std::tuple{3.})), 3., 1e-6);

  EXPECT_NEAR(wrap(Distance{}, std::vector{3.})[0U], 3., 1e-6);
  EXPECT_NEAR(wrap(Distance{}, std::array{-3.})[0U], 3., 1e-6);
  EXPECT_NEAR((wrap(Distance{}, std::array{-3.})[std::integral_constant<std::size_t, 0>{}]), 3., 1e-6);
  EXPECT_NEAR((wrap(Distance{}, std::tuple{-3.})[std::integral_constant<std::size_t, 0>{}]), 3., 1e-6);
  EXPECT_NEAR(wrap(Distance{}, std::tuple{-3.})[0U], 3., 1e-6);
}
