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
 * \brief Tests for coordinates::Dimensions
 */

#include <type_traits>
#include "coordinates/concepts/fixed_pattern.hpp"
#include "coordinates/concepts/pattern.hpp"
#include "coordinates/concepts/euclidean_pattern.hpp"
#include "coordinates/concepts/descriptor.hpp"
#include "coordinates/functions/get_stat_dimension.hpp"
#include "coordinates/functions/get_is_euclidean.hpp"
#include "coordinates/traits/dimension_of.hpp"
#include "coordinates/traits/stat_dimension_of.hpp"
#include "coordinates/descriptors/Dimensions.hpp"
#include "collections/tests/tests.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;

TEST(coordinates, Dimensions_fixed)
{
  static_assert(descriptor<Dimensions<2>>);
  static_assert(descriptor<Dimensions<3>>);
  static_assert(descriptor<Axis>);
  static_assert(pattern<Dimensions<3>>);

  static_assert(get_dimension(Dimensions<3>{}) == 3);
  static_assert(get_dimension(Axis{}) == 1);
  static_assert(get_stat_dimension(Dimensions<3>{}) == 3);
  static_assert(get_stat_dimension(Axis{}) == 1);
  static_assert(get_dimension(Dimensions{Axis {}}) == 1);
  static_assert(get_dimension(Dimensions{std::integral_constant<int, 3> {}}) == 3);
  static_assert(get_is_euclidean(Dimensions<3>{}));

  static_assert(dimension_of_v<Dimensions<0>> == 0);
  static_assert(dimension_of_v<Dimensions<3>> == 3);
  static_assert(dimension_of_v<Axis> == 1);
  static_assert(stat_dimension_of_v<Dimensions<3>> == 3);
  static_assert(stat_dimension_of_v<Axis> == 1);

  static_assert(fixed_pattern<Dimensions<3>>);
  static_assert(fixed_pattern<Axis>);
  static_assert(euclidean_pattern<Dimensions<1>>);
  static_assert(euclidean_pattern<Dimensions<2>>);
  static_assert(euclidean_pattern<Axis>);

  static_assert(static_cast<std::integral_constant<int, 3>>(Dimensions{std::integral_constant<int, 3> {}}) == 3);
  static_assert(static_cast<std::size_t>(Dimensions{std::integral_constant<int, 3> {}}) == 3);
}


TEST(coordinates, Dimensions_dynamic)
{
  using D = Dimensions<>;

  static_assert(pattern<D>);
  static_assert(dynamic_pattern<D>);
  static_assert(not fixed_pattern<D>);
  static_assert(dynamic_pattern<D>);
  static_assert(euclidean_pattern<D>);
  static_assert(euclidean_pattern<D>);
  static_assert(descriptor<D>);
  static_assert(dimension_of_v<D> == stdex::dynamic_extent);
  static_assert(stat_dimension_of_v<D> == stdex::dynamic_extent);
  static_assert(get_dimension(D {0_uz}) == 0);
  static_assert(get_dimension(D {3_uz}) == 3);
  static_assert(get_dimension(Dimensions {0u}) == 0);
  static_assert(get_dimension(Dimensions {3u}) == 3);
  static_assert(get_stat_dimension(Dimensions{3u}) == 3);
  static_assert(get_dimension(Dimensions<stdex::dynamic_extent> {Axis {}}) == 1);
  static_assert(get_dimension(Dimensions<stdex::dynamic_extent> {Dimensions<3> {}}) == 3);
  static_assert(static_cast<std::size_t>(Dimensions {3u}) == 3);
}


TEST(coordinates, Dimensions_assignment)
{
  static_assert(std::is_assignable_v<Dimensions<10>&, Dimensions<10>>);
  static_assert(not std::is_assignable_v<Dimensions<10>&, Dimensions<11>>);
  static_assert(not std::is_assignable_v<Dimensions<10>&, Dimensions<>>);
  static_assert(not std::is_assignable_v<Dimensions<10>&, std::size_t>);

  using D = Dimensions<>;

  static_assert(std::is_assignable_v<D&, D>);
  static_assert(std::is_assignable_v<D&, Dimensions<10>>);
  static_assert(std::is_assignable_v<D&, std::integral_constant<int, 3>>);
  static_assert(std::is_assignable_v<D&, Dimensions<11>>);
  static_assert(std::is_assignable_v<D&, std::vector<std::size_t>>);
  static_assert(std::is_assignable_v<D&, std::array<std::size_t, 3>>);
  static_assert(std::is_assignable_v<D&, std::size_t>);

  Dimensions d {5u};
  EXPECT_EQ(get_dimension(d), 5);
  d = 6u;
  EXPECT_EQ(get_dimension(d), 6);
  d = Dimensions<7>{};
  EXPECT_EQ(get_dimension(d), 7);
  d = std::tuple {Dimensions<3>{}, Dimensions<5>{}};
  EXPECT_EQ(get_dimension(d), 8);
  d = std::vector {4_uz, 5_uz};
  EXPECT_EQ(get_dimension(d), 9);
}

#include "coordinates/functions/to_stat_space.hpp"
#include "coordinates/functions/from_stat_space.hpp"
#include "coordinates/functions/wrap.hpp"

TEST(coordinates, Dimensions_transformations)
{
  EXPECT_NEAR(to_stat_space(Axis{}, std::array{3.})[0U], 3., 1e-6);
  EXPECT_NEAR(from_stat_space(Axis{}, std::array{3.})[0U], 3., 1e-6);
  EXPECT_NEAR(wrap(Axis{}, std::array{3.})[0U], 3., 1e-6);

  EXPECT_NEAR(to_stat_space(Dimensions<5>{}, std::array{1., 2., 3., 4., 5.})[2U], 3., 1e-6);
  EXPECT_NEAR(from_stat_space(Dimensions<5>{}, std::array{1., 2., 3., 4., 5.})[2U], 3., 1e-6);
  EXPECT_NEAR(wrap(Dimensions<5>{}, std::array{1., 2., 3., 4., 5.})[2U], 3., 1e-6);

  EXPECT_NEAR(to_stat_space(Dimensions{5}, std::vector{1., 2., 3., 4., 5.})[2U], 3., 1e-6);
  EXPECT_NEAR(from_stat_space(Dimensions{5}, std::vector{1., 2., 3., 4., 5.})[2U], 3., 1e-6);
  EXPECT_NEAR(wrap(Dimensions{5}, std::vector{1., 2., 3., 4., 5.})[2U], 3., 1e-6);

  EXPECT_NEAR(std::get<2>(to_stat_space(Dimensions<5>{}, std::tuple{1., 2., 3., 4., 5.})), 3., 1e-6);
  EXPECT_NEAR(std::get<3>(from_stat_space(Dimensions<5>{}, std::tuple{1., 2., 3., 4, 5.})), 4, 1e-6);
  EXPECT_NEAR(std::get<4>(wrap(Dimensions<5>{}, std::tuple{1., 2., 3., 4., 5.f})), 5.f, 1e-6);
}
