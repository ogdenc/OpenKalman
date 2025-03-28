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
 * \brief Tests for coordinate::Dimensions
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/functions/get_euclidean_size.hpp"
#include "linear-algebra/coordinates/functions/get_component_count.hpp"
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp"
#include "linear-algebra/coordinates/traits/size_of.hpp"
#include "linear-algebra/coordinates/traits/euclidean_size_of.hpp"
#include "linear-algebra/coordinates/traits/component_count_of.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinate;

TEST(coordinates, Dimensions_fixed)
{
  static_assert(descriptor<Dimensions<3>>);
  static_assert(fixed_pattern<Dimensions<3>>);
  static_assert(pattern<Dimensions<3>>);
  static_assert(descriptor<Dimensions<1>>);
  static_assert(descriptor<Dimensions<2>>);
  static_assert(euclidean_pattern<Dimensions<1>>);
  static_assert(euclidean_pattern<Dimensions<2>>);
  static_assert(euclidean_pattern<Axis>);

  static_assert(get_size(Dimensions<3>{}) == 3);
  static_assert(get_size(Axis{}) == 1);
  static_assert(get_euclidean_size(Dimensions<3>{}) == 3);
  static_assert(get_euclidean_size(Axis{}) == 1);
  static_assert(get_component_count(Dimensions<3>{}) == 1);
  static_assert(get_size(Dimensions{Axis {}}) == 1);
  static_assert(get_size(Dimensions{std::integral_constant<int, 3> {}}) == 3);
  static_assert(get_is_euclidean(Dimensions<3>{}));

  static_assert(size_of_v<Dimensions<0>> == 0);
  static_assert(size_of_v<Dimensions<3>> == 3);
  static_assert(size_of_v<Axis> == 1);
  static_assert(euclidean_size_of_v<Dimensions<3>> == 3);
  static_assert(euclidean_size_of_v<Axis> == 1);
  static_assert(component_count_of_v<Dimensions<3>> == 1);

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
  static_assert(size_of_v<D> == dynamic_size);
  static_assert(euclidean_size_of_v<D> == dynamic_size);
  static_assert(get_size(D {0_uz}) == 0);
  static_assert(get_size(D {3_uz}) == 3);
  static_assert(get_size(Dimensions {0u}) == 0);
  static_assert(get_size(Dimensions {3u}) == 3);
  static_assert(get_euclidean_size(Dimensions{3u}) == 3);
  static_assert(get_size(Dimensions<dynamic_size> {Axis {}}) == 1);
  static_assert(get_size(Dimensions<dynamic_size> {Dimensions<3> {}}) == 3);
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
  EXPECT_EQ(get_size(d), 5);
  d = 6u;
  EXPECT_EQ(get_size(d), 6);
  d = Dimensions<7>{};
  EXPECT_EQ(get_size(d), 7);
  d = std::tuple {Dimensions<3>{}, Dimensions<5>{}};
  EXPECT_EQ(get_size(d), 8);
  d = std::vector {4_uz, 5_uz};
  EXPECT_EQ(get_size(d), 9);
}
