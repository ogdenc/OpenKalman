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
 * \brief Tests for patterns::fixed_pattern objects
 */

#include "collections/tests/tests.hpp"
#include "patterns/concepts/descriptor.hpp"
#include "patterns/concepts/pattern.hpp"
#include "patterns/concepts/euclidean_pattern.hpp"
#include "patterns/functions/get_stat_dimension.hpp"
#include "patterns/functions/get_is_euclidean.hpp"
#include "patterns/traits/dimension_of.hpp"
#include "patterns/traits/stat_dimension_of.hpp"
#include "patterns/concepts/fixed_pattern.hpp"
#include "patterns/descriptors/Dimensions.hpp"
#include "patterns/descriptors/Distance.hpp"
#include "patterns/descriptors/Angle.hpp"
#include "patterns/descriptors/Inclination.hpp"
#include "patterns/descriptors/Polar.hpp"
#include "patterns/descriptors/Spherical.hpp"

using namespace OpenKalman;
using namespace OpenKalman::patterns;

TEST(patterns, fixed_pattern)
{
  static_assert(dimension_of_v<std::tuple<>> == 0);
  static_assert(fixed_pattern<std::tuple<>>);
  static_assert(euclidean_pattern<std::tuple<>>);
  static_assert(dimension_of_v<stdex::ranges::empty_view<Dimensions<1>>> == 0);
  static_assert(fixed_pattern<stdex::ranges::empty_view<Dimensions<1>>>);
  static_assert(euclidean_pattern<stdex::ranges::empty_view<Dimensions<1>>>);
  static_assert(euclidean_pattern<stdex::ranges::empty_view<Distance>>); // Euclidean because it is empty

  static_assert(fixed_pattern<std::vector<Dimensions<0>>>);
  static_assert(dimension_of_v<std::vector<Dimensions<0>>> == 0);
  static_assert(stat_dimension_of_v<std::vector<Dimensions<0>>> == 0);
  static_assert(euclidean_pattern<std::vector<Dimensions<0>>>);

  static_assert(fixed_pattern<std::tuple<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>>);
  static_assert(not fixed_pattern<std::tuple<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, std::size_t>>);
  static_assert(fixed_pattern<std::array<std::integral_constant<std::size_t, 2>, 3>>);
  static_assert(not fixed_pattern<std::array<std::size_t, 3>>);
  static_assert(euclidean_pattern<std::tuple<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>>);

  static_assert(fixed_pattern<std::tuple<Axis, Axis, Axis>>);
  static_assert(euclidean_pattern<std::tuple<Axis, Axis, Axis>>);
  static_assert(fixed_pattern<std::tuple<Axis, Axis, angle::Radians>>);
  static_assert(not euclidean_pattern<std::tuple<Axis, Axis, angle::Radians>>);
  static_assert(fixed_pattern<std::tuple<angle::Radians, Axis, Axis>>);
  static_assert(not euclidean_pattern<std::tuple<angle::Radians, Axis, Axis>>);
  static_assert(fixed_pattern<std::tuple<Axis, Axis, angle::Radians>>);
  static_assert(descriptor_collection<std::tuple<Axis>>);
  static_assert(not descriptor<std::tuple<Axis>>);

  static_assert(descriptor<std::reference_wrapper<Distance>>);
  static_assert(descriptor<stdex::reference_wrapper<Distance>>);
  static_assert(descriptor_collection<std::tuple<stdex::reference_wrapper<Distance>>>);
  static_assert(descriptor_collection<std::tuple<stdex::reference_wrapper<Distance>, Angle<>>>);

  static_assert(fixed_pattern<std::reference_wrapper<Distance>>);
  static_assert(fixed_pattern<stdex::reference_wrapper<Distance>>);
  static_assert(fixed_pattern<std::tuple<stdex::reference_wrapper<Distance>>>);
  static_assert(fixed_pattern<std::tuple<stdex::reference_wrapper<Distance>, Angle<>>>);
  static_assert(not fixed_pattern<std::tuple<stdex::reference_wrapper<Any<>>, Angle<>>>);

  static_assert(dimension_of_v<Distance[5]> == 5);
  static_assert(stat_dimension_of_v<Distance[5]> == 5);
  static_assert(dimension_of_v<Polar<>[5]> == 10);
  static_assert(stat_dimension_of_v<Polar<>[5]> == 15);
  static_assert(fixed_pattern<Polar<>[5]>);
  static_assert(descriptor_collection<Polar<>[5]>);
  static_assert(euclidean_pattern<Axis[5]>);
  static_assert(not euclidean_pattern<Polar<>[5]>);

  static_assert(get_dimension(Dimensions{std::integral_constant<std::size_t, 2> {}}) == 2u);
  static_assert(get_dimension(std::tuple<Axis, Axis, angle::Radians>{}) == 3u);
  static_assert(get_dimension(std::tuple{Polar<Distance, angle::PositiveRadians>{}}) == 2u);
  static_assert(get_stat_dimension(std::tuple{Polar<angle::PositiveDegrees, Distance>{}}) == 3u);
  static_assert(get_stat_dimension(std::tuple<Axis, Dimensions<2>, angle::Radians>{}) == 5u);
  static_assert(get_is_euclidean(std::tuple<Axis, Dimensions<5>, Axis>{}));
  static_assert(not get_is_euclidean(std::tuple<Axis, Dimensions<5>, angle::Radians>{}));

  static_assert(dimension_of_v<std::tuple<Axis, Axis, angle::Radians, Polar<>, Spherical<>>> == 8u);
  static_assert(stat_dimension_of_v<std::tuple<angle::Radians, Axis, Axis, Polar<>, Spherical<>>> == 11u);
}

