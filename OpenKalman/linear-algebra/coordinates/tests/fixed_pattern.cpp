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
 * \brief Tests for coordinates::fixed_pattern objects
 */

#include "basics/tests/tests.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/functions/get_stat_dimension.hpp"
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"
#include "linear-algebra/coordinates/traits/stat_dimension_of.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/descriptors/Distance.hpp"
#include "linear-algebra/coordinates/descriptors/Angle.hpp"
#include "linear-algebra/coordinates/descriptors/Inclination.hpp"
#include "linear-algebra/coordinates/descriptors/Polar.hpp"
#include "linear-algebra/coordinates/descriptors/Spherical.hpp"
#include "linear-algebra/coordinates/views/comparison.hpp"

using namespace OpenKalman::coordinates;

TEST(coordinates, fixed_pattern)
{
  static_assert(fixed_pattern<std::tuple<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>>);
  static_assert(not fixed_pattern<std::tuple<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>, std::size_t>>);
  static_assert(fixed_pattern<std::array<std::integral_constant<std::size_t, 2>, 3>>);
  static_assert(fixed_pattern<comparison_view<std::tuple<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>>>);
  static_assert(not fixed_pattern<std::array<std::size_t, 3>>);
  static_assert(euclidean_pattern<std::tuple<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>>);

  static_assert(fixed_pattern<std::tuple<Axis, Axis, Axis>>);
  static_assert(euclidean_pattern<std::tuple<Axis, Axis, Axis>>);
  static_assert(fixed_pattern<std::tuple<Axis, Axis, angle::Radians>>);
  static_assert(not euclidean_pattern<std::tuple<Axis, Axis, angle::Radians>>);
  static_assert(fixed_pattern<std::tuple<angle::Radians, Axis, Axis>>);
  static_assert(not euclidean_pattern<std::tuple<angle::Radians, Axis, Axis>>);
  static_assert(fixed_pattern<std::tuple<Axis, Axis, angle::Radians>>);
  static_assert(descriptor_tuple<std::tuple<Axis>>);
  static_assert(not descriptor<std::tuple<Axis>>);

  static_assert(get_dimension(Dimensions{std::tuple<Axis, Axis> {}}) == 2u);
  static_assert(get_dimension(std::tuple<Axis, Axis, angle::Radians>{}) == 3u);
  static_assert(get_dimension(std::tuple{Polar<Distance, angle::PositiveRadians>{}}) == 2u);
  static_assert(get_stat_dimension(std::tuple{Polar<angle::PositiveDegrees, Distance>{}}) == 3u);
  static_assert(get_stat_dimension(std::tuple<Axis, Dimensions<2>, angle::Radians>{}) == 5u);
  static_assert(get_is_euclidean(std::tuple<Axis, Dimensions<5>, Axis>{}));
  static_assert(not get_is_euclidean(std::tuple<Axis, Dimensions<5>, angle::Radians>{}));

  static_assert(dimension_of_v<std::tuple<Axis, Axis, angle::Radians, Polar<>, Spherical<>>> == 8u);
  static_assert(stat_dimension_of_v<std::tuple<angle::Radians, Axis, Axis, Polar<>, Spherical<>>> == 11u);
}

