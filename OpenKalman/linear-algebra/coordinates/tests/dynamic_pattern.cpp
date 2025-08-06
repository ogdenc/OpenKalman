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
 * \brief Tests for \ref dynamic_pattern objects
 */

#include "basics/tests/tests.hpp"
#include "basics/basics.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"
#include "linear-algebra/coordinates/traits/stat_dimension_of.hpp"
#include "linear-algebra/coordinates/functions/get_dimension.hpp"
#include "linear-algebra/coordinates/functions/get_stat_dimension.hpp"
#include "linear-algebra/coordinates/concepts/dynamic_pattern.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/descriptors/Distance.hpp"
#include "linear-algebra/coordinates/descriptors/Angle.hpp"
#include "linear-algebra/coordinates/descriptors/Inclination.hpp"
#include "linear-algebra/coordinates/descriptors/Polar.hpp"
#include "linear-algebra/coordinates/descriptors/Spherical.hpp"
#include "linear-algebra/coordinates/descriptors/Any.hpp"
#include "linear-algebra/coordinates/functions/make_pattern.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;
using stdcompat::numbers::pi;

TEST(coordinates, dynamic_pattern_traits)
{
  static_assert(pattern<std::vector<std::size_t>>);
  static_assert(dynamic_pattern<std::vector<std::size_t>>);
  static_assert(not fixed_pattern<std::vector<std::size_t>>);
  static_assert(euclidean_pattern<std::size_t>);
  static_assert(pattern<std::vector<Distance>>);
  static_assert(dynamic_pattern<std::vector<Distance>>);
  static_assert(not dynamic_pattern<std::array<Distance, 4>>);
  static_assert(not dynamic_pattern<std::tuple<Axis, Distance, Dimensions<3>>>);
  static_assert(dynamic_pattern<std::tuple<Axis, Distance, Dimensions<>>>);
  static_assert(dynamic_pattern<std::vector<Any<double>>>);
  static_assert(dimension_of_v<std::tuple<Axis, angle::Degrees, Dimensions<>>> == dynamic_size);
  static_assert(dimension_of_v<std::tuple<Axis, angle::Degrees, Dimensions<3>>> != dynamic_size);
  static_assert(stat_dimension_of_v<std::tuple<Axis, angle::Degrees, Dimensions<>>> == dynamic_size);
  static_assert(stat_dimension_of_v<std::tuple<Axis, angle::Degrees, Dimensions<3>>> != dynamic_size);

  static_assert(euclidean_pattern<stdcompat::ranges::repeat_view<Dimensions<1>>>);
  static_assert(not euclidean_pattern<stdcompat::ranges::repeat_view<Distance>>);
}


TEST(coordinates, dynamic_pattern_functions)
{
  EXPECT_EQ(get_dimension(std::tuple{4u}), 4u);
  EXPECT_EQ(get_dimension(std::tuple{Dimensions{0}}), 0u);
  EXPECT_EQ(get_dimension(std::tuple{Dimensions{5}}), 5u);
  EXPECT_EQ(get_dimension(std::vector{Axis{}}), 1u);
  EXPECT_EQ(get_dimension(std::vector{angle::Degrees{}}), 1u);
  EXPECT_EQ(get_stat_dimension(std::vector{angle::Degrees{}}), 2u);
  EXPECT_EQ(get_dimension(std::vector{Dimensions<5>{}}), 5u);
  EXPECT_EQ(get_dimension(std::vector{Dimensions{5}}), 5u);
  EXPECT_EQ(get_dimension(std::vector{Polar{}}), 2u);
  EXPECT_EQ(get_dimension(std::vector{Polar<Distance, angle::PositiveRadians>{}}), 2u);
  EXPECT_EQ(get_stat_dimension(std::vector{Polar<angle::PositiveDegrees, Distance>{}}), 3u);
  EXPECT_EQ(get_dimension(std::vector{Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{}}), 3u);
  EXPECT_EQ(get_stat_dimension(std::vector{Spherical<inclination::Radians, Distance, angle::PositiveDegrees>{}}), 4u);

  EXPECT_EQ(get_dimension(std::tuple{Dimensions{5}, Dimensions<1>{}, angle::Degrees{}}), 7u);
  EXPECT_EQ(get_dimension(std::tuple{Dimensions{5}, std::integral_constant<unsigned, 1>{}, angle::Degrees{}}), 7u);
  EXPECT_EQ(get_stat_dimension(std::tuple{Axis{}, Dimensions{5}, angle::Degrees{}}), 8u);
  EXPECT_EQ(get_dimension(std::tuple{Axis{}, inclination::Radians{}, angle::Degrees{}, Dimensions{5}}), 8u);
  EXPECT_EQ(get_stat_dimension(std::tuple{Dimensions{5}, Axis{}, inclination::Radians{}, angle::Degrees{}}), 10u);

  EXPECT_EQ(get_stat_dimension(std::array{Any{Dimensions{5}}, Any{Axis{}}, Any{inclination::Radians{}}, Any{angle::Degrees{}}}), 10u);

  EXPECT_TRUE(get_is_euclidean(std::vector{Any{Dimensions<3>{}}, Any{Dimensions<2>{}}, Any{Dimensions<5>{}}}));
  EXPECT_FALSE(get_is_euclidean(std::vector{Any{angle::Radians{}}, Any{Dimensions<2>{}}, Any{Dimensions<3>{}}, Any{Dimensions<3>{}}}));
  EXPECT_FALSE(get_is_euclidean(std::vector{Any{Dimensions<3>{}}, Any{Dimensions<2>{}}, Any{angle::Radians{}}, Any{Dimensions<5>{}}}));
}
