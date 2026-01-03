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
 * \brief Tests for \ref dynamic_pattern objects
 */

#include "collections/tests/tests.hpp"
#include "patterns/concepts/pattern.hpp"
#include "patterns/concepts/euclidean_pattern.hpp"
#include "patterns/concepts/descriptor.hpp"
#include "patterns/traits/dimension_of.hpp"
#include "patterns/traits/stat_dimension_of.hpp"
#include "patterns/functions/get_dimension.hpp"
#include "patterns/functions/get_stat_dimension.hpp"
#include "patterns/concepts/dynamic_pattern.hpp"
#include "patterns/descriptors/Dimensions.hpp"
#include "patterns/descriptors/Distance.hpp"
#include "patterns/descriptors/Angle.hpp"
#include "patterns/descriptors/Inclination.hpp"
#include "patterns/descriptors/Polar.hpp"
#include "patterns/descriptors/Spherical.hpp"
#include "patterns/descriptors/Any.hpp"
#include "patterns/functions/make_descriptor_range.hpp"

using namespace OpenKalman;
using namespace OpenKalman::patterns;
using stdex::numbers::pi;

TEST(patterns, dynamic_pattern_traits)
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
  static_assert(dimension_of_v<std::tuple<Axis, angle::Degrees, Dimensions<>>> == stdex::dynamic_extent);
  static_assert(dimension_of_v<std::tuple<Axis, angle::Degrees, Dimensions<3>>> != stdex::dynamic_extent);
  static_assert(stat_dimension_of_v<std::tuple<Axis, angle::Degrees, Dimensions<>>> == stdex::dynamic_extent);
  static_assert(stat_dimension_of_v<std::tuple<Axis, angle::Degrees, Dimensions<3>>> != stdex::dynamic_extent);

  static constexpr unsigned a1[3] = {2U, 3U, 4U};
  static_assert(get_dimension(a1) == 9U);
  static_assert(get_stat_dimension(a1) == 9U);
  static_assert(get_is_euclidean(a1));

  Any<> a2[3] = {Any{Distance{}}, Any{Axis{}}, Any{Polar{}}};
  EXPECT_EQ(get_dimension(a2), 4U);
  EXPECT_EQ(get_stat_dimension(a2), 5U);
  EXPECT_FALSE(get_is_euclidean(a2));

  static_assert(dimension_of_v<unsigned[5]> == stdex::dynamic_extent);
  static_assert(stat_dimension_of_v<unsigned[5]> == stdex::dynamic_extent);
  static_assert(dimension_of_v<Any<>[5]> == stdex::dynamic_extent);
  static_assert(stat_dimension_of_v<Any<>[5]> == stdex::dynamic_extent);
  static_assert(dynamic_pattern<Any<>[5]>);
  static_assert(descriptor_collection<Any<>[5]>);
  static_assert(euclidean_pattern<unsigned[5]>);
  static_assert(not euclidean_pattern<Any<>[5]>);

  static_assert(euclidean_pattern<collections::replicate_view<std::tuple<Dimensions<1>>, std::integral_constant<std::size_t, 8>>>);
  static_assert(not euclidean_pattern<collections::replicate_view<std::tuple<Distance>, std::integral_constant<std::size_t, 8>>>);
  static_assert(euclidean_pattern<stdex::ranges::repeat_view<Dimensions<1>, std::size_t>>);
  static_assert(euclidean_pattern<stdex::ranges::repeat_view<Dimensions<1>>>);

  static_assert(descriptor<std::reference_wrapper<Any<>>>);
  static_assert(descriptor<stdex::reference_wrapper<Any<>>>);
  static_assert(descriptor_collection<std::vector<stdex::reference_wrapper<Distance>>>);
  static_assert(descriptor_collection<std::vector<stdex::reference_wrapper<Any<>>>>);
  static_assert(descriptor_collection<std::tuple<stdex::reference_wrapper<Any<>>, Distance>>);

  static_assert(dynamic_pattern<std::reference_wrapper<Any<>>>);
  static_assert(dynamic_pattern<stdex::reference_wrapper<Any<>>>);
  static_assert(dynamic_pattern<std::vector<stdex::reference_wrapper<Distance>>>);
  static_assert(dynamic_pattern<std::vector<stdex::reference_wrapper<Any<>>>>);
  static_assert(dynamic_pattern<std::tuple<stdex::reference_wrapper<Any<>>, Distance>>);

  static_assert(pattern<std::vector<std::size_t>>);
  static_assert(dynamic_pattern<std::vector<std::size_t>>);
}


TEST(patterns, dynamic_pattern_functions)
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


#include "patterns/functions/make_descriptor_range.hpp"

TEST(patterns, make_descriptor_range)
{
  static_assert(stdex::same_as<decltype(make_descriptor_range()), stdex::ranges::empty_view<Dimensions<1>>>);
  static_assert(stdex::same_as<decltype(make_descriptor_range(Dimensions{5})), stdex::ranges::single_view<Dimensions<>>>);
  static_assert(stdex::same_as<decltype(make_descriptor_range(Dimensions{1}, Dimensions{5}, Dimensions{2})), std::array<Dimensions<>, 3>>);
  static_assert(stdex::same_as<decltype(make_descriptor_range(Distance{}, Distance{})), std::array<Distance, 2>>);
  static_assert(stdex::same_as<decltype(make_descriptor_range(Dimensions{1}, angle::Radians{}, Dimensions{1})), std::array<Any<>, 3>>);
}
