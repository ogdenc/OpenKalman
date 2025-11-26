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
 * \brief Tests for \ref coordinates::pattern equivalence
 */

#include "collections/tests/tests.hpp"
#include "coordinates/descriptors/Dimensions.hpp"
#include "coordinates/descriptors/Distance.hpp"
#include "coordinates/descriptors/Angle.hpp"
#include "coordinates/descriptors/Inclination.hpp"
#include "coordinates/descriptors/Polar.hpp"
#include "coordinates/descriptors/Spherical.hpp"
#include "coordinates/descriptors/Any.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;

#include "coordinates/functions/compare_three_way.hpp"
#include "coordinates/functions/compare.hpp"

TEST(coordinates, compare_dynamic_pattern_dynamic_size)
{
  EXPECT_TRUE(compare_three_way(std::vector<Axis>{}, std::vector<Axis>{}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, std::vector{Axis{}}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Axis{}, std::vector{Axis{}}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, Axis{}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, std::vector{Axis{}, Axis{}}) == stdex::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}, Axis{}}, std::vector{Axis{}}) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}, Axis{}}, std::vector{Axis{}}) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, std::vector{angle::Radians{}}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, std::vector{Polar<>{}}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector<Axis>{}, std::vector{angle::Radians{}}) == stdex::partial_ordering::less);

  EXPECT_TRUE(compare_three_way(std::vector{angle::Radians{}}, std::vector<Axis>{}) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Radians{}}, std::vector{angle::Radians{}}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Degrees{}}, std::vector{Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>{}}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Degrees{}}, std::vector{Angle<values::fixed_value<double, -180>, values::fixed_value<long double, 180>>{}}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{angle::PositiveDegrees{}}, std::vector{Angle<std::integral_constant<int, 0>, std::integral_constant<std::size_t, 360>>{}}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Degrees{}}, std::vector{angle::PositiveDegrees{}}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Degrees{}}, std::vector{angle::Radians{}}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{inclination::Radians{}}, std::vector{inclination::Degrees{}}) == stdex::partial_ordering::unordered);

  EXPECT_TRUE(compare_three_way(std::vector<Axis>{}, std::vector{inclination::Radians{}}) == stdex::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::vector{inclination::Radians{}}, std::vector<Axis>{}) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{inclination::Radians{}}, std::vector{inclination::Radians{}}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{inclination::Degrees{}}, std::vector{Inclination<std::integral_constant<int, 180>>{}}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{inclination::Degrees{}}, std::vector{Inclination<values::fixed_value<long double, 180>>{}}) == stdex::partial_ordering::equivalent);

  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, Dimensions{1}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, Dimensions<1>{}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Dimensions{1}, std::vector{Axis{}}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}, Axis{}, Axis{}}, Dimensions<3>{}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}, Axis{}}, Dimensions<3>{}) == stdex::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::tuple<Axis, Axis>{}, std::vector{Dimensions<4>{}}) == stdex::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::vector{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdex::partial_ordering::greater);

  EXPECT_TRUE(compare_three_way(std::tuple<>{}, std::vector<Axis>{}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector<Axis>{}, std::tuple<>{}) == stdex::partial_ordering::equivalent);

  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, Dimensions{1}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Dimensions{1}, std::vector{Axis{}}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, Dimensions{1}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}, Axis{}, Axis{}}, Dimensions{3}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}, Axis{}}, Dimensions{4}) == stdex::partial_ordering::less);

  EXPECT_TRUE(compare_three_way(Dimensions{4}, std::vector{Axis{}, Axis{}}) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Radians{}}, angle::Radians{}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Degrees{}}, std::tuple<angle::Degrees>{}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(angle::Radians{}, std::vector{angle::Radians{}}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Dimensions<3>{}}, std::tuple<Dimensions<3>, angle::Degrees>{}) == stdex::partial_ordering::less);

  EXPECT_TRUE(compare_three_way(angle::Radians{}, std::vector{angle::Radians{}}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{inclination::Radians{}}, inclination::Radians{}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(angle::Radians{}, std::vector{inclination::Radians{}}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{Polar<Distance, angle::Radians>{}}, Dimensions<5>{}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(Polar<Distance, angle::Radians>{}, std::vector{Dimensions<5>{}}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{Spherical<Distance, inclination::Radians, angle::Radians>{}}, Dimensions<5>{}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(Spherical<Distance, inclination::Radians, angle::Radians>{}, std::vector{Dimensions<5>{}}) == stdex::partial_ordering::unordered);
}

TEST(coordinates, compare_unsized_pattern)
{
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Dimensions<0>{}), stdex::ranges::views::repeat(Dimensions<0>{})) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Axis{}), stdex::ranges::views::repeat(Dimensions<0>{})) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Dimensions<0>{}), stdex::ranges::views::repeat(Axis{})) == stdex::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Distance{}), stdex::ranges::views::repeat(Dimensions<0>{})) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Dimensions<0>{}), stdex::ranges::views::repeat(Distance{})) == stdex::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Distance{})) == stdex::partial_ordering::unordered);

  EXPECT_TRUE(compare_three_way(Dimensions<0>{}, stdex::ranges::views::repeat(Dimensions<0>{})) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Dimensions<0>{}, stdex::ranges::views::repeat(Axis{})) == stdex::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(Dimensions<0>{}, stdex::ranges::views::repeat(Distance{})) == stdex::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(Distance{}, stdex::ranges::views::repeat(Distance{})) == stdex::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(Distance{}, stdex::ranges::views::repeat(Angle{})) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(Angle{}, stdex::ranges::views::repeat(Distance{})) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::tuple{Angle{}, Distance{}}, stdex::ranges::views::repeat(Distance{})) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::tuple{Distance{}, Distance{}}, stdex::ranges::views::repeat(Distance{})) == stdex::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::vector{Distance{}, Distance{}}, stdex::ranges::views::repeat(Distance{})) == stdex::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::tuple{Any{Angle{}}, Any{Distance{}}}, stdex::ranges::views::repeat(Distance{})) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{Any{Angle{}}, Any{Distance{}}}, stdex::ranges::views::repeat(Distance{})) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{Any{Distance{}}, Any{Distance{}}}, stdex::ranges::views::repeat(Distance{})) == stdex::partial_ordering::less);

  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Dimensions<0>{}), Dimensions<0>{}) == stdex::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Axis{}), Dimensions<0>{}) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Distance{}), Dimensions<0>{}) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Distance{}), Distance{}) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Angle{}), Distance{}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Distance{}), Angle{}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Distance{}), std::tuple{Angle{}, Distance{}}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Distance{}), std::tuple{Distance{}, Distance{}}) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Distance{}), std::vector{Distance{}, Distance{}}) == stdex::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Distance{}), std::tuple{Any{Angle{}}, Any{Distance{}}}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Distance{}), std::vector{Any{Angle{}}, Any{Distance{}}}) == stdex::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(stdex::ranges::views::repeat(Distance{}), std::vector{Any{Distance{}}, Any{Distance{}}}) == stdex::partial_ordering::greater);

  static_assert(compare<stdex::is_neq>(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Distance{})));
  static_assert(not compare<stdex::is_eq>(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Distance{})));
  static_assert(not compare<stdex::is_lt>(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Distance{})));
  static_assert(not compare<stdex::is_gt>(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Distance{})));
  static_assert(not compare<stdex::is_lteq>(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Distance{})));
  static_assert(not compare<stdex::is_gteq>(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Distance{})));

  static_assert(compare<stdex::is_eq>(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Angle{})));
  static_assert(compare<stdex::is_gteq>(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Angle{})));
  static_assert(compare<stdex::is_lteq>(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Angle{})));
  static_assert(not compare<stdex::is_neq>(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Angle{})));
  static_assert(not compare<stdex::is_lt>(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Angle{})));
  static_assert(not compare<stdex::is_gt>(stdex::ranges::views::repeat(Angle{}), stdex::ranges::views::repeat(Angle{})));
}