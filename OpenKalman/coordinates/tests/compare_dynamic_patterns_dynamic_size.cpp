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

TEST(coordinates, compare_dynamic_pattern_dynamic_size)
{
  EXPECT_TRUE(compare_three_way(std::vector<Axis>{}, std::vector<Axis>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, std::vector{Axis{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Axis{}, std::vector{Axis{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, Axis{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, std::vector{Axis{}, Axis{}}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}, Axis{}}, std::vector{Axis{}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}, Axis{}}, std::vector{Axis{}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, std::vector{angle::Radians{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, std::vector{Polar<>{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector<Axis>{}, std::vector{angle::Radians{}}) == stdcompat::partial_ordering::less);

  EXPECT_TRUE(compare_three_way(std::vector{angle::Radians{}}, std::vector<Axis>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Radians{}}, std::vector{angle::Radians{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Degrees{}}, std::vector{Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Degrees{}}, std::vector{Angle<values::fixed_value<double, -180>, values::fixed_value<long double, 180>>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{angle::PositiveDegrees{}}, std::vector{Angle<std::integral_constant<int, 0>, std::integral_constant<std::size_t, 360>>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Degrees{}}, std::vector{angle::PositiveDegrees{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Degrees{}}, std::vector{angle::Radians{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{inclination::Radians{}}, std::vector{inclination::Degrees{}}) == stdcompat::partial_ordering::unordered);

  EXPECT_TRUE(compare_three_way(std::vector<Axis>{}, std::vector{inclination::Radians{}}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::vector{inclination::Radians{}}, std::vector<Axis>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{inclination::Radians{}}, std::vector{inclination::Radians{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{inclination::Degrees{}}, std::vector{Inclination<std::integral_constant<int, 180>>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{inclination::Degrees{}}, std::vector{Inclination<values::fixed_value<long double, 180>>{}}) == stdcompat::partial_ordering::equivalent);

  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, Dimensions{1}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, Dimensions<1>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Dimensions{1}, std::vector{Axis{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}, Axis{}, Axis{}}, Dimensions<3>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}, Axis{}}, Dimensions<3>{}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::tuple<Axis, Axis>{}, std::vector{Dimensions<4>{}}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::vector{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);

  EXPECT_TRUE(compare_three_way(std::tuple<>{}, std::vector<Axis>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector<Axis>{}, std::tuple<>{}) == stdcompat::partial_ordering::equivalent);

  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, Dimensions{1}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Dimensions{1}, std::vector{Axis{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}}, Dimensions{1}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}, Axis{}, Axis{}}, Dimensions{3}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Axis{}, Axis{}}, Dimensions{4}) == stdcompat::partial_ordering::less);

  EXPECT_TRUE(compare_three_way(Dimensions{4}, std::vector{Axis{}, Axis{}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Radians{}}, angle::Radians{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{angle::Degrees{}}, std::tuple<angle::Degrees>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(angle::Radians{}, std::vector{angle::Radians{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{Dimensions<3>{}}, std::tuple<Dimensions<3>, angle::Degrees>{}) == stdcompat::partial_ordering::less);

  EXPECT_TRUE(compare_three_way(angle::Radians{}, std::vector{angle::Radians{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::vector{inclination::Radians{}}, inclination::Radians{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(angle::Radians{}, std::vector{inclination::Radians{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{Polar<Distance, angle::Radians>{}}, Dimensions<5>{}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(Polar<Distance, angle::Radians>{}, std::vector{Dimensions<5>{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{Spherical<Distance, inclination::Radians, angle::Radians>{}}, Dimensions<5>{}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(Spherical<Distance, inclination::Radians, angle::Radians>{}, std::vector{Dimensions<5>{}}) == stdcompat::partial_ordering::unordered);
}

TEST(coordinates, compare_unsized_pattern)
{
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Dimensions<0>{}), stdcompat::ranges::views::repeat(Dimensions<0>{})) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Axis{}), stdcompat::ranges::views::repeat(Dimensions<0>{})) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Dimensions<0>{}), stdcompat::ranges::views::repeat(Axis{})) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Distance{}), stdcompat::ranges::views::repeat(Dimensions<0>{})) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Dimensions<0>{}), stdcompat::ranges::views::repeat(Distance{})) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Angle{}), stdcompat::ranges::views::repeat(Distance{})) == stdcompat::partial_ordering::unordered);

  EXPECT_TRUE(compare_three_way(Dimensions<0>{}, stdcompat::ranges::views::repeat(Dimensions<0>{})) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Dimensions<0>{}, stdcompat::ranges::views::repeat(Axis{})) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(Dimensions<0>{}, stdcompat::ranges::views::repeat(Distance{})) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(Distance{}, stdcompat::ranges::views::repeat(Distance{})) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(Distance{}, stdcompat::ranges::views::repeat(Angle{})) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(Angle{}, stdcompat::ranges::views::repeat(Distance{})) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::tuple{Angle{}, Distance{}}, stdcompat::ranges::views::repeat(Distance{})) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::tuple{Distance{}, Distance{}}, stdcompat::ranges::views::repeat(Distance{})) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::vector{Distance{}, Distance{}}, stdcompat::ranges::views::repeat(Distance{})) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::tuple{Any{Angle{}}, Any{Distance{}}}, stdcompat::ranges::views::repeat(Distance{})) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{Any{Angle{}}, Any{Distance{}}}, stdcompat::ranges::views::repeat(Distance{})) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::vector{Any{Distance{}}, Any{Distance{}}}, stdcompat::ranges::views::repeat(Distance{})) == stdcompat::partial_ordering::less);

  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Dimensions<0>{}), Dimensions<0>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Axis{}), Dimensions<0>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Distance{}), Dimensions<0>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Distance{}), Distance{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Angle{}), Distance{}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Distance{}), Angle{}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Distance{}), std::tuple{Angle{}, Distance{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Distance{}), std::tuple{Distance{}, Distance{}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Distance{}), std::vector{Distance{}, Distance{}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Distance{}), std::tuple{Any{Angle{}}, Any{Distance{}}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Distance{}), std::vector{Any{Angle{}}, Any{Distance{}}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(stdcompat::ranges::views::repeat(Distance{}), std::vector{Any{Distance{}}, Any{Distance{}}}) == stdcompat::partial_ordering::greater);
}