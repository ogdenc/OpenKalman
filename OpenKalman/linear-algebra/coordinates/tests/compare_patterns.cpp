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

#include "basics/tests/tests.hpp"
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

#include "linear-algebra/coordinates/functions/compare.hpp"

TEST(coordinates, compare_fixed_pattern)
{
  static_assert(compare(Dimensions<3>{}, std::tuple<Axis, Axis, Axis>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(Dimensions<3>{}, std::tuple<Axis, Axis, Axis, Axis>{}) == stdcompat::partial_ordering::less);
  static_assert(compare(Dimensions<4>{}, std::tuple<Axis, Axis, Axis, Axis>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(Dimensions<4>{}, std::tuple<Axis, Axis, Axis>{}) == stdcompat::partial_ordering::greater);

  static_assert(compare(std::tuple{Axis{}}, Dimensions<1>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(Dimensions<1>{}, std::tuple{Axis{}}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(std::tuple{Axis{}}, Dimensions<1>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(make_pattern(Axis{}, Axis{}, Axis{}), Dimensions<3>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(make_pattern(Axis{}, Axis{}), Dimensions<3>{}) == stdcompat::partial_ordering::less);
  static_assert(compare(std::tuple<Axis, Axis, Axis>{}, Dimensions<3>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(std::tuple<Axis, Axis, Axis, Axis>{}, Dimensions<3>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare(std::tuple<Axis, Axis, Axis, Axis>{}, Dimensions<4>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(std::tuple<Axis, Axis, Axis>{}, Dimensions<4>{}) == stdcompat::partial_ordering::less);

  static_assert(compare(std::tuple<Axis, Axis>{}, make_pattern(Dimensions<4>{})) == stdcompat::partial_ordering::less);
  static_assert(compare(make_pattern(Dimensions<4>{}), std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare(make_pattern(Dimensions<4>{}), std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare(std::tuple<>{}, make_pattern()) == stdcompat::partial_ordering::equivalent);

  static_assert(compare(std::tuple<Axis, Axis>{}, std::tuple{Dimensions<4>{}}) == stdcompat::partial_ordering::less);
  static_assert(compare(std::tuple{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare(std::tuple{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);

  static_assert(compare(std::tuple{}, std::tuple{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(stdcompat::ranges::views::empty<Dimensions<1>>, std::tuple{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(std::tuple{}, stdcompat::ranges::views::empty<Dimensions<1>>) == stdcompat::partial_ordering::equivalent);

  static_assert(compare(std::tuple<Axis, angle::Radians>{}, std::tuple<Axis, angle::Radians>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(std::tuple<>{}, std::tuple<Axis, angle::Radians>{}) == stdcompat::partial_ordering::less);
  static_assert(compare(make_pattern(), std::tuple<Axis, angle::Radians>{}) == stdcompat::partial_ordering::less);
  static_assert(compare(std::tuple<Axis>{}, std::tuple<Axis, angle::Radians>{}) == stdcompat::partial_ordering::less);
  static_assert(compare(std::tuple<Axis, angle::Radians>{}, std::tuple<Axis, angle::Radians, Axis>{}) == stdcompat::partial_ordering::less);
  static_assert(compare(std::tuple<Axis, angle::Radians, Axis>{}, std::tuple<Axis, angle::Radians>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare(std::tuple<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{}, std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(make_pattern(Axis{}, Dimensions<3>{}, angle::Radians{}, Dimensions<5>{}), std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare(make_pattern(Axis{}, Dimensions<3>{}, angle::Radians{}, Axis{}, Dimensions<2>{}), std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<4>>{}) == stdcompat::partial_ordering::less);
}


TEST(coordinates, compare_dynamic_pattern)
{
  static_assert(compare(Dimensions<0>{}, std::tuple<>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(Dimensions{0}, std::tuple<>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(std::tuple<>{}, Dimensions{0}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(Dimensions<2>{}, std::tuple<>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare(std::tuple<>{}, std::tuple<Axis>{}) == stdcompat::partial_ordering::less);

  static_assert(compare(Dimensions{3}, std::tuple<Axis, Axis, Axis>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(Dimensions{4}, std::tuple<Axis, Dimensions<2>, Axis>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(Dimensions{3}, std::tuple<Axis, Dimensions<2>>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(Dimensions{3}, std::tuple<Axis, Dimensions<2>, Axis>{}) == stdcompat::partial_ordering::less);
  static_assert(compare(Dimensions{3}, std::tuple<Axis, Dimensions<2>, Axis>{}) == stdcompat::partial_ordering::less);
  static_assert(compare(Dimensions{3}, std::tuple<Axis, Axis, Axis, Axis>{}) == stdcompat::partial_ordering::less);

  static_assert(compare(std::tuple<Axis, Axis, Axis>{}, Dimensions{3}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(std::tuple<Axis, Dimensions<2>, Axis>{}, Dimensions{4}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(std::tuple<Axis, Dimensions<2>>{}, Dimensions{3}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare(std::tuple<Axis, Dimensions<2>, Axis>{}, Dimensions{3}) == stdcompat::partial_ordering::greater);
  static_assert(compare(std::tuple<Axis, Axis, Axis, Axis>{}, Dimensions{3}) == stdcompat::partial_ordering::greater);

  EXPECT_TRUE(compare(Any{Axis{}}, Any{Axis{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(Any{Dimensions<3>{}}, Any{Dimensions<3>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(Any{Dimensions<4>{}}, Any{Dimensions<3>{}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(Any{Dimensions{4}}, Any{Dimensions{4}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(Any{Dimensions{6}}, Any{Dimensions{5}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(Any{Dimensions<2>{}}, Dimensions<2>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(Any{Dimensions<2>{}}, Dimensions{2}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(Any{Dimensions<2>{}}, Dimensions{3}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(Dimensions{3}, Any{Dimensions<3>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(Dimensions{2}, Any{Dimensions<3>{}}) == stdcompat::partial_ordering::less);

  EXPECT_TRUE(compare(Any{angle::Radians{}}, Any{angle::Radians{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(Any{angle::Radians{}}, Any{angle::Degrees{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare(Any{angle::Radians{}}, angle::Radians{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(angle::Radians{}, Any{angle::Radians{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(angle::Degrees{}, Any{angle::Radians{}}) == stdcompat::partial_ordering::unordered);

  EXPECT_TRUE(compare(std::vector<Axis>{}, std::vector<Axis>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{Axis{}}, std::vector{Axis{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(Axis{}, std::vector{Axis{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{Axis{}}, Axis{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{Axis{}}, std::vector{Axis{}, Axis{}}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(std::vector{Axis{}, Axis{}}, std::vector{Axis{}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(std::vector{Axis{}, Axis{}}, std::vector{Axis{}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(std::vector{Axis{}}, std::vector{angle::Radians{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare(std::vector{Axis{}}, std::vector{Polar<>{}}) == stdcompat::partial_ordering::unordered);

  EXPECT_TRUE(compare(std::vector<Axis>{}, std::vector{angle::Radians{}}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(std::vector{angle::Radians{}}, std::vector<Axis>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(std::vector{angle::Radians{}}, std::vector{angle::Radians{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{angle::Degrees{}}, std::vector{Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{angle::Degrees{}}, std::vector{Angle<values::Fixed<double, -180>, values::Fixed<long double, 180>>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{angle::PositiveDegrees{}}, std::vector{Angle<std::integral_constant<int, 0>, std::integral_constant<std::size_t, 360>>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{angle::Degrees{}}, std::vector{angle::PositiveDegrees{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare(std::vector{angle::Degrees{}}, std::vector{angle::Radians{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare(std::vector{inclination::Radians{}}, std::vector{inclination::Degrees{}}) == stdcompat::partial_ordering::unordered);

  EXPECT_TRUE(compare(std::vector<Axis>{}, std::vector{inclination::Radians{}}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(std::vector{inclination::Radians{}}, std::vector<Axis>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(std::vector{inclination::Radians{}}, std::vector{inclination::Radians{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{inclination::Degrees{}}, std::vector{Inclination<std::integral_constant<int, 180>>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{inclination::Degrees{}}, std::vector{Inclination<values::Fixed<long double, 180>>{}}) == stdcompat::partial_ordering::equivalent);

  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}, Dimensions{1}), make_pattern(Dimensions{1}, angle::Radians{}, Dimensions{1})) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}, Dimensions{1}), make_pattern(Dimensions{1}, angle::Radians{}, Dimensions{1}, Dimensions{1})) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}, Dimensions{1}, Dimensions{1}), make_pattern(Dimensions{1}, angle::Radians{}, Dimensions{1})) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}, Dimensions{1}), make_pattern(Dimensions{1}, angle::Radians{}, Dimensions{1})) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Dimensions{3}, Dimensions{2}, angle::Radians{}, Dimensions{5}), make_pattern(Dimensions{2}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{})) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}, Dimensions{1}), make_pattern(Dimensions{1}, angle::Radians{}, Dimensions{1})) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Dimensions{3}, Dimensions<2>{}, angle::Radians{}, Dimensions<4>{}), make_pattern(Dimensions{2}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{})) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(make_pattern(Dimensions{3}, Dimensions<2>{}, angle::Radians{}, Dimensions<5>{}), make_pattern(Dimensions{2}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<2>{})) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(std::vector<Polar<>>{}, std::vector<Polar<>>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Polar{}, Dimensions{1}), make_pattern(Polar{}, Dimensions{1})) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Polar{}, Dimensions{1}), std::vector<Polar<>>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(std::vector{Spherical<Distance, angle::Radians, inclination::Radians>{}}, std::vector{Spherical<Distance, angle::Radians, inclination::Radians>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}, angle::Radians{}), make_pattern(Dimensions{1}, angle::Radians{}, Dimensions{1})) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}), make_pattern(Polar<Distance, angle::Radians>{})) == stdcompat::partial_ordering::unordered);

  EXPECT_TRUE(compare(std::vector{Axis{}}, Dimensions{1}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{Axis{}}, Dimensions<1>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(Dimensions{1}, std::vector{Axis{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{Axis{}, Axis{}, Axis{}}, Dimensions<3>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{Axis{}, Axis{}}, Dimensions<3>{}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(std::tuple<Axis, Axis>{}, std::vector{Dimensions<4>{}}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(std::vector{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(std::vector{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(std::tuple<>{}, std::vector<Axis>{}) == stdcompat::partial_ordering::equivalent);

  EXPECT_TRUE(compare(std::vector{Axis{}}, Dimensions{1}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(Dimensions{1}, std::vector{Axis{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{Axis{}}, Dimensions{1}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{Axis{}, Axis{}, Axis{}}, Dimensions{3}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{Axis{}, Axis{}}, Dimensions{4}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(std::tuple<Axis, Axis>{}, make_pattern(Dimensions{4})) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(make_pattern(Dimensions{4}), std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(make_pattern(Dimensions{2}), std::tuple<Axis, Axis, Axis, Axis>{}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(Dimensions{4}, std::vector{Axis{}, Axis{}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(std::vector{angle::Radians{}}, angle::Radians{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{angle::Degrees{}}, std::tuple<angle::Degrees>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Dimensions<3>{}, angle::Degrees{}), std::tuple<Dimensions<3>, angle::Degrees>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(angle::Radians{}, std::vector{angle::Radians{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Dimensions<3>{}, angle::Degrees{}), Dimensions<3>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(std::vector{Dimensions<3>{}}, std::tuple<Dimensions<3>, angle::Degrees>{}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(make_pattern(Dimensions<3>{}, angle::Degrees{}, Dimensions<5>{}), std::tuple<Dimensions<3>, angle::Degrees, Dimensions<5>>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::tuple<Dimensions<3>, angle::Degrees, Dimensions<5>>{}, make_pattern(Dimensions<3>{}, angle::Degrees{}, Dimensions<5>{})) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}), std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}), std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}), std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<4>>{}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}), std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<2>>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, Dimensions<3>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}), std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}) == stdcompat::partial_ordering::unordered);
  EXPECT_FALSE(compare(make_pattern(Dimensions{1}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}), std::tuple<Dimensions<4>, angle::Degrees, Dimensions<3>, Dimensions<3>>{}) == stdcompat::partial_ordering::less);

  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}), std::tuple<Axis, angle::Radians>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::tuple<Axis, angle::Radians>{}, make_pattern(Dimensions{1}, angle::Radians{}, Dimensions{1})) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}), std::tuple<Axis, angle::Radians, Axis>{}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(angle::Radians{}, std::vector{angle::Radians{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(std::vector{inclination::Radians{}}, inclination::Radians{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(angle::Radians{}, std::vector{inclination::Radians{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare(angle::Radians{}, inclination::Radians{}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare(std::vector{Polar<Distance, angle::Radians>{}}, Dimensions<5>{}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare(Polar<Distance, angle::Radians>{}, std::vector{Dimensions<5>{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare(std::vector{Spherical<Distance, inclination::Radians, angle::Radians>{}}, Dimensions<5>{}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare(Spherical<Distance, inclination::Radians, angle::Radians>{}, std::vector{Dimensions<5>{}}) == stdcompat::partial_ordering::unordered);

  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}, Distance{}), make_pattern<float> (Dimensions{1}, angle::Radians{}, Distance{})) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}, Distance{}), make_pattern<long double> (Dimensions{1}, angle::Radians{}, Distance{})) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}, Distance{}), make_pattern<float> (Dimensions{1}, angle::Radians{}, Distance{}, Dimensions{1})) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare(make_pattern(Dimensions{1}, angle::Radians{}, Distance{}, Dimensions{1}), make_pattern<float> (Dimensions{1}, angle::Radians{}, Distance{})) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare(make_pattern(Dimensions{2}, angle::Radians{}, Distance{}, Dimensions{1}), make_pattern<float> (Dimensions{1}, angle::Radians{}, Distance{})) == stdcompat::partial_ordering::unordered);
}
