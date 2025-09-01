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

TEST(coordinates, compare_dynamic_pattern_fixed_size)
{
  static_assert(compare_three_way(Dimensions<0>{}, std::tuple<>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(Dimensions{0}, std::tuple<>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple<>{}, Dimensions{0}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(Dimensions<2>{}, std::tuple<>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare_three_way(std::tuple<>{}, std::tuple<Axis>{}) == stdcompat::partial_ordering::less);

  static_assert(compare_three_way(Dimensions{3}, std::tuple<Axis, Axis, Axis>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(Dimensions{4}, std::tuple<Axis, Dimensions<2>, Axis>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(Dimensions{3}, std::tuple<Axis, Dimensions<2>>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(Dimensions{3}, std::tuple<Axis, Dimensions<2>, Axis>{}) == stdcompat::partial_ordering::less);
  static_assert(compare_three_way(Dimensions{3}, std::tuple<Axis, Dimensions<2>, Axis>{}) == stdcompat::partial_ordering::less);
  static_assert(compare_three_way(Dimensions{3}, std::tuple<Axis, Axis, Axis, Axis>{}) == stdcompat::partial_ordering::less);

  static_assert(compare_three_way(std::tuple<Axis, Axis, Axis>{}, Dimensions{3}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple<Axis, Dimensions<2>, Axis>{}, Dimensions{4}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple<Axis, Dimensions<2>>{}, Dimensions{3}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple<Axis, Dimensions<2>, Axis>{}, Dimensions{3}) == stdcompat::partial_ordering::greater);
  static_assert(compare_three_way(std::tuple<Axis, Axis, Axis, Axis>{}, Dimensions{3}) == stdcompat::partial_ordering::greater);

  EXPECT_TRUE(compare_three_way(Any{Axis{}}, Any{Axis{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Any{Dimensions<3>{}}, Any{Dimensions<3>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Any{Dimensions<4>{}}, Any{Dimensions<3>{}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(Any{Dimensions{4}}, Any{Dimensions{4}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Any{Dimensions{6}}, Any{Dimensions{5}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(Any{Dimensions<2>{}}, Dimensions<2>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Any{Dimensions<2>{}}, Dimensions{2}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Any{Dimensions<2>{}}, Dimensions{3}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(Dimensions{3}, Any{Dimensions<3>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Dimensions{2}, Any{Dimensions<3>{}}) == stdcompat::partial_ordering::less);

  EXPECT_TRUE(compare_three_way(Any{angle::Radians{}}, Any{angle::Radians{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(Any{angle::Radians{}}, Any{angle::Degrees{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(Any{angle::Radians{}}, angle::Radians{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(angle::Radians{}, Any{angle::Radians{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(angle::Degrees{}, Any{angle::Radians{}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(angle::Radians{}, inclination::Radians{}) == stdcompat::partial_ordering::unordered);

  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}, Dimensions{1}}, std::tuple{Dimensions{1}, angle::Radians{}, Dimensions{1}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}, Dimensions{1}}, std::tuple{Dimensions{1}, angle::Radians{}, Dimensions{1}, Dimensions{1}}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}, Dimensions{1}, Dimensions{1}}, std::tuple{Dimensions{1}, angle::Radians{}, Dimensions{1}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}, Dimensions{1}}, std::tuple{Dimensions{1}, angle::Radians{}, Dimensions{1}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{3}, Dimensions{2}, angle::Radians{}, Dimensions{5}}, std::tuple{Dimensions{2}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}, Dimensions{1}}, std::tuple{Dimensions{1}, angle::Radians{}, Dimensions{1}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{3}, Dimensions<2>{}, angle::Radians{}, Dimensions<4>{}}, std::tuple{Dimensions{2}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<3>{}}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{3}, Dimensions<2>{}, angle::Radians{}, Dimensions<5>{}}, std::tuple{Dimensions{2}, Dimensions<3>{}, angle::Radians{}, Dimensions<2>{}, Dimensions<2>{}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector<Polar<>>{}, std::vector<Polar<>>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Polar{}, Dimensions{1}}, std::tuple{Polar{}, Dimensions{1}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Polar{}, Dimensions{1}}, std::vector<Polar<>>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::vector{Spherical<Distance, angle::Radians, inclination::Radians>{}}, std::vector{Spherical<Distance, angle::Radians, inclination::Radians>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}, angle::Radians{}}, std::tuple{Dimensions{1}, angle::Radians{}, Dimensions{1}}) == stdcompat::partial_ordering::unordered);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}}, std::tuple{Polar<Distance, angle::Radians>{}}) == stdcompat::partial_ordering::unordered);

  EXPECT_TRUE(compare_three_way(std::tuple<Axis, Axis>{}, std::tuple{Dimensions{4}}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{4}}, std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{2}}, std::tuple<Axis, Axis, Axis, Axis>{}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions<3>{}, angle::Degrees{}}, Dimensions<3>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions<3>{}, angle::Degrees{}, Dimensions<5>{}}, std::tuple<Dimensions<3>, angle::Degrees, Dimensions<5>>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple<Dimensions<3>, angle::Degrees, Dimensions<5>>{}, std::tuple{Dimensions<3>{}, angle::Degrees{}, Dimensions<5>{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions<3>{}, angle::Degrees{}}, std::tuple<Dimensions<3>, angle::Degrees>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}}, std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}}, std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}}, std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<4>>{}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}}, std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<2>>{}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, Dimensions<3>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}}, std::tuple<Dimensions<2>, Axis, angle::Degrees, Dimensions<2>, Dimensions<3>>{}) == stdcompat::partial_ordering::unordered);
  EXPECT_FALSE(compare_three_way(std::tuple{Dimensions{1}, Dimensions<2>{}, angle::Degrees{}, Dimensions<3>{}, Dimensions<2>{}}, std::tuple<Dimensions<4>, angle::Degrees, Dimensions<3>, Dimensions<3>>{}) == stdcompat::partial_ordering::less);

  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}}, std::tuple<Axis, angle::Radians>{}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple<Axis, angle::Radians>{}, std::tuple{Dimensions{1}, angle::Radians{}, Dimensions{1}}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}}, std::tuple<Axis, angle::Radians, Axis>{}) == stdcompat::partial_ordering::less);

  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}, Distance{}}, std::tuple{Dimensions{1}, angle::Radians{}, Distance{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}, Distance{}}, std::tuple{Dimensions{1}, angle::Radians{}, Distance{}}) == stdcompat::partial_ordering::equivalent);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}, Distance{}}, std::tuple{Dimensions{1}, angle::Radians{}, Distance{}, Dimensions{1}}) == stdcompat::partial_ordering::less);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{1}, angle::Radians{}, Distance{}, Dimensions{1}}, std::tuple{Dimensions{1}, angle::Radians{}, Distance{}}) == stdcompat::partial_ordering::greater);
  EXPECT_TRUE(compare_three_way(std::tuple{Dimensions{2}, angle::Radians{}, Distance{}, Dimensions{1}}, std::tuple{Dimensions{1}, angle::Radians{}, Distance{}}) == stdcompat::partial_ordering::unordered);
}

