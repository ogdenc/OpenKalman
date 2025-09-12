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

TEST(coordinates, compare_three_way_fixed_pattern)
{
  static_assert(values::to_value_type(compare_three_way(Dimensions<3>{}, std::tuple<Axis, Axis, Axis>{})) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(Dimensions<3>{}, std::tuple<Axis, Axis, Axis, Axis>{}) == stdcompat::partial_ordering::less);
  static_assert(compare_three_way(Dimensions<4>{}, std::tuple<Axis, Axis, Axis, Axis>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(Dimensions<4>{}, std::tuple<Axis, Axis, Axis>{}) == stdcompat::partial_ordering::greater);

  static_assert(compare_three_way(std::tuple{Axis{}}, Dimensions<1>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(Dimensions<1>{}, std::tuple{Axis{}}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple{Axis{}}, Dimensions<1>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(std::array{Axis{}, Axis{}, Axis{}}, Dimensions<3>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(std::array{Axis{}, Axis{}}, Dimensions<3>{}) == stdcompat::partial_ordering::less);
  static_assert(compare_three_way(std::tuple<Axis, Axis, Axis>{}, Dimensions<3>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple<Axis, Axis, Axis, Axis>{}, Dimensions<3>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare_three_way(std::tuple<Axis, Axis, Axis, Axis>{}, Dimensions<4>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple<Axis, Axis, Axis>{}, Dimensions<4>{}) == stdcompat::partial_ordering::less);

  static_assert(compare_three_way(std::tuple<Axis, Axis>{}, std::array{Dimensions<4>{}}) == stdcompat::partial_ordering::less);
  static_assert(compare_three_way(std::array{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare_three_way(std::array{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare_three_way(std::tuple<>{}, std::array<Inclination<>, 0>{}) == stdcompat::partial_ordering::equivalent);

  static_assert(compare_three_way(std::tuple<Axis, Axis>{}, std::tuple{Dimensions<4>{}}) == stdcompat::partial_ordering::less);
  static_assert(compare_three_way(std::tuple{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare_three_way(std::tuple{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdcompat::partial_ordering::greater);

  static_assert(compare_three_way(std::tuple{}, std::tuple{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(stdcompat::ranges::views::empty<Dimensions<1>>, std::tuple{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple{}, stdcompat::ranges::views::empty<Dimensions<1>>) == stdcompat::partial_ordering::equivalent);

  static_assert(compare_three_way(std::tuple<Axis, angle::Radians>{}, std::tuple<Axis, angle::Radians>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple<>{}, std::tuple<Axis, angle::Radians>{}) == stdcompat::partial_ordering::less);
  static_assert(compare_three_way(std::array<Polar<>, 0>{}, std::tuple<Axis, angle::Radians>{}) == stdcompat::partial_ordering::less);
  static_assert(compare_three_way(std::tuple<Axis>{}, std::tuple<Axis, angle::Radians>{}) == stdcompat::partial_ordering::less);
  static_assert(compare_three_way(std::tuple<Axis, angle::Radians>{}, std::tuple<Axis, angle::Radians, Axis>{}) == stdcompat::partial_ordering::less);
  static_assert(compare_three_way(std::tuple<Axis, angle::Radians, Axis>{}, std::tuple<Axis, angle::Radians>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare_three_way(std::tuple<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{}, std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}) == stdcompat::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple(Axis{}, Dimensions<3>{}, angle::Radians{}, Dimensions<5>{}), std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}) == stdcompat::partial_ordering::greater);
  static_assert(compare_three_way(std::tuple(Axis{}, Dimensions<3>{}, angle::Radians{}, Axis{}, Dimensions<2>{}), std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<4>>{}) == stdcompat::partial_ordering::less);

  static_assert(compare<stdcompat::is_eq>(std::tuple<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{}, std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}));
  static_assert(compare<stdcompat::is_gteq>(std::tuple<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{}, std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}));
  static_assert(compare<stdcompat::is_lteq>(std::tuple<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{}, std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}));
  static_assert(compare<stdcompat::is_gt>(std::tuple(Axis{}, Dimensions<3>{}, angle::Radians{}, Dimensions<5>{}), std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}));
  static_assert(compare<stdcompat::is_lt>(std::tuple(Axis{}, Dimensions<3>{}, angle::Radians{}, Axis{}, Dimensions<2>{}), std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<4>>{}));
  static_assert(compare<stdcompat::is_neq>(std::tuple(Axis{}, Dimensions<3>{}, angle::Radians{}, Axis{}, Dimensions<2>{}), std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<4>>{}));
}
