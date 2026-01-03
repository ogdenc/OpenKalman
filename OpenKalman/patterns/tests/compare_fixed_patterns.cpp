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
 * \brief Tests for \ref patterns::pattern equivalence
 */

#include "collections/tests/tests.hpp"
#include "patterns/descriptors/Dimensions.hpp"
#include "patterns/descriptors/Distance.hpp"
#include "patterns/descriptors/Angle.hpp"
#include "patterns/descriptors/Inclination.hpp"
#include "patterns/descriptors/Polar.hpp"
#include "patterns/descriptors/Spherical.hpp"
#include "patterns/descriptors/Any.hpp"

using namespace OpenKalman;
using namespace OpenKalman::patterns;

#include "patterns/functions/compare_three_way.hpp"
#include "patterns/functions/compare.hpp"

TEST(patterns, compare_three_way_fixed_pattern)
{
  static_assert(values::to_value_type(compare_three_way(Dimensions<3>{}, std::tuple<Axis, Axis, Axis>{})) == stdex::partial_ordering::equivalent);
  static_assert(compare_three_way(Dimensions<3>{}, std::tuple<Axis, Axis, Axis, Axis>{}) == stdex::partial_ordering::less);
  static_assert(compare_three_way(Dimensions<4>{}, std::tuple<Axis, Axis, Axis, Axis>{}) == stdex::partial_ordering::equivalent);
  static_assert(compare_three_way(Dimensions<4>{}, std::tuple<Axis, Axis, Axis>{}) == stdex::partial_ordering::greater);

  static_assert(compare_three_way(std::tuple{Axis{}}, Dimensions<1>{}) == stdex::partial_ordering::equivalent);
  static_assert(compare_three_way(Dimensions<1>{}, std::tuple{Axis{}}) == stdex::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple{Axis{}}, Dimensions<1>{}) == stdex::partial_ordering::equivalent);
  static_assert(compare_three_way(std::array{Axis{}, Axis{}, Axis{}}, Dimensions<3>{}) == stdex::partial_ordering::equivalent);
  static_assert(compare_three_way(std::array{Axis{}, Axis{}}, Dimensions<3>{}) == stdex::partial_ordering::less);
  static_assert(compare_three_way(std::tuple<Axis, Axis, Axis>{}, Dimensions<3>{}) == stdex::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple<Axis, Axis, Axis, Axis>{}, Dimensions<3>{}) == stdex::partial_ordering::greater);
  static_assert(compare_three_way(std::tuple<Axis, Axis, Axis, Axis>{}, Dimensions<4>{}) == stdex::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple<Axis, Axis, Axis>{}, Dimensions<4>{}) == stdex::partial_ordering::less);

  static_assert(compare_three_way(std::tuple<Axis, Axis>{}, std::array{Dimensions<4>{}}) == stdex::partial_ordering::less);
  static_assert(compare_three_way(std::array{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdex::partial_ordering::greater);
  static_assert(compare_three_way(std::array{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdex::partial_ordering::greater);
  static_assert(compare_three_way(std::tuple<>{}, std::array<Inclination<>, 0>{}) == stdex::partial_ordering::equivalent);

  static_assert(compare_three_way(std::tuple<Axis, Axis>{}, std::tuple{Dimensions<4>{}}) == stdex::partial_ordering::less);
  static_assert(compare_three_way(std::tuple{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdex::partial_ordering::greater);
  static_assert(compare_three_way(std::tuple{Dimensions<4>{}}, std::tuple<Axis, Axis>{}) == stdex::partial_ordering::greater);

  static_assert(compare_three_way(std::tuple{}, std::tuple{}) == stdex::partial_ordering::equivalent);
  static_assert(compare_three_way(stdex::ranges::views::empty<Dimensions<1>>, std::tuple{}) == stdex::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple{}, stdex::ranges::views::empty<Dimensions<1>>) == stdex::partial_ordering::equivalent);

  static_assert(compare_three_way(std::tuple<Axis, angle::Radians>{}, std::tuple<Axis, angle::Radians>{}) == stdex::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple<>{}, std::tuple<Axis, angle::Radians>{}) == stdex::partial_ordering::less);
  static_assert(compare_three_way(std::array<Polar<>, 0>{}, std::tuple<Axis, angle::Radians>{}) == stdex::partial_ordering::less);
  static_assert(compare_three_way(std::tuple<Axis>{}, std::tuple<Axis, angle::Radians>{}) == stdex::partial_ordering::less);
  static_assert(compare_three_way(std::tuple<Axis, angle::Radians>{}, std::tuple<Axis, angle::Radians, Axis>{}) == stdex::partial_ordering::less);
  static_assert(compare_three_way(std::tuple<Axis, angle::Radians, Axis>{}, std::tuple<Axis, angle::Radians>{}) == stdex::partial_ordering::greater);
  static_assert(compare_three_way(std::tuple<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{}, std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}) == stdex::partial_ordering::equivalent);
  static_assert(compare_three_way(std::tuple(Axis{}, Dimensions<3>{}, angle::Radians{}, Dimensions<5>{}), std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}) == stdex::partial_ordering::greater);
  static_assert(compare_three_way(std::tuple(Axis{}, Dimensions<3>{}, angle::Radians{}, Axis{}, Dimensions<2>{}), std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<4>>{}) == stdex::partial_ordering::less);

  static_assert(compare<stdex::is_eq>(std::tuple<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{}, std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}));
  static_assert(compare<stdex::is_gteq>(std::tuple<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{}, std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}));
  static_assert(compare<stdex::is_lteq>(std::tuple<Axis, Dimensions<3>, angle::Radians, Dimensions<4>>{}, std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}));
  static_assert(compare<stdex::is_gt>(std::tuple(Axis{}, Dimensions<3>{}, angle::Radians{}, Dimensions<5>{}), std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<3>, Axis>{}));
  static_assert(compare<stdex::is_lt>(std::tuple(Axis{}, Dimensions<3>{}, angle::Radians{}, Axis{}, Dimensions<2>{}), std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<4>>{}));
  static_assert(compare<stdex::is_neq>(std::tuple(Axis{}, Dimensions<3>{}, angle::Radians{}, Axis{}, Dimensions<2>{}), std::tuple<Dimensions<2>, Dimensions<2>, angle::Radians, Dimensions<4>>{}));
}
