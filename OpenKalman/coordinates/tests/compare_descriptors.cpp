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
 * \brief Tests for \ref coordinates::descriptor equivalence
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

#include "coordinates/functions/compare.hpp"

TEST(coordinates, compare_descriptors)
{
  static_assert(compare(std::integral_constant<std::size_t, 2>{}, 2u, stdcompat::is_eq));
  static_assert(compare(std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 2>{}, stdcompat::is_eq));
  static_assert(compare(std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 3>{}, stdcompat::is_neq));
  static_assert(compare(std::integral_constant<std::size_t, 2>{}, Dimensions<2>{}, stdcompat::is_eq));
  static_assert(compare(std::integral_constant<std::size_t, 2>{}, Dimensions{2}, stdcompat::is_eq));

  static_assert(compare(Dimensions<3>{}, Dimensions<3>{}, stdcompat::is_eq));
  static_assert(compare(Dimensions<3>{}, Dimensions<3>{}, stdcompat::is_lteq));
  static_assert(compare(Dimensions<3>{}, Dimensions<3>{}, stdcompat::is_gteq));
  static_assert(compare(Dimensions<3>{}, Dimensions<4>{}, stdcompat::is_neq));
  static_assert(compare(Dimensions<3>{}, Dimensions<4>{}, stdcompat::is_lt));
  static_assert(compare(Dimensions<3>{}, Dimensions<4>{}, stdcompat::is_lteq));
  static_assert(compare(Dimensions<4>{}, Dimensions<3>{}, stdcompat::is_gt));
  static_assert(compare(Dimensions<4>{}, Dimensions<3>{}, stdcompat::is_gteq));

  static_assert(compare(angle::Degrees{}, angle::Degrees{}, stdcompat::is_eq));
  static_assert(compare(angle::Radians{}, angle::Degrees{}, stdcompat::is_neq));
  static_assert(compare(coordinates::internal::get_descriptor_hash_code(angle::Degrees{}), coordinates::internal::get_descriptor_hash_code(Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>{}), stdcompat::is_eq));
  static_assert(compare(angle::Degrees{}, Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>{}, stdcompat::is_eq));
  static_assert(compare(angle::Degrees{}, Angle<std::integral_constant<int, -180>, std::integral_constant<std::size_t, 180>>{}, stdcompat::is_eq));
  static_assert(compare(angle::Degrees{}, Angle<values::fixed_value<long double, -180>, values::fixed_value<double, 180>>{}, stdcompat::is_eq));
  static_assert(compare(angle::PositiveDegrees{}, Angle<std::integral_constant<std::size_t, 0>, std::integral_constant<int, 360>>{}, stdcompat::is_eq));
  static_assert(compare(angle::Degrees{}, angle::PositiveDegrees{}, stdcompat::is_neq));
  static_assert(compare(angle::Radians{}, Angle<values::fixed_minus_pi<double>, values::fixed_pi<float>>{}, stdcompat::is_eq));
  static_assert(compare(angle::Radians{}, angle::Radians{}, stdcompat::is_eq));
  static_assert(compare(angle::Radians{}, inclination::Radians{}, stdcompat::is_neq));

  static_assert(compare(inclination::Degrees{}, inclination::Degrees{}, stdcompat::is_eq));
  static_assert(compare(inclination::Radians{}, inclination::Radians{}, stdcompat::is_eq));
  static_assert(compare(inclination::Radians{}, inclination::Degrees{}, stdcompat::is_neq));
  static_assert(compare(inclination::Degrees{}, Inclination<std::integral_constant<int, 180>>{}, stdcompat::is_eq));
  static_assert(compare(inclination::Degrees{}, Inclination<values::fixed_value<double, 180>>{}, stdcompat::is_eq));
  static_assert(compare(inclination::Radians{}, Inclination<values::fixed_pi<double>>{}, stdcompat::is_eq));
  static_assert(compare(inclination::Radians{}, Inclination<values::fixed_pi<float>>{}, stdcompat::is_eq));

  static_assert(compare(Polar<Distance, angle::Degrees>{}, Polar<Distance, angle::PositiveDegrees>{}, stdcompat::is_neq));
  static_assert(compare(Polar<angle::Radians, Distance>{}, Polar<angle::PositiveRadians, Distance>{}, stdcompat::is_neq));
  static_assert(compare(Polar<Distance, angle::Radians>{}, Dimensions<5>{}, stdcompat::is_neq));
  static_assert(not compare(Polar<Distance, angle::Radians>{}, Dimensions<5>{}, stdcompat::is_lt));

  static_assert(compare(Spherical<Distance, inclination::Radians, angle::Degrees>{}, Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{}, stdcompat::is_neq));
  static_assert(compare(Spherical<Distance, inclination::Degrees, angle::Degrees>{}, Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{}, stdcompat::is_neq));
  static_assert(compare(Spherical<Distance, inclination::Radians, angle::Radians>{}, Dimensions<5>{}, stdcompat::is_neq));
  static_assert(not compare(Spherical<Distance, inclination::Radians, angle::Radians>{}, Dimensions<5>{}, stdcompat::is_lt));

  static_assert(compare(Dimensions{0}, Dimensions{0}, stdcompat::is_eq));
  static_assert(compare(Dimensions{3}, Dimensions{3}, stdcompat::is_eq));
  static_assert(compare(Dimensions{3}, Dimensions{3}, stdcompat::is_eq));
  static_assert(compare(Dimensions{3}, Dimensions{3}, stdcompat::is_lteq));
  static_assert(compare(Dimensions{3}, Dimensions{3}, stdcompat::is_gteq));
  static_assert(compare(Dimensions{3}, Dimensions{4}, stdcompat::is_neq));
  static_assert(compare(Dimensions{3}, Dimensions{4}, stdcompat::is_lt));
  static_assert(compare(Dimensions{3}, Dimensions{4}, stdcompat::is_lteq));
  static_assert(compare(Dimensions{4}, Dimensions{3}, stdcompat::is_gt));
  static_assert(compare(Dimensions{4}, Dimensions{3}, stdcompat::is_gteq));

  static_assert(compare(Dimensions{3}, Dimensions<3>{}, stdcompat::is_eq));
  static_assert(compare(Dimensions{3}, Dimensions<3>{}, stdcompat::is_lteq));
  static_assert(compare(Dimensions{3}, Dimensions<3>{}, stdcompat::is_gteq));
  static_assert(compare(Dimensions{3}, Dimensions<4>{}, stdcompat::is_neq));
  static_assert(compare(Dimensions{3}, Dimensions<4>{}, stdcompat::is_lt));
  static_assert(compare(Dimensions{3}, Dimensions<4>{}, stdcompat::is_lteq));
  static_assert(compare(Dimensions{4}, Dimensions<3>{}, stdcompat::is_gt));
  static_assert(compare(Dimensions{4}, Dimensions<3>{}, stdcompat::is_gteq));

  static_assert(compare(Polar<>{}, Dimensions{5}, stdcompat::is_neq));
  static_assert(compare(Dimensions{5}, Polar<>{}, stdcompat::is_neq));
  static_assert(not compare(Polar<>{}, Dimensions{5}, stdcompat::is_lt));
  static_assert(compare(Spherical<>{}, Dimensions{5}, stdcompat::is_neq));

  EXPECT_TRUE(compare(Any{angle::Radians{}}, Any{angle::Radians{}}));
  EXPECT_TRUE(compare(Any{angle::Radians{}}, Any{angle::Degrees{}}, stdcompat::is_neq));
  EXPECT_TRUE(compare(Any{inclination::Radians{}}, Any{inclination::Radians{}}));
  EXPECT_TRUE(compare(Any{inclination::Radians{}}, Any{inclination::Degrees{}}, stdcompat::is_neq));
  EXPECT_TRUE(compare(Any{angle::Radians{}}, Any{inclination::Radians{}}, stdcompat::is_neq));
}

