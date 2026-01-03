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
 * \brief Tests for \ref patterns::descriptor equivalence
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

#include "patterns/functions/compare.hpp"

TEST(patterns, compare_descriptors)
{
  static_assert(compare<stdex::is_eq>(std::integral_constant<std::size_t, 2>{}, 2u));
  static_assert(compare<stdex::is_eq>(std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 2>{}));
  static_assert(compare<stdex::is_neq>(std::integral_constant<std::size_t, 2>{}, std::integral_constant<std::size_t, 3>{}));
  static_assert(compare<stdex::is_eq>(std::integral_constant<std::size_t, 2>{}, Dimensions<2>{}));
  static_assert(compare<stdex::is_eq>(std::integral_constant<std::size_t, 2>{}, Dimensions{2}));

  static_assert(compare<stdex::is_eq>(Dimensions<3>{}, Dimensions<3>{}));
  static_assert(compare<stdex::is_lteq>(Dimensions<3>{}, Dimensions<3>{}));
  static_assert(compare<stdex::is_gteq>(Dimensions<3>{}, Dimensions<3>{}));
  static_assert(compare<stdex::is_neq>(Dimensions<3>{}, Dimensions<4>{}));
  static_assert(compare<stdex::is_lt>(Dimensions<3>{}, Dimensions<4>{}));
  static_assert(compare<stdex::is_lteq>(Dimensions<3>{}, Dimensions<4>{}));
  static_assert(compare<stdex::is_gt>(Dimensions<4>{}, Dimensions<3>{}));
  static_assert(compare<stdex::is_gteq>(Dimensions<4>{}, Dimensions<3>{}));

  static_assert(compare<stdex::is_eq>(angle::Degrees{}, angle::Degrees{}));
  static_assert(compare<stdex::is_neq>(angle::Radians{}, angle::Degrees{}));
  static_assert(compare<stdex::is_eq>(patterns::internal::get_descriptor_hash_code(angle::Degrees{}), patterns::internal::get_descriptor_hash_code(Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>{})));
  static_assert(compare<stdex::is_eq>(angle::Degrees{}, Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>{}));
  static_assert(compare<stdex::is_eq>(angle::Degrees{}, Angle<std::integral_constant<int, -180>, std::integral_constant<std::size_t, 180>>{}));
  static_assert(compare<stdex::is_eq>(angle::Degrees{}, Angle<values::fixed_value<long double, -180>, values::fixed_value<double, 180>>{}));
  static_assert(compare<stdex::is_eq>(angle::PositiveDegrees{}, Angle<std::integral_constant<std::size_t, 0>, std::integral_constant<int, 360>>{}));
  static_assert(compare<stdex::is_neq>(angle::Degrees{}, angle::PositiveDegrees{}));
  static_assert(compare<stdex::is_eq>(angle::Radians{}, Angle<values::fixed_minus_pi<double>, values::fixed_pi<float>>{}));
  static_assert(compare<stdex::is_eq>(angle::Radians{}, angle::Radians{}));
  static_assert(compare<stdex::is_neq>(angle::Radians{}, inclination::Radians{}));

  static_assert(compare<stdex::is_eq>(inclination::Degrees{}, inclination::Degrees{}));
  static_assert(compare<stdex::is_eq>(inclination::Radians{}, inclination::Radians{}));
  static_assert(compare<stdex::is_neq>(inclination::Radians{}, inclination::Degrees{}));
  static_assert(compare<stdex::is_eq>(inclination::Degrees{}, Inclination<std::integral_constant<int, 180>>{}));
  static_assert(compare<stdex::is_eq>(inclination::Degrees{}, Inclination<values::fixed_value<double, 180>>{}));
  static_assert(compare<stdex::is_eq>(inclination::Radians{}, Inclination<values::fixed_pi<double>>{}));
  static_assert(compare<stdex::is_eq>(inclination::Radians{}, Inclination<values::fixed_pi<float>>{}));

  static_assert(compare<stdex::is_neq>(Polar<Distance, angle::Degrees>{}, Polar<Distance, angle::PositiveDegrees>{}));
  static_assert(compare<stdex::is_neq>(Polar<angle::Radians, Distance>{}, Polar<angle::PositiveRadians, Distance>{}));
  static_assert(compare<stdex::is_neq>(Polar<Distance, angle::Radians>{}, Dimensions<5>{}));
  static_assert(not compare<stdex::is_lt>(Polar<Distance, angle::Radians>{}, Dimensions<5>{}));

  static_assert(compare<stdex::is_neq>(Spherical<Distance, inclination::Radians, angle::Degrees>{}, Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{}));
  static_assert(compare<stdex::is_neq>(Spherical<Distance, inclination::Degrees, angle::Degrees>{}, Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{}));
  static_assert(compare<stdex::is_neq>(Spherical<Distance, inclination::Radians, angle::Radians>{}, Dimensions<5>{}));
  static_assert(not compare<stdex::is_lt>(Spherical<Distance, inclination::Radians, angle::Radians>{}, Dimensions<5>{}));

  static_assert(compare<stdex::is_eq>(Dimensions{0}, Dimensions{0}));
  static_assert(compare<stdex::is_eq>(Dimensions{3}, Dimensions{3}));
  static_assert(compare<stdex::is_eq>(Dimensions{3}, Dimensions{3}));
  static_assert(compare<stdex::is_lteq>(Dimensions{3}, Dimensions{3}));
  static_assert(compare<stdex::is_gteq>(Dimensions{3}, Dimensions{3}));
  static_assert(compare<stdex::is_neq>(Dimensions{3}, Dimensions{4}));
  static_assert(compare<stdex::is_lt>(Dimensions{3}, Dimensions{4}));
  static_assert(compare<stdex::is_lteq>(Dimensions{3}, Dimensions{4}));
  static_assert(compare<stdex::is_gt>(Dimensions{4}, Dimensions{3}));
  static_assert(compare<stdex::is_gteq>(Dimensions{4}, Dimensions{3}));

  static_assert(compare<stdex::is_eq>(Dimensions{3}, Dimensions<3>{}));
  static_assert(compare<stdex::is_lteq>(Dimensions{3}, Dimensions<3>{}));
  static_assert(compare<stdex::is_gteq>(Dimensions{3}, Dimensions<3>{}));
  static_assert(compare<stdex::is_neq>(Dimensions{3}, Dimensions<4>{}));
  static_assert(compare<stdex::is_lt>(Dimensions{3}, Dimensions<4>{}));
  static_assert(compare<stdex::is_lteq>(Dimensions{3}, Dimensions<4>{}));
  static_assert(compare<stdex::is_gt>(Dimensions{4}, Dimensions<3>{}));
  static_assert(compare<stdex::is_gteq>(Dimensions{4}, Dimensions<3>{}));

  static_assert(compare<stdex::is_neq>(Polar<>{}, Dimensions{5}));
  static_assert(compare<stdex::is_neq>(Dimensions{5}, Polar<>{}));
  static_assert(not compare<stdex::is_lt>(Polar<>{}, Dimensions{5}));
  static_assert(compare<stdex::is_neq>(Spherical<>{}, Dimensions{5}));

  EXPECT_TRUE(compare(Any{angle::Radians{}}, Any{angle::Radians{}}));
  EXPECT_TRUE(compare<stdex::is_neq>(Any{angle::Radians{}}, Any{angle::Degrees{}}));
  EXPECT_TRUE(compare(Any{inclination::Radians{}}, Any{inclination::Radians{}}));
  EXPECT_TRUE(compare<stdex::is_neq>(Any{inclination::Radians{}}, Any{inclination::Degrees{}}));
  EXPECT_TRUE(compare<stdex::is_neq>(Any{angle::Radians{}}, Any{inclination::Radians{}}));
}

