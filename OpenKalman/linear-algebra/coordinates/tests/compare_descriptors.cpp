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

#include "basics/tests/tests.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/descriptors/Distance.hpp"
#include "linear-algebra/coordinates/descriptors/Angle.hpp"
#include "linear-algebra/coordinates/descriptors/Inclination.hpp"
#include "linear-algebra/coordinates/descriptors/Polar.hpp"
#include "linear-algebra/coordinates/descriptors/Spherical.hpp"
#include "linear-algebra/coordinates/descriptors/Any.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;

#include "linear-algebra/coordinates/functions/comparison-operators.hpp"

TEST(coordinates, compare_descriptors)
{
  static_assert(std::integral_constant<std::size_t, 2>{} == 2u);
  static_assert(std::integral_constant<std::size_t, 2>{} == std::integral_constant<std::size_t, 2>{});
  static_assert(std::integral_constant<std::size_t, 2>{} != std::integral_constant<std::size_t, 3>{});
  static_assert(std::integral_constant<std::size_t, 2>{} == Dimensions<2>{});
  static_assert(std::integral_constant<std::size_t, 2>{} == Dimensions{2});

  static_assert(Dimensions<3>{} == Dimensions<3>{});
  static_assert(Dimensions<3>{} <= Dimensions<3>{});
  static_assert(Dimensions<3>{} >= Dimensions<3>{});
  static_assert((Dimensions<3>{} != Dimensions<4>{}));
  static_assert((Dimensions<3>{} < Dimensions<4>{}));
  static_assert((Dimensions<3>{} <= Dimensions<4>{}));
  static_assert((Dimensions<4>{} > Dimensions<3>{}));
  static_assert((Dimensions<4>{} >= Dimensions<3>{}));

  static_assert(angle::Degrees{} == angle::Degrees{});
  static_assert(coordinates::internal::get_hash_code(angle::Degrees{}) == coordinates::internal::get_hash_code(Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>{}));
  static_assert(angle::Degrees{} == Angle<std::integral_constant<int, -180>, std::integral_constant<int, 180>>{});
  static_assert(angle::Degrees{} == Angle<std::integral_constant<int, -180>, std::integral_constant<std::size_t, 180>>{});
  static_assert(angle::Degrees{} == Angle<values::Fixed<long double, -180>, values::Fixed<double, 180>>{});
  static_assert(angle::PositiveDegrees{} == Angle<std::integral_constant<std::size_t, 0>, std::integral_constant<int, 360>>{});
  static_assert(angle::Degrees{} != angle::PositiveDegrees{});
  static_assert(angle::Radians{} == Angle<values::fixed_minus_pi<double>, values::fixed_pi<float>>{});
  static_assert(angle::Radians{} == angle::Radians{});
  static_assert(angle::Radians{} != inclination::Radians{});
  static_assert(not (angle::Radians{} < inclination::Radians{}));

  static_assert(inclination::Degrees{} == inclination::Degrees{});
  static_assert(inclination::Radians{} == inclination::Radians{});
  static_assert(inclination::Radians{} != inclination::Degrees{});
  static_assert(inclination::Degrees{} == Inclination<std::integral_constant<int, -90>, std::integral_constant<int, 90>>{});
  static_assert(inclination::Degrees{} == Inclination<std::integral_constant<int, -90>, std::integral_constant<std::size_t, 90>>{});
  static_assert(inclination::Degrees{} == Inclination<values::Fixed<long double, -90>, values::Fixed<double, 90>>{});
  static_assert(inclination::Radians{} == Inclination<values::fixed_minus_half_pi<double>, values::fixed_half_pi<float>>{});

  static_assert(Polar<Distance, angle::Degrees>{} != Polar<Distance, angle::PositiveDegrees>{});
  static_assert(Polar<angle::Radians, Distance>{} != Polar<angle::PositiveRadians, Distance>{});
  static_assert((Polar<Distance, angle::Radians>{} != Dimensions<5>{}));
  static_assert(not (Polar<Distance, angle::Radians>{} < Dimensions<5>{}));

  static_assert(Spherical<Distance, inclination::Radians, angle::Degrees>{} != Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{});
  static_assert(Spherical<Distance, inclination::Degrees, angle::Degrees>{} != Spherical<Distance, inclination::Radians, angle::PositiveDegrees>{});
  static_assert((Spherical<Distance, inclination::Radians, angle::Radians>{} != Dimensions<5>{}));
  static_assert(not (Spherical<Distance, inclination::Radians, angle::Radians>{} < Dimensions<5>{}));

  static_assert(Dimensions{0} == Dimensions{0});
  static_assert(Dimensions{3} == Dimensions{3});
  static_assert(Dimensions{3} == Dimensions{3});
  static_assert(Dimensions{3} <= Dimensions{3});
  static_assert(Dimensions{3} >= Dimensions{3});
  static_assert(Dimensions{3} != Dimensions{4});
  static_assert(Dimensions{3} < Dimensions{4});
  static_assert(Dimensions{3} <= Dimensions{4});
  static_assert(Dimensions{4} > Dimensions{3});
  static_assert(Dimensions{4} >= Dimensions{3});

  static_assert(Dimensions{3} == Dimensions<3>{});
  static_assert(Dimensions{3} <= Dimensions<3>{});
  static_assert(Dimensions{3} >= Dimensions<3>{});
  static_assert(Dimensions{3} != Dimensions<4>{});
  static_assert(Dimensions{3} < Dimensions<4>{});
  static_assert(Dimensions{3} <= Dimensions<4>{});
  static_assert(Dimensions{4} > Dimensions<3>{});
  static_assert(Dimensions{4} >= Dimensions<3>{});

  static_assert(Polar<>{} != Dimensions{5});
  static_assert(Dimensions{5} != Polar<>{});
  static_assert(not (Polar<>{} < Dimensions{5}));
  static_assert(Spherical<>{} != Dimensions{5});
  static_assert(not (Spherical<>{} > Dimensions{5}));
}

