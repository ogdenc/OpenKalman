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
 * \brief Tests for coordinates::Inclination
 */

#include "collections/tests/tests.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/functions/get_stat_dimension.hpp"
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"
#include "linear-algebra/coordinates/traits/stat_dimension_of.hpp"
#include "linear-algebra/coordinates/descriptors/Inclination.hpp"

using namespace OpenKalman;
using namespace OpenKalman::coordinates;

using OpenKalman::stdcompat::numbers::pi;

TEST(coordinates, Inclination)
{
  static_assert(descriptor<inclination::Radians>);
  static_assert(fixed_pattern<inclination::Radians>);
  static_assert(pattern<inclination::Degrees>);
  static_assert(not euclidean_pattern<inclination::Radians>);

  static_assert(get_dimension(inclination::Radians{}) == 1);
  static_assert(get_stat_dimension(inclination::Radians{}) == 2);
  static_assert(not get_is_euclidean(inclination::Radians{}));
  static_assert(dimension_of_v<inclination::Radians> == 1);
  static_assert(stat_dimension_of_v<inclination::Radians> == 2);

  static_assert(std::is_same_v<std::common_type_t<inclination::Degrees, Inclination<std::integral_constant<int, 180>>>, inclination::Degrees>);
}

#include "linear-algebra/coordinates/functions/to_stat_space.hpp"
#include "linear-algebra/coordinates/functions/from_stat_space.hpp"
#include "linear-algebra/coordinates/functions/wrap.hpp"

namespace
{
  constexpr double sqrt3_2 = values::sqrt(3.)/2;
}

TEST(coordinates, Inclination_transformations)
{
  static_assert(values::internal::near(to_stat_space(inclination::Radians{}, std::array{pi/3})[0U], 0.5, 1e-6));
  static_assert(values::internal::near(to_stat_space(inclination::Degrees{}, std::array{-210})[1U], 0.5, 1e-6));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(to_stat_space(inclination::Degrees{}, std::tuple{values::Fixed<double, -210>{}}).get<1>())>, 0.5, 1e-6));

  EXPECT_TRUE(test::is_near(to_stat_space(inclination::Radians{}, std::vector{pi/3}), std::array{0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(inclination::Radians{}, std::array{-pi/3}), std::array{0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(inclination::Radians{}, std::vector{pi/6}), std::array{sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(inclination::Radians{}, std::array{-pi/6}), std::array{sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(inclination::Radians{}, std::array{7*pi/6}), std::array{-sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(inclination::Radians{}, std::array{-7*pi/6}), std::array{-sqrt3_2, 0.5}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(inclination::Degrees{}, std::vector{-60}), std::array{0.5, sqrt3_2}, 1e-6));
  EXPECT_TRUE(test::is_near(to_stat_space(inclination::Degrees{}, std::array{-210}), std::array{-sqrt3_2, 0.5}, 1e-6));

  static_assert(values::internal::near(from_stat_space(inclination::Radians{}, std::array{0.5, sqrt3_2})[0U], pi/3, 1e-6));
  static_assert(values::internal::near(from_stat_space(inclination::Radians{}, std::array{0.5, -sqrt3_2})[0U], pi/3, 1e-6));
  static_assert(values::internal::near(from_stat_space(inclination::Degrees{}, std::array{-0.5, sqrt3_2})[0U], 120, 1e-6));
  static_assert(values::internal::near(from_stat_space(inclination::Degrees{}, std::array{-0.5, -sqrt3_2})[0U], 120, 1e-6));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(from_stat_space(inclination::Degrees{}, std::tuple{values::Fixed<double, -1>{}, values::Fixed<double, 1>{}}).get<0>())>, 135, 1e-6));

  EXPECT_TRUE(test::is_near(from_stat_space(inclination::Radians{}, std::vector{0.5, sqrt3_2}), std::array{pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(inclination::Radians{}, std::vector{0.5, -sqrt3_2}), std::array{pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(inclination::Radians{}, std::vector{-0.5, sqrt3_2}), std::array{2*pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(inclination::Radians{}, std::vector{-0.5, -sqrt3_2}), std::array{2*pi/3}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(inclination::Degrees{}, std::vector{0.5, sqrt3_2}), std::array{60}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(inclination::Degrees{}, std::vector{0.5, -sqrt3_2}), std::array{60}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(inclination::Degrees{}, std::vector{-0.5, sqrt3_2}), std::array{120}, 1e-6));
  EXPECT_TRUE(test::is_near(from_stat_space(inclination::Degrees{}, std::vector{-0.5, -sqrt3_2}), std::array{120}, 1e-6));

  static_assert(values::internal::near(wrap(inclination::Radians{}, std::array{1.2*pi})[0U], 0.8*pi, 1e-6));
  static_assert(values::internal::near(wrap(inclination::Radians{}, std::array{-2.2*pi})[0U], 0.2*pi, 1e-6));
  static_assert(values::internal::near(wrap(inclination::Degrees{}, std::array{200})[0U], 160, 1e-6));
  static_assert(values::internal::near(wrap(inclination::Degrees{}, std::array{-380})[0U], 20, 1e-6));

  static_assert(values::internal::near(values::fixed_number_of_v<decltype(wrap(inclination::Degrees{}, std::array{values::Fixed<double, 200>{}}).get<0>())>, 160, 1e-6));

  EXPECT_TRUE(test::is_near(wrap(inclination::Radians{}, std::vector{0.7*pi}), std::array{0.7*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(inclination::Radians{}, std::vector{-0.7*pi}), std::array{0.7*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(inclination::Radians{}, std::vector{1.2*pi}), std::array{0.8*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(inclination::Radians{}, std::vector{-1.2*pi}), std::array{0.8*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(inclination::Radians{}, std::vector{2.2*pi}), std::array{0.2*pi}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(inclination::Radians{}, std::vector{-2.2*pi}), std::array{0.2*pi}, 1e-6));

  EXPECT_TRUE(test::is_near(wrap(inclination::Degrees{}, std::vector{110}), std::array{110}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(inclination::Degrees{}, std::vector{-110}), std::array{110}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(inclination::Degrees{}, std::vector{200}), std::array{160}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(inclination::Degrees{}, std::vector{-200}), std::array{160}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(inclination::Degrees{}, std::vector{380}), std::array{20}, 1e-6));
  EXPECT_TRUE(test::is_near(wrap(inclination::Degrees{}, std::vector{-380}), std::array{20}, 1e-6));
}
